"""
Signal Engine Module for Polymarket Tracker.

Evaluates incoming bets against three signal types and produces a composite
conviction score to identify high-conviction trading opportunities.

Signal types:
- Smart Money: Grades whale wallets and weights their bets by historical accuracy.
- Volume Spike: Detects bets that are abnormally large relative to recent market activity.
- Cluster: Identifies coordinated buying from multiple wallets in a short window.

A composite score is computed as a weighted sum of the three signals.
Signals that exceed the conviction threshold are persisted and returned.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from .database import Database, db

logger = logging.getLogger(__name__)


@dataclass
class SignalConfig:
    """Configuration for signal evaluation thresholds and weights."""

    # Composite weights (must sum to 1.0)
    WEIGHT_SMART_MONEY: float = 0.4
    WEIGHT_VOLUME_SPIKE: float = 0.3
    WEIGHT_CLUSTER: float = 0.3

    # Signal fire threshold
    CONVICTION_THRESHOLD: float = 60.0

    # Smart money config
    GRADE_SCORES: dict = None  # {"A": 90, "B": 70, "C": 40, "D": 20, "ungraded": 0}

    # Volume spike config
    SPIKE_MULTIPLIER: float = 3.0       # Bet must be Nx above rolling avg
    ROLLING_WINDOW_DAYS: int = 7

    # Cluster config
    CLUSTER_WINDOW_MINUTES: int = 30
    CLUSTER_MIN_WALLETS: int = 3

    # Deduplication: suppress duplicate signals for same market+outcome
    DEDUP_WINDOW_MINUTES: int = 60

    def __post_init__(self):
        if self.GRADE_SCORES is None:
            self.GRADE_SCORES = {"A": 90, "B": 70, "C": 40, "D": 20, "ungraded": 0}


class SignalEngine:
    """
    Evaluates bets against smart money, volume spike, and cluster signals
    to produce composite conviction scores for trading opportunities.
    """

    def __init__(self, database: Optional[Database] = None, config: Optional[SignalConfig] = None):
        """
        Initialize the signal engine.

        Args:
            database: Database instance to use. Defaults to the global db.
            config: Signal configuration. Defaults to SignalConfig().
        """
        self.db = database or db
        self.config = config or SignalConfig()

    def score_smart_money(self, wallet_address: str, bet_amount: float, market_id: str) -> float:
        """
        Score a bet based on the whale profile of the wallet.

        Returns 0-100. Looks up whale profile from DB. If no profile or
        "ungraded", returns 0. Base score from GRADE_SCORES[grade]. Boost +10
        (capped at 100) if whale's category specialization win rate >= 65%
        for this market's category.

        Args:
            wallet_address: The wallet address placing the bet.
            bet_amount: The bet amount in USD.
            market_id: The market being bet on.

        Returns:
            Score from 0 to 100.
        """
        profile = self.db.get_whale_profile(wallet_address)
        if profile is None:
            return 0.0

        grade = profile.get("grade", "ungraded")
        if grade == "ungraded":
            return 0.0

        base_score = float(self.config.GRADE_SCORES.get(grade, 0))

        # Check for category specialization boost
        market = self.db.get_market(market_id)
        if market and market.get("category"):
            cat_spec_raw = profile.get("category_specialization")
            if cat_spec_raw:
                try:
                    cat_spec = json.loads(cat_spec_raw) if isinstance(cat_spec_raw, str) else cat_spec_raw
                    market_category = market["category"]
                    category_win_rate = cat_spec.get(market_category, 0.0)
                    if category_win_rate >= 65.0:
                        base_score = min(base_score + 10.0, 100.0)
                except (json.JSONDecodeError, TypeError):
                    pass

        return base_score

    def score_volume_spike(self, market_id: str, bet_usd: float) -> float:
        """
        Score a bet based on how much it spikes above recent market activity.

        Returns 0-100. Queries the bets table for this market's average bet
        USD value (amount * price) over the last ROLLING_WINDOW_DAYS. If fewer
        than 5 bets or avg <= 0, returns 0. Calculates ratio = bet_usd / avg.
        If ratio < SPIKE_MULTIPLIER (3x), returns 0. Scale: 3x=50, then +10
        per additional multiplier, capped at 100.

        Args:
            market_id: The market to check.
            bet_usd: The bet value in USD (amount * price).

        Returns:
            Score from 0 to 100.
        """
        cutoff = datetime.utcnow() - timedelta(days=self.config.ROLLING_WINDOW_DAYS)

        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT COUNT(*) as bet_count, AVG(amount * price) as avg_amount
                FROM bets
                WHERE market_id = ? AND timestamp >= ?
                """,
                (market_id, cutoff)
            )
            row = cursor.fetchone()

        bet_count = row["bet_count"] if row else 0
        avg_amount = row["avg_amount"] if row and row["avg_amount"] else 0.0

        if bet_count < 5 or avg_amount <= 0:
            return 0.0

        ratio = bet_usd / avg_amount
        if ratio < self.config.SPIKE_MULTIPLIER:
            return 0.0

        # Scale: 3x=50, then +10 per additional multiplier, capped at 100
        score = 50.0 + (ratio - self.config.SPIKE_MULTIPLIER) * 10.0
        return min(score, 100.0)

    def score_cluster(self, market_id: str, outcome: str, timestamp: datetime) -> float:
        """
        Score based on coordinated buying activity from multiple wallets.

        Returns 0-100. Counts distinct wallet_addresses betting BUY on the
        same outcome within +/- CLUSTER_WINDOW_MINUTES of timestamp. Compares
        against the baseline rate of distinct wallets for this market to
        avoid false positives on popular markets. The effective minimum is
        max(CLUSTER_MIN_WALLETS, baseline_rate * SPIKE_MULTIPLIER).

        Args:
            market_id: The market to check.
            outcome: The outcome being bet on (e.g. "Yes" or "No").
            timestamp: The timestamp of the current bet.

        Returns:
            Score from 0 to 100.
        """
        window = timedelta(minutes=self.config.CLUSTER_WINDOW_MINUTES)
        start_time = timestamp - window
        end_time = timestamp + window
        rolling_cutoff = timestamp - timedelta(days=self.config.ROLLING_WINDOW_DAYS)

        with self.db.get_connection() as conn:
            # Count distinct wallets in the cluster window
            cursor = conn.execute(
                """
                SELECT COUNT(DISTINCT wallet_address) as wallet_count
                FROM bets
                WHERE market_id = ?
                  AND outcome_bet = ?
                  AND side = 'BUY'
                  AND timestamp >= ?
                  AND timestamp <= ?
                """,
                (market_id, outcome, start_time, end_time)
            )
            row = cursor.fetchone()
            wallet_count = row["wallet_count"] if row else 0

            # Calculate baseline: total distinct wallets in the rolling window
            cursor = conn.execute(
                """
                SELECT COUNT(DISTINCT wallet_address) as total_wallets
                FROM bets
                WHERE market_id = ?
                  AND outcome_bet = ?
                  AND side = 'BUY'
                  AND timestamp >= ?
                """,
                (market_id, outcome, rolling_cutoff)
            )
            baseline_row = cursor.fetchone()

        total_wallets = baseline_row["total_wallets"] if baseline_row else 0

        # Calculate dynamic minimum based on market activity.
        # Assume uniform distribution of wallets across the rolling window.
        # If the cluster window contains significantly more wallets than
        # expected by chance, it's a genuine cluster.
        if total_wallets > 0:
            rolling_window_hours = self.config.ROLLING_WINDOW_DAYS * 24
            cluster_window_hours = (self.config.CLUSTER_WINDOW_MINUTES * 2) / 60  # +/- window
            window_fraction = cluster_window_hours / rolling_window_hours
            expected_if_uniform = total_wallets * window_fraction
            # Need at least 3x the expected wallets for this to count as a cluster
            dynamic_min = max(
                self.config.CLUSTER_MIN_WALLETS,
                int(expected_if_uniform * self.config.SPIKE_MULTIPLIER) + 1,
            )
        else:
            dynamic_min = self.config.CLUSTER_MIN_WALLETS

        if wallet_count < dynamic_min:
            return 0.0

        # Scale: threshold = 50, +12.5 per additional wallet above threshold, capped at 100
        score = 50.0 + (wallet_count - dynamic_min) * 12.5
        return min(score, 100.0)

    def calculate_composite_score(
        self, smart_money: float, volume_spike: float, cluster: float
    ) -> float:
        """
        Calculate the weighted composite conviction score.

        Args:
            smart_money: Smart money signal score (0-100).
            volume_spike: Volume spike signal score (0-100).
            cluster: Cluster signal score (0-100).

        Returns:
            Weighted composite score.
        """
        return (
            smart_money * self.config.WEIGHT_SMART_MONEY
            + volume_spike * self.config.WEIGHT_VOLUME_SPIKE
            + cluster * self.config.WEIGHT_CLUSTER
        )

    def evaluate_bet(
        self,
        wallet_address: str,
        market_id: str,
        outcome: str,
        amount: float,
        price: float,
        timestamp: datetime,
    ) -> Optional[dict]:
        """
        Evaluate a single bet against all three signal types.

        Calls all three scoring methods, calculates the composite score.
        If below CONVICTION_THRESHOLD, returns None. Otherwise, builds a
        details dict, calls db.insert_signal, logs it, and returns the
        signal dict.

        Args:
            wallet_address: The wallet address placing the bet.
            market_id: The market being bet on.
            outcome: The outcome being bet on.
            amount: The bet amount in shares.
            price: The price per share.
            timestamp: The bet timestamp.

        Returns:
            Signal dict if conviction is above threshold, else None.
        """
        bet_usd = amount * price
        sm_score = self.score_smart_money(wallet_address, bet_usd, market_id)
        vs_score = self.score_volume_spike(market_id, bet_usd)
        cl_score = self.score_cluster(market_id, outcome, timestamp)

        composite = self.calculate_composite_score(sm_score, vs_score, cl_score)

        if composite < self.config.CONVICTION_THRESHOLD:
            return None

        # Deduplication: skip if a signal already exists for this market+outcome recently
        dedup_cutoff = datetime.utcnow() - timedelta(minutes=self.config.DEDUP_WINDOW_MINUTES)
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT COUNT(*) as cnt FROM signals
                WHERE market_id = ? AND outcome = ? AND timestamp >= ?
                """,
                (market_id, outcome, dedup_cutoff)
            )
            row = cursor.fetchone()
            if row and row["cnt"] > 0:
                logger.debug(
                    "Signal suppressed (dedup): market=%s outcome=%s already signaled within %dm",
                    market_id, outcome, self.config.DEDUP_WINDOW_MINUTES,
                )
                return None

        details = {
            "wallet_address": wallet_address,
            "bet_amount_usd": bet_usd,
            "bet_shares": amount,
            "bet_price": price,
            "smart_money_score": sm_score,
            "volume_spike_score": vs_score,
            "cluster_score": cl_score,
            "composite_score": composite,
        }

        signal_id = self.db.insert_signal(
            market_id=market_id,
            outcome=outcome,
            conviction_score=composite,
            smart_money_score=sm_score,
            volume_spike_score=vs_score,
            cluster_score=cl_score,
            contributing_wallets=json.dumps([wallet_address]),
            details=json.dumps(details),
        )

        logger.info(
            "Signal fired: market=%s outcome=%s conviction=%.1f "
            "(sm=%.1f vs=%.1f cl=%.1f) signal_id=%d",
            market_id, outcome, composite, sm_score, vs_score, cl_score, signal_id,
        )

        return {
            "signal_id": signal_id,
            "market_id": market_id,
            "outcome": outcome,
            "conviction_score": composite,
            "smart_money_score": sm_score,
            "volume_spike_score": vs_score,
            "cluster_score": cl_score,
            "wallet_address": wallet_address,
            "details": details,
        }

    def evaluate_recent_bets(self, since_minutes: int = 5) -> list[dict]:
        """
        Query recent BUY bets and evaluate each for signals.

        Args:
            since_minutes: Look back this many minutes for new bets.

        Returns:
            List of signal dicts that fired (above conviction threshold).
        """
        cutoff = datetime.utcnow() - timedelta(minutes=since_minutes)
        signals = []

        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT wallet_address, market_id, outcome_bet, amount, price, timestamp
                FROM bets
                WHERE side = 'BUY' AND timestamp >= ?
                ORDER BY timestamp ASC
                """,
                (cutoff,)
            )
            rows = cursor.fetchall()

        for row in rows:
            bet_timestamp = row["timestamp"]
            if isinstance(bet_timestamp, str):
                bet_timestamp = datetime.fromisoformat(bet_timestamp)

            result = self.evaluate_bet(
                wallet_address=row["wallet_address"],
                market_id=row["market_id"],
                outcome=row["outcome_bet"],
                amount=row["amount"],
                price=row["price"],
                timestamp=bet_timestamp,
            )
            if result is not None:
                signals.append(result)

        logger.info(
            "Evaluated %d recent bets, %d signals fired",
            len(rows), len(signals),
        )
        return signals
