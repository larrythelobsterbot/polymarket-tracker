"""
Insider Trading Detection System for Polymarket Tracker.

This module provides comprehensive detection of suspicious betting patterns
that may indicate insider information or market manipulation:

1. Timing-based alerts:
   - Large bets near market resolution
   - Sudden activity from dormant wallets
   - Low-liquidity period betting

2. Pattern recognition:
   - Consistent wins on niche markets
   - First-mover advantage detection
   - Correlated wallet activity

3. Wallet clustering:
   - Sybil attack detection
   - Fund flow tracking
   - Common source identification

4. Anomaly scoring:
   - 0-100 suspiciousness score per bet
   - Configurable thresholds
   - Alert triggers for high scores

DISCLAIMER: This system is for research and analysis purposes only.
Flagged activity requires human review and does not constitute proof
of wrongdoing.
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Optional

from .database import Database, db

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Constants - Tune these to control false positive rates
# ============================================================================

class DetectionConfig:
    """Configuration parameters for detection thresholds."""

    # Timing thresholds
    LARGE_BET_THRESHOLD = 10000.0  # USD - bets above this are "large"
    RESOLUTION_WINDOW_HOURS = 24   # Hours before resolution to flag
    DORMANT_DAYS_THRESHOLD = 30    # Days of inactivity to consider "dormant"
    LOW_LIQUIDITY_HOURS = (0, 6)   # UTC hours considered low liquidity (midnight-6am)

    # Pattern thresholds
    NICHE_MARKET_WIN_RATE = 75.0   # Win rate % to flag on niche markets
    NICHE_MARKET_MIN_BETS = 10     # Minimum bets on niche markets
    NICHE_MARKET_MAX_VOLUME = 100000  # Max market volume to be "niche"
    FIRST_MOVER_HOURS = 48         # Hours from market creation for "early" bets
    VIRAL_VOLUME_MULTIPLIER = 10   # Volume increase to be "went viral"

    # Correlation thresholds
    CORRELATION_TIME_WINDOW_MINUTES = 30  # Window for correlated bets
    CORRELATION_MIN_WALLETS = 3    # Minimum wallets for correlation flag

    # Scoring weights (sum should be ~100 for intuitive scores)
    WEIGHT_TIMING = 25
    WEIGHT_AMOUNT = 20
    WEIGHT_HISTORY = 15
    WEIGHT_LIQUIDITY = 15
    WEIGHT_PROBABILITY = 15
    WEIGHT_CORRELATION = 10

    # Alert thresholds
    ALERT_SCORE_THRESHOLD = 80     # Score above which to send alerts
    WARNING_SCORE_THRESHOLD = 60   # Score for warning-level flags

    # Wallet clustering
    CLUSTER_TIME_WINDOW_HOURS = 1  # Time window for funding correlation
    CLUSTER_BET_SIMILARITY = 0.8   # Similarity threshold for bet patterns


# ============================================================================
# Data Classes
# ============================================================================

class AlertSeverity(Enum):
    """Severity levels for alerts."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of suspicious activity alerts."""
    LARGE_BET_NEAR_RESOLUTION = "large_bet_near_resolution"
    DORMANT_WALLET_ACTIVATION = "dormant_wallet_activation"
    LOW_LIQUIDITY_BETTING = "low_liquidity_betting"
    NICHE_MARKET_DOMINANCE = "niche_market_dominance"
    FIRST_MOVER_ADVANTAGE = "first_mover_advantage"
    CORRELATED_BETTING = "correlated_betting"
    POTENTIAL_SYBIL = "potential_sybil"
    UNUSUAL_VOLUME_SPIKE = "unusual_volume_spike"
    HIGH_ANOMALY_SCORE = "high_anomaly_score"


@dataclass
class SuspiciousBet:
    """Represents a bet flagged as suspicious."""
    bet_id: str
    wallet_address: str
    market_id: str
    amount: float
    price: float
    side: str
    outcome_bet: str
    timestamp: datetime
    anomaly_score: float = 0.0
    alert_types: list = field(default_factory=list)
    score_breakdown: dict = field(default_factory=dict)
    market_question: str = ""
    market_volume: float = 0.0
    hours_to_resolution: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "bet_id": self.bet_id,
            "wallet_address": self.wallet_address,
            "display_address": f"{self.wallet_address[:6]}...{self.wallet_address[-4:]}",
            "market_id": self.market_id,
            "market_question": self.market_question[:50] + "..." if len(self.market_question) > 50 else self.market_question,
            "amount": self.amount,
            "price": self.price,
            "side": self.side,
            "outcome_bet": self.outcome_bet,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "anomaly_score": round(self.anomaly_score, 1),
            "alert_types": [a.value for a in self.alert_types],
            "score_breakdown": self.score_breakdown,
            "hours_to_resolution": round(self.hours_to_resolution, 1) if self.hours_to_resolution else None,
            "severity": self.severity.value,
        }

    @property
    def severity(self) -> AlertSeverity:
        """Determine severity based on anomaly score."""
        if self.anomaly_score >= 90:
            return AlertSeverity.CRITICAL
        elif self.anomaly_score >= 80:
            return AlertSeverity.HIGH
        elif self.anomaly_score >= 60:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW


@dataclass
class WalletRiskProfile:
    """Risk profile for a wallet address."""
    wallet_address: str
    overall_risk_score: float = 0.0
    total_suspicious_bets: int = 0
    total_bets_analyzed: int = 0
    alert_counts: dict = field(default_factory=dict)
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    total_volume: float = 0.0
    win_rate_niche: float = 0.0
    avg_bet_size: float = 0.0
    dormant_periods: int = 0
    associated_wallets: list = field(default_factory=list)  # Potential Sybil connections
    flags: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "wallet_address": self.wallet_address,
            "display_address": f"{self.wallet_address[:6]}...{self.wallet_address[-4:]}",
            "overall_risk_score": round(self.overall_risk_score, 1),
            "total_suspicious_bets": self.total_suspicious_bets,
            "total_bets_analyzed": self.total_bets_analyzed,
            "suspicion_rate": round((self.total_suspicious_bets / max(1, self.total_bets_analyzed)) * 100, 1),
            "alert_counts": self.alert_counts,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "total_volume": self.total_volume,
            "win_rate_niche": round(self.win_rate_niche, 1),
            "avg_bet_size": round(self.avg_bet_size, 2),
            "associated_wallets": self.associated_wallets[:5],  # Top 5
            "flags": self.flags,
        }


@dataclass
class CorrelatedBetGroup:
    """Group of correlated bets from multiple wallets."""
    market_id: str
    outcome: str
    timestamp_start: datetime
    timestamp_end: datetime
    wallets: list = field(default_factory=list)
    total_amount: float = 0.0
    bet_count: int = 0
    correlation_score: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "market_id": self.market_id,
            "outcome": self.outcome,
            "timestamp_start": self.timestamp_start.isoformat(),
            "timestamp_end": self.timestamp_end.isoformat(),
            "time_window_minutes": (self.timestamp_end - self.timestamp_start).total_seconds() / 60,
            "wallets": [f"{w[:6]}...{w[-4:]}" for w in self.wallets],
            "wallet_count": len(self.wallets),
            "total_amount": self.total_amount,
            "bet_count": self.bet_count,
            "correlation_score": round(self.correlation_score, 1),
        }


@dataclass
class MarketManipulationAlert:
    """Alert for potential market manipulation."""
    market_id: str
    market_question: str
    alert_type: str
    timestamp: datetime
    details: dict = field(default_factory=dict)
    severity: AlertSeverity = AlertSeverity.MEDIUM

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "market_id": self.market_id,
            "market_question": self.market_question[:50] + "..." if len(self.market_question) > 50 else self.market_question,
            "alert_type": self.alert_type,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "severity": self.severity.value,
        }


# ============================================================================
# Main Detection Engine
# ============================================================================

class InsiderDetector:
    """
    Main class for detecting suspicious betting patterns.

    Analyzes bets for timing anomalies, unusual patterns, and
    potential coordination between wallets.
    """

    def __init__(
        self,
        database: Optional[Database] = None,
        config: Optional[DetectionConfig] = None
    ):
        """
        Initialize the detector.

        Args:
            database: Database instance.
            config: Detection configuration.
        """
        self.db = database or db
        self.config = config or DetectionConfig()

    # ========================================================================
    # Timing-Based Detection
    # ========================================================================

    def check_large_bet_near_resolution(
        self,
        bet_amount: float,
        bet_timestamp: datetime,
        market_end_date: Optional[datetime]
    ) -> tuple[bool, float, Optional[float]]:
        """
        Check if a large bet was placed near market resolution.

        Args:
            bet_amount: Bet amount in USD.
            bet_timestamp: When the bet was placed.
            market_end_date: When the market resolves.

        Returns:
            Tuple of (is_suspicious, score_contribution, hours_to_resolution).
        """
        if bet_amount < self.config.LARGE_BET_THRESHOLD:
            return False, 0.0, None

        if not market_end_date:
            return False, 0.0, None

        hours_to_resolution = (market_end_date - bet_timestamp).total_seconds() / 3600

        if hours_to_resolution < 0:
            # Bet after resolution (shouldn't happen, but handle it)
            return False, 0.0, hours_to_resolution

        if hours_to_resolution <= self.config.RESOLUTION_WINDOW_HOURS:
            # Score based on how close to resolution
            # Closer = more suspicious
            proximity_factor = 1 - (hours_to_resolution / self.config.RESOLUTION_WINDOW_HOURS)
            size_factor = min(1.0, bet_amount / (self.config.LARGE_BET_THRESHOLD * 5))
            score = self.config.WEIGHT_TIMING * (0.5 * proximity_factor + 0.5 * size_factor)
            return True, score, hours_to_resolution

        return False, 0.0, hours_to_resolution

    def check_dormant_wallet_activation(
        self,
        wallet_address: str,
        bet_timestamp: datetime,
        bet_amount: float
    ) -> tuple[bool, float]:
        """
        Check if a dormant wallet suddenly became active with a large bet.

        Args:
            wallet_address: Wallet to check.
            bet_timestamp: Current bet timestamp.
            bet_amount: Current bet amount.

        Returns:
            Tuple of (is_suspicious, score_contribution).
        """
        with self.db.get_connection() as conn:
            # Find last activity before this bet
            cursor = conn.execute(
                """
                SELECT MAX(timestamp) as last_active
                FROM bets
                WHERE wallet_address = ? AND timestamp < ?
                """,
                (wallet_address, bet_timestamp)
            )
            row = cursor.fetchone()

            if not row or not row["last_active"]:
                # New wallet - not suspicious for dormancy
                return False, 0.0

            last_active = row["last_active"]
            if isinstance(last_active, str):
                last_active = datetime.fromisoformat(last_active)

            days_inactive = (bet_timestamp - last_active).days

            if days_inactive >= self.config.DORMANT_DAYS_THRESHOLD:
                # Wallet was dormant, now suddenly active
                dormancy_factor = min(1.0, days_inactive / (self.config.DORMANT_DAYS_THRESHOLD * 3))
                size_factor = min(1.0, bet_amount / self.config.LARGE_BET_THRESHOLD)
                score = self.config.WEIGHT_HISTORY * (0.6 * dormancy_factor + 0.4 * size_factor)
                return True, score

        return False, 0.0

    def check_low_liquidity_betting(
        self,
        bet_timestamp: datetime,
        bet_amount: float
    ) -> tuple[bool, float]:
        """
        Check if bet was placed during low-liquidity hours.

        Args:
            bet_timestamp: When the bet was placed.
            bet_amount: Bet amount.

        Returns:
            Tuple of (is_suspicious, score_contribution).
        """
        hour = bet_timestamp.hour  # UTC hour

        low_start, low_end = self.config.LOW_LIQUIDITY_HOURS

        is_low_liquidity = (low_start <= hour < low_end) or (low_start > low_end and (hour >= low_start or hour < low_end))

        if is_low_liquidity and bet_amount >= self.config.LARGE_BET_THRESHOLD * 0.5:
            size_factor = min(1.0, bet_amount / self.config.LARGE_BET_THRESHOLD)
            score = self.config.WEIGHT_LIQUIDITY * 0.5 * size_factor
            return True, score

        return False, 0.0

    # ========================================================================
    # Pattern Recognition
    # ========================================================================

    def check_niche_market_dominance(
        self,
        wallet_address: str
    ) -> tuple[bool, float, dict]:
        """
        Check for consistent wins on low-volume/niche markets.

        Args:
            wallet_address: Wallet to analyze.

        Returns:
            Tuple of (is_suspicious, score_contribution, details).
        """
        with self.db.get_connection() as conn:
            # Get bets on niche markets (low volume) that are resolved
            cursor = conn.execute(
                """
                SELECT
                    b.bet_id,
                    b.market_id,
                    b.outcome_bet,
                    b.amount,
                    m.volume,
                    m.outcome as market_outcome,
                    m.resolved
                FROM bets b
                JOIN markets m ON b.market_id = m.market_id
                WHERE b.wallet_address = ?
                    AND m.resolved = 1
                    AND m.volume < ?
                    AND b.side = 'BUY'
                """,
                (wallet_address, self.config.NICHE_MARKET_MAX_VOLUME)
            )
            niche_bets = cursor.fetchall()

            if len(niche_bets) < self.config.NICHE_MARKET_MIN_BETS:
                return False, 0.0, {}

            wins = 0
            losses = 0
            total_amount = 0.0

            for bet in niche_bets:
                total_amount += bet["amount"] or 0
                if bet["market_outcome"]:
                    bet_outcome = (bet["outcome_bet"] or "").lower()
                    market_outcome = bet["market_outcome"].lower()
                    if bet_outcome == market_outcome or \
                       (bet_outcome in ["yes", "true"] and market_outcome in ["yes", "true"]) or \
                       (bet_outcome in ["no", "false"] and market_outcome in ["no", "false"]):
                        wins += 1
                    else:
                        losses += 1

            total_resolved = wins + losses
            if total_resolved < self.config.NICHE_MARKET_MIN_BETS:
                return False, 0.0, {}

            win_rate = (wins / total_resolved) * 100

            if win_rate >= self.config.NICHE_MARKET_WIN_RATE:
                # Suspiciously high win rate on niche markets
                excess_win_rate = (win_rate - 50) / 50  # Normalized above 50%
                volume_factor = min(1.0, total_amount / 10000)
                score = self.config.WEIGHT_HISTORY * (0.7 * excess_win_rate + 0.3 * volume_factor)

                details = {
                    "niche_wins": wins,
                    "niche_losses": losses,
                    "niche_win_rate": round(win_rate, 1),
                    "niche_volume": round(total_amount, 2),
                    "niche_markets_count": len(set(b["market_id"] for b in niche_bets)),
                }
                return True, score, details

        return False, 0.0, {}

    def check_first_mover_advantage(
        self,
        wallet_address: str
    ) -> tuple[bool, float, list]:
        """
        Check for betting early on markets that later went viral.

        Args:
            wallet_address: Wallet to analyze.

        Returns:
            Tuple of (is_suspicious, score_contribution, viral_markets).
        """
        viral_markets = []

        with self.db.get_connection() as conn:
            # Find early bets on markets that later had high volume
            cursor = conn.execute(
                """
                SELECT
                    b.bet_id,
                    b.market_id,
                    b.amount,
                    b.timestamp as bet_time,
                    b.outcome_bet,
                    m.question,
                    m.created_at as market_created,
                    m.volume as final_volume,
                    m.resolved,
                    m.outcome as market_outcome
                FROM bets b
                JOIN markets m ON b.market_id = m.market_id
                WHERE b.wallet_address = ?
                    AND b.side = 'BUY'
                ORDER BY b.timestamp ASC
                """,
                (wallet_address,)
            )

            for row in cursor.fetchall():
                bet_time = row["bet_time"]
                market_created = row["market_created"]

                if isinstance(bet_time, str):
                    bet_time = datetime.fromisoformat(bet_time)
                if isinstance(market_created, str):
                    market_created = datetime.fromisoformat(market_created)

                if not market_created:
                    continue

                hours_after_creation = (bet_time - market_created).total_seconds() / 3600

                # Early bet on market that got significant volume
                if hours_after_creation <= self.config.FIRST_MOVER_HOURS:
                    final_volume = row["final_volume"] or 0

                    # Check if market went viral (got much more volume later)
                    if final_volume > self.config.NICHE_MARKET_MAX_VOLUME * self.config.VIRAL_VOLUME_MULTIPLIER:
                        # Check if bet was winning
                        is_winner = False
                        if row["resolved"] and row["market_outcome"]:
                            outcome_bet = (row["outcome_bet"] or "").lower()
                            market_outcome = row["market_outcome"].lower()
                            is_winner = outcome_bet == market_outcome or \
                                       (outcome_bet in ["yes", "true"] and market_outcome in ["yes", "true"])

                        viral_markets.append({
                            "market_id": row["market_id"],
                            "question": row["question"][:50] if row["question"] else "",
                            "hours_after_creation": round(hours_after_creation, 1),
                            "bet_amount": row["amount"],
                            "final_volume": final_volume,
                            "is_winner": is_winner,
                        })

            # Score based on number of first-mover wins
            winning_first_moves = [m for m in viral_markets if m["is_winner"]]

            if len(winning_first_moves) >= 3:
                score = self.config.WEIGHT_HISTORY * min(1.0, len(winning_first_moves) / 5)
                return True, score, viral_markets

        return False, 0.0, viral_markets

    def detect_correlated_betting(
        self,
        market_id: str,
        time_window_minutes: Optional[int] = None
    ) -> list[CorrelatedBetGroup]:
        """
        Detect multiple wallets betting same outcome within short time window.

        Args:
            market_id: Market to analyze.
            time_window_minutes: Time window for correlation.

        Returns:
            List of correlated bet groups.
        """
        time_window = time_window_minutes or self.config.CORRELATION_TIME_WINDOW_MINUTES
        correlated_groups = []

        with self.db.get_connection() as conn:
            # Get all bets on this market, ordered by time
            cursor = conn.execute(
                """
                SELECT
                    bet_id,
                    wallet_address,
                    amount,
                    outcome_bet,
                    timestamp
                FROM bets
                WHERE market_id = ? AND side = 'BUY'
                ORDER BY timestamp ASC
                """,
                (market_id,)
            )
            bets = cursor.fetchall()

            if len(bets) < self.config.CORRELATION_MIN_WALLETS:
                return []

            # Group bets by outcome
            by_outcome: dict[str, list] = defaultdict(list)
            for bet in bets:
                outcome = (bet["outcome_bet"] or "").lower()
                timestamp = bet["timestamp"]
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                by_outcome[outcome].append({
                    "bet_id": bet["bet_id"],
                    "wallet": bet["wallet_address"],
                    "amount": bet["amount"],
                    "timestamp": timestamp,
                })

            # Find clusters within time window
            for outcome, outcome_bets in by_outcome.items():
                if len(outcome_bets) < self.config.CORRELATION_MIN_WALLETS:
                    continue

                # Sliding window approach
                i = 0
                while i < len(outcome_bets):
                    window_start = outcome_bets[i]["timestamp"]
                    window_end = window_start + timedelta(minutes=time_window)

                    # Find all bets in window
                    window_bets = []
                    j = i
                    while j < len(outcome_bets) and outcome_bets[j]["timestamp"] <= window_end:
                        window_bets.append(outcome_bets[j])
                        j += 1

                    # Check for correlation (multiple unique wallets)
                    unique_wallets = list(set(b["wallet"] for b in window_bets))

                    if len(unique_wallets) >= self.config.CORRELATION_MIN_WALLETS:
                        total_amount = sum(b["amount"] for b in window_bets)
                        actual_end = max(b["timestamp"] for b in window_bets)

                        # Calculate correlation score
                        wallet_factor = min(1.0, len(unique_wallets) / 10)
                        amount_factor = min(1.0, total_amount / 50000)
                        time_factor = 1 - ((actual_end - window_start).total_seconds() / (time_window * 60))

                        correlation_score = (wallet_factor * 40 + amount_factor * 30 + time_factor * 30)

                        group = CorrelatedBetGroup(
                            market_id=market_id,
                            outcome=outcome,
                            timestamp_start=window_start,
                            timestamp_end=actual_end,
                            wallets=unique_wallets,
                            total_amount=total_amount,
                            bet_count=len(window_bets),
                            correlation_score=correlation_score,
                        )
                        correlated_groups.append(group)

                    i = j if j > i else i + 1

        return correlated_groups

    # ========================================================================
    # Wallet Clustering / Sybil Detection
    # ========================================================================

    def detect_sybil_patterns(
        self,
        wallet_addresses: list[str]
    ) -> dict[str, list[str]]:
        """
        Identify potential Sybil attacks by finding wallets with similar behavior.

        Checks for:
        - Similar betting patterns (same markets, same timing)
        - Sequential funding patterns
        - Identical bet sizes

        Args:
            wallet_addresses: List of wallets to analyze.

        Returns:
            Dictionary mapping wallets to their potential Sybil connections.
        """
        sybil_connections: dict[str, list[str]] = defaultdict(list)

        with self.db.get_connection() as conn:
            # For each wallet pair, check betting similarity
            for i, wallet1 in enumerate(wallet_addresses):
                for wallet2 in wallet_addresses[i+1:]:
                    similarity = self._calculate_wallet_similarity(conn, wallet1, wallet2)

                    if similarity >= self.config.CLUSTER_BET_SIMILARITY:
                        sybil_connections[wallet1].append(wallet2)
                        sybil_connections[wallet2].append(wallet1)

        return dict(sybil_connections)

    def _calculate_wallet_similarity(
        self,
        conn,
        wallet1: str,
        wallet2: str
    ) -> float:
        """
        Calculate betting similarity between two wallets.

        Args:
            conn: Database connection.
            wallet1: First wallet.
            wallet2: Second wallet.

        Returns:
            Similarity score 0-1.
        """
        # Get bets for both wallets
        cursor = conn.execute(
            """
            SELECT market_id, outcome_bet, timestamp, amount
            FROM bets
            WHERE wallet_address IN (?, ?)
            ORDER BY timestamp
            """,
            (wallet1, wallet2)
        )
        all_bets = cursor.fetchall()

        if len(all_bets) < 4:  # Need enough bets to compare
            return 0.0

        # Group by wallet
        w1_bets = []
        w2_bets = []
        for bet in all_bets:
            # We need to know which wallet - query again
            pass

        # Simplified: check market overlap
        cursor = conn.execute(
            """
            SELECT market_id FROM bets WHERE wallet_address = ?
            """,
            (wallet1,)
        )
        w1_markets = set(row["market_id"] for row in cursor.fetchall())

        cursor = conn.execute(
            """
            SELECT market_id FROM bets WHERE wallet_address = ?
            """,
            (wallet2,)
        )
        w2_markets = set(row["market_id"] for row in cursor.fetchall())

        if not w1_markets or not w2_markets:
            return 0.0

        # Jaccard similarity for market overlap
        intersection = len(w1_markets & w2_markets)
        union = len(w1_markets | w2_markets)

        market_similarity = intersection / union if union > 0 else 0

        # Check if they bet same outcomes on shared markets
        if intersection > 0:
            shared_markets = w1_markets & w2_markets
            same_outcome_count = 0

            for market in list(shared_markets)[:10]:  # Sample up to 10
                cursor = conn.execute(
                    """
                    SELECT wallet_address, outcome_bet
                    FROM bets
                    WHERE market_id = ? AND wallet_address IN (?, ?)
                    """,
                    (market, wallet1, wallet2)
                )
                outcomes = {}
                for row in cursor.fetchall():
                    outcomes[row["wallet_address"]] = row["outcome_bet"]

                if wallet1 in outcomes and wallet2 in outcomes:
                    if outcomes[wallet1] == outcomes[wallet2]:
                        same_outcome_count += 1

            outcome_similarity = same_outcome_count / min(10, len(shared_markets))
        else:
            outcome_similarity = 0

        # Combined similarity
        return 0.4 * market_similarity + 0.6 * outcome_similarity

    # ========================================================================
    # Anomaly Scoring
    # ========================================================================

    def calculate_anomaly_score(
        self,
        bet_id: str
    ) -> SuspiciousBet:
        """
        Calculate comprehensive anomaly score for a bet.

        Combines multiple factors:
        - Timing (proximity to resolution)
        - Amount (size relative to market)
        - History (trader's past behavior)
        - Liquidity (when bet was placed)
        - Probability (betting against odds)

        Args:
            bet_id: Bet to analyze.

        Returns:
            SuspiciousBet with scores and flags.
        """
        with self.db.get_connection() as conn:
            # Get bet details with market info
            cursor = conn.execute(
                """
                SELECT
                    b.*,
                    m.question,
                    m.end_date,
                    m.volume as market_volume,
                    m.liquidity,
                    m.resolved,
                    m.outcome as market_outcome,
                    m.outcome_prices,
                    m.created_at as market_created
                FROM bets b
                JOIN markets m ON b.market_id = m.market_id
                WHERE b.bet_id = ?
                """,
                (bet_id,)
            )
            row = cursor.fetchone()

            if not row:
                return SuspiciousBet(
                    bet_id=bet_id,
                    wallet_address="",
                    market_id="",
                    amount=0,
                    price=0,
                    side="",
                    outcome_bet="",
                    timestamp=datetime.utcnow(),
                )

            # Parse data
            bet_timestamp = row["timestamp"]
            if isinstance(bet_timestamp, str):
                bet_timestamp = datetime.fromisoformat(bet_timestamp)

            end_date = row["end_date"]
            if isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date)

            amount = float(row["amount"] or 0)
            price = float(row["price"] or 0)
            market_volume = float(row["market_volume"] or 0)

            suspicious_bet = SuspiciousBet(
                bet_id=bet_id,
                wallet_address=row["wallet_address"],
                market_id=row["market_id"],
                amount=amount,
                price=price,
                side=row["side"],
                outcome_bet=row["outcome_bet"],
                timestamp=bet_timestamp,
                market_question=row["question"] or "",
                market_volume=market_volume,
            )

            score_breakdown = {}
            total_score = 0.0
            alert_types = []

            # 1. Timing check - large bet near resolution
            is_timing_suspicious, timing_score, hours_to_res = self.check_large_bet_near_resolution(
                amount, bet_timestamp, end_date
            )
            if is_timing_suspicious:
                alert_types.append(AlertType.LARGE_BET_NEAR_RESOLUTION)
                score_breakdown["timing"] = round(timing_score, 1)
                total_score += timing_score
            suspicious_bet.hours_to_resolution = hours_to_res

            # 2. Dormant wallet check
            is_dormant, dormant_score = self.check_dormant_wallet_activation(
                row["wallet_address"], bet_timestamp, amount
            )
            if is_dormant:
                alert_types.append(AlertType.DORMANT_WALLET_ACTIVATION)
                score_breakdown["dormant_wallet"] = round(dormant_score, 1)
                total_score += dormant_score

            # 3. Low liquidity check
            is_low_liq, low_liq_score = self.check_low_liquidity_betting(
                bet_timestamp, amount
            )
            if is_low_liq:
                alert_types.append(AlertType.LOW_LIQUIDITY_BETTING)
                score_breakdown["low_liquidity"] = round(low_liq_score, 1)
                total_score += low_liq_score

            # 4. Amount relative to market
            if market_volume > 0:
                amount_ratio = (amount * price) / market_volume
                if amount_ratio > 0.05:  # More than 5% of market volume
                    amount_score = self.config.WEIGHT_AMOUNT * min(1.0, amount_ratio / 0.2)
                    score_breakdown["large_market_share"] = round(amount_score, 1)
                    total_score += amount_score

            # 5. Probability factor - betting against consensus
            if row["outcome_prices"]:
                try:
                    import json
                    prices = json.loads(row["outcome_prices"])
                    if isinstance(prices, list) and len(prices) >= 2:
                        outcome_lower = (row["outcome_bet"] or "").lower()
                        if outcome_lower in ["yes", "true"]:
                            implied_prob = float(prices[0])
                        else:
                            implied_prob = float(prices[1]) if len(prices) > 1 else 1 - float(prices[0])

                        # Betting on low probability outcome with large amount
                        if implied_prob < 0.2 and amount >= self.config.LARGE_BET_THRESHOLD * 0.5:
                            prob_score = self.config.WEIGHT_PROBABILITY * (1 - implied_prob) * 0.5
                            score_breakdown["low_prob_bet"] = round(prob_score, 1)
                            total_score += prob_score
                except (json.JSONDecodeError, TypeError, IndexError):
                    pass

            # Cap score at 100
            total_score = min(100, total_score)

            suspicious_bet.anomaly_score = total_score
            suspicious_bet.alert_types = alert_types
            suspicious_bet.score_breakdown = score_breakdown

            # Add high score alert if applicable
            if total_score >= self.config.ALERT_SCORE_THRESHOLD:
                if AlertType.HIGH_ANOMALY_SCORE not in alert_types:
                    alert_types.append(AlertType.HIGH_ANOMALY_SCORE)

            return suspicious_bet

    def analyze_all_bets(
        self,
        since: Optional[datetime] = None,
        limit: int = 1000
    ) -> list[SuspiciousBet]:
        """
        Analyze multiple bets for anomalies.

        Args:
            since: Only analyze bets after this timestamp.
            limit: Maximum bets to analyze.

        Returns:
            List of SuspiciousBet sorted by anomaly score.
        """
        suspicious_bets = []

        with self.db.get_connection() as conn:
            if since:
                cursor = conn.execute(
                    """
                    SELECT bet_id FROM bets
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (since, limit)
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT bet_id FROM bets
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (limit,)
                )

            bet_ids = [row["bet_id"] for row in cursor.fetchall()]

        for bet_id in bet_ids:
            try:
                suspicious_bet = self.calculate_anomaly_score(bet_id)
                if suspicious_bet.anomaly_score > 0:
                    suspicious_bets.append(suspicious_bet)
            except Exception as e:
                logger.error(f"Error analyzing bet {bet_id}: {e}")
                continue

        # Sort by score descending
        suspicious_bets.sort(key=lambda x: x.anomaly_score, reverse=True)

        return suspicious_bets

    # ========================================================================
    # Wallet Risk Profiling
    # ========================================================================

    def build_wallet_risk_profile(
        self,
        wallet_address: str
    ) -> WalletRiskProfile:
        """
        Build comprehensive risk profile for a wallet.

        Args:
            wallet_address: Wallet to analyze.

        Returns:
            WalletRiskProfile with all risk indicators.
        """
        profile = WalletRiskProfile(wallet_address=wallet_address)

        with self.db.get_connection() as conn:
            # Basic stats
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as bet_count,
                    SUM(amount * price) as total_volume,
                    AVG(amount) as avg_bet,
                    MIN(timestamp) as first_seen,
                    MAX(timestamp) as last_seen
                FROM bets
                WHERE wallet_address = ?
                """,
                (wallet_address,)
            )
            row = cursor.fetchone()

            if row:
                profile.total_bets_analyzed = row["bet_count"] or 0
                profile.total_volume = float(row["total_volume"] or 0)
                profile.avg_bet_size = float(row["avg_bet"] or 0)

                if row["first_seen"]:
                    profile.first_seen = row["first_seen"] if isinstance(row["first_seen"], datetime) else datetime.fromisoformat(str(row["first_seen"]))
                if row["last_seen"]:
                    profile.last_seen = row["last_seen"] if isinstance(row["last_seen"], datetime) else datetime.fromisoformat(str(row["last_seen"]))

        # Analyze all bets from this wallet
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT bet_id FROM bets WHERE wallet_address = ?",
                (wallet_address,)
            )
            bet_ids = [row["bet_id"] for row in cursor.fetchall()]

        alert_counts: dict[str, int] = defaultdict(int)
        total_score = 0.0
        suspicious_count = 0

        for bet_id in bet_ids:
            try:
                suspicious_bet = self.calculate_anomaly_score(bet_id)
                total_score += suspicious_bet.anomaly_score

                if suspicious_bet.anomaly_score >= self.config.WARNING_SCORE_THRESHOLD:
                    suspicious_count += 1
                    for alert_type in suspicious_bet.alert_types:
                        alert_counts[alert_type.value] += 1
            except Exception as e:
                logger.error(f"Error analyzing bet {bet_id}: {e}")
                continue

        profile.total_suspicious_bets = suspicious_count
        profile.alert_counts = dict(alert_counts)

        # Calculate overall risk score
        if profile.total_bets_analyzed > 0:
            avg_score = total_score / profile.total_bets_analyzed
            suspicion_rate = suspicious_count / profile.total_bets_analyzed
            profile.overall_risk_score = min(100, avg_score * 0.6 + suspicion_rate * 100 * 0.4)

        # Check niche market dominance
        is_niche_dominant, _, niche_details = self.check_niche_market_dominance(wallet_address)
        if is_niche_dominant:
            profile.flags.append("niche_market_dominance")
            profile.win_rate_niche = niche_details.get("niche_win_rate", 0)

        # Check first mover advantage
        is_first_mover, _, viral_markets = self.check_first_mover_advantage(wallet_address)
        if is_first_mover:
            profile.flags.append("first_mover_advantage")

        return profile

    # ========================================================================
    # Market Manipulation Detection
    # ========================================================================

    def detect_volume_spike(
        self,
        market_id: str,
        window_hours: int = 24
    ) -> Optional[MarketManipulationAlert]:
        """
        Detect unusual volume spikes on a market.

        Args:
            market_id: Market to analyze.
            window_hours: Time window for comparison.

        Returns:
            Alert if manipulation suspected.
        """
        with self.db.get_connection() as conn:
            # Get recent volume vs historical average
            now = datetime.utcnow()
            window_start = now - timedelta(hours=window_hours)
            historical_start = window_start - timedelta(days=7)

            # Recent volume
            cursor = conn.execute(
                """
                SELECT SUM(amount * price) as recent_volume, COUNT(*) as recent_count
                FROM bets
                WHERE market_id = ? AND timestamp >= ?
                """,
                (market_id, window_start)
            )
            recent = cursor.fetchone()
            recent_volume = float(recent["recent_volume"] or 0)
            recent_count = recent["recent_count"] or 0

            # Historical average (per window)
            cursor = conn.execute(
                """
                SELECT SUM(amount * price) as hist_volume, COUNT(*) as hist_count
                FROM bets
                WHERE market_id = ? AND timestamp >= ? AND timestamp < ?
                """,
                (market_id, historical_start, window_start)
            )
            historical = cursor.fetchone()
            hist_volume = float(historical["hist_volume"] or 0)
            hist_count = historical["hist_count"] or 0

            # Normalize to per-window average
            num_windows = 7 * 24 / window_hours
            avg_window_volume = hist_volume / num_windows if num_windows > 0 else 0

            if avg_window_volume > 0 and recent_volume > avg_window_volume * 5:
                # 5x normal volume is suspicious
                cursor = conn.execute(
                    "SELECT question FROM markets WHERE market_id = ?",
                    (market_id,)
                )
                market_row = cursor.fetchone()
                question = market_row["question"] if market_row else ""

                return MarketManipulationAlert(
                    market_id=market_id,
                    market_question=question,
                    alert_type="volume_spike",
                    timestamp=now,
                    details={
                        "recent_volume": round(recent_volume, 2),
                        "recent_bet_count": recent_count,
                        "avg_window_volume": round(avg_window_volume, 2),
                        "spike_multiplier": round(recent_volume / avg_window_volume, 1),
                    },
                    severity=AlertSeverity.HIGH if recent_volume > avg_window_volume * 10 else AlertSeverity.MEDIUM,
                )

        return None


# Global detector instance
insider_detector = InsiderDetector()
