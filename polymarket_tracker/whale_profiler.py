"""
Whale Profiler Module for Polymarket Tracker.

Identifies high-volume wallets, calculates their historical accuracy
on resolved markets, and assigns performance grades (A/B/C/D/ungraded).

Whales are identified by:
- Cumulative USD volume exceeding a threshold
- Any single bet exceeding a size threshold

Performance grades are based on win rate on resolved markets:
- A: >= 65% win rate with sufficient resolved bets
- B: >= 55% win rate with sufficient resolved bets
- C: >= 45% win rate with sufficient resolved bets
- D: < 45% win rate with sufficient resolved bets
- ungraded: insufficient resolved bets for grading
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from .database import Database, db

logger = logging.getLogger(__name__)


@dataclass
class WhaleConfig:
    """Configuration for whale identification and grading thresholds."""

    VOLUME_THRESHOLD: float = 50000.0       # Cumulative USD volume to be a whale
    SINGLE_BET_THRESHOLD: float = 5000.0    # Single bet size to flag wallet
    MIN_RESOLVED_BETS: int = 20             # Minimum resolved bets for grading
    STALE_DAYS: int = 90                    # Days inactive to flag as stale
    GRADE_A_THRESHOLD: float = 65.0
    GRADE_B_THRESHOLD: float = 55.0
    GRADE_C_THRESHOLD: float = 45.0


class WhaleProfiler:
    """
    Identifies whale wallets and profiles their trading performance.

    Whales are high-volume traders whose betting accuracy on resolved
    markets is tracked and graded to assess "smart money" quality.
    """

    def __init__(
        self,
        database: Optional[Database] = None,
        config: Optional[WhaleConfig] = None,
    ):
        """
        Initialize WhaleProfiler.

        Args:
            database: Database instance. Uses global db if not provided.
            config: WhaleConfig instance. Uses defaults if not provided.
        """
        self.db = database or db
        self.config = config or WhaleConfig()

    @staticmethod
    def classify_grade(
        win_rate: float,
        resolved_bets: int,
        min_bets: int = 20,
    ) -> str:
        """
        Assign a performance grade based on win rate and sample size.

        Args:
            win_rate: Win rate as a percentage (0-100).
            resolved_bets: Number of resolved market bets.
            min_bets: Minimum resolved bets required for a grade.

        Returns:
            Grade string: "A", "B", "C", "D", or "ungraded".
        """
        if resolved_bets < min_bets:
            return "ungraded"
        if win_rate >= 65.0:
            return "A"
        if win_rate >= 55.0:
            return "B"
        if win_rate >= 45.0:
            return "C"
        return "D"

    def identify_whale_addresses(self) -> list[str]:
        """
        Find wallet addresses that qualify as whales.

        A wallet is a whale if:
        - Its cumulative total_volume >= VOLUME_THRESHOLD, OR
        - It has any single bet where (amount * price) >= SINGLE_BET_THRESHOLD

        Returns:
            Deduplicated list of whale wallet addresses.
        """
        whale_addresses: set[str] = set()

        with self.db.get_connection() as conn:
            # Wallets by cumulative volume
            cursor = conn.execute(
                "SELECT wallet_address FROM traders WHERE total_volume >= ?",
                (self.config.VOLUME_THRESHOLD,),
            )
            for row in cursor.fetchall():
                whale_addresses.add(row["wallet_address"])

            # Wallets with any single large bet
            cursor = conn.execute(
                "SELECT DISTINCT wallet_address FROM bets WHERE (amount * price) >= ?",
                (self.config.SINGLE_BET_THRESHOLD,),
            )
            for row in cursor.fetchall():
                whale_addresses.add(row["wallet_address"])

        return list(whale_addresses)

    def calculate_wallet_stats(self, wallet_address: str) -> dict:
        """
        Calculate performance statistics for a wallet on resolved markets.

        For each resolved market with a decisive outcome (not tie/cancelled),
        determines whether the wallet won or lost:
        - BUY on winning outcome = win
        - BUY on losing outcome = loss
        - SELL on winning outcome = loss
        - SELL on losing outcome = win

        Only counts one result per market_id to avoid double-counting.

        Args:
            wallet_address: The wallet address to analyze.

        Returns:
            Dict with keys: win_rate, resolved_bets, wins, losses,
            category_specialization, activity_pattern, total_volume.
        """
        wins = 0
        losses = 0
        category_wins: dict[str, int] = defaultdict(int)
        category_totals: dict[str, int] = defaultdict(int)
        bet_timestamps: list[datetime] = []

        with self.db.get_connection() as conn:
            # Get all bets joined with resolved markets
            cursor = conn.execute(
                """
                SELECT b.market_id, b.side, b.outcome_bet, b.timestamp,
                       m.outcome AS market_outcome, m.category
                FROM bets b
                JOIN markets m ON b.market_id = m.market_id
                WHERE b.wallet_address = ?
                  AND m.resolved = 1
                  AND m.outcome IS NOT NULL
                ORDER BY b.timestamp ASC
                """,
                (wallet_address,),
            )
            rows = cursor.fetchall()

            # Track per-market results to avoid double-counting
            market_results: dict[str, dict] = {}
            for row in rows:
                market_id = row["market_id"]
                side = row["side"]
                outcome_bet = row["outcome_bet"]
                market_outcome = row["market_outcome"]
                category = row["category"] or "unknown"
                timestamp = row["timestamp"]

                # Collect all bet timestamps for activity pattern
                if isinstance(timestamp, str):
                    try:
                        ts = datetime.fromisoformat(timestamp)
                    except (ValueError, TypeError):
                        ts = datetime.utcnow()
                else:
                    ts = timestamp
                bet_timestamps.append(ts)

                # Skip ties / cancelled markets
                if market_outcome and market_outcome.lower() in ("tie", "cancelled", "n/a"):
                    continue

                # Only count first result per market to avoid double-counting
                if market_id in market_results:
                    continue

                # Determine win or loss
                bet_won = False
                if side == "BUY":
                    bet_won = (outcome_bet == market_outcome)
                else:  # SELL
                    bet_won = (outcome_bet != market_outcome)

                market_results[market_id] = {
                    "won": bet_won,
                    "category": category,
                }

            # Aggregate results
            for result in market_results.values():
                if result["won"]:
                    wins += 1
                    category_wins[result["category"]] += 1
                else:
                    losses += 1
                category_totals[result["category"]] += 1

        resolved_bets = wins + losses
        win_rate = (wins / resolved_bets * 100) if resolved_bets > 0 else 0.0

        # Category specialization: win rate per category
        category_specialization: dict[str, float] = {}
        for cat, total in category_totals.items():
            cat_wins = category_wins.get(cat, 0)
            category_specialization[cat] = (cat_wins / total * 100) if total > 0 else 0.0

        # Activity pattern analysis
        activity_pattern = self._determine_activity_pattern(bet_timestamps)

        # Get total volume from traders table
        trader = self.db.get_trader(wallet_address)
        total_volume = trader["total_volume"] if trader else 0.0

        return {
            "win_rate": win_rate,
            "resolved_bets": resolved_bets,
            "wins": wins,
            "losses": losses,
            "category_specialization": category_specialization,
            "activity_pattern": activity_pattern,
            "total_volume": total_volume,
        }

    def _determine_activity_pattern(self, timestamps: list[datetime]) -> str:
        """
        Classify activity pattern from a list of bet timestamps.

        Args:
            timestamps: Sorted list of bet timestamps.

        Returns:
            "dormant_burst" if max gap >= 30 days,
            "steady" if 10+ bets and no big gaps,
            "sporadic" otherwise.
        """
        if len(timestamps) < 2:
            return "sporadic"

        sorted_ts = sorted(timestamps)
        max_gap_days = 0.0
        for i in range(1, len(sorted_ts)):
            gap = (sorted_ts[i] - sorted_ts[i - 1]).total_seconds() / 86400.0
            if gap > max_gap_days:
                max_gap_days = gap

        if max_gap_days >= 30:
            return "dormant_burst"
        if len(timestamps) >= 10:
            return "steady"
        return "sporadic"

    def profile_whale(self, wallet_address: str) -> dict:
        """
        Build a complete whale profile for a wallet address.

        Calculates wallet stats, assigns a grade, and persists the
        profile to the whale_profiles table.

        Args:
            wallet_address: The wallet address to profile.

        Returns:
            Profile dict with grade, win_rate, resolved_bets,
            category_specialization, activity_pattern, total_volume.
        """
        stats = self.calculate_wallet_stats(wallet_address)

        grade = self.classify_grade(
            win_rate=stats["win_rate"],
            resolved_bets=stats["resolved_bets"],
            min_bets=self.config.MIN_RESOLVED_BETS,
        )

        cat_spec_json = json.dumps(stats["category_specialization"])

        # Persist to database
        self.db.upsert_whale_profile(
            wallet_address=wallet_address,
            grade=grade,
            win_rate=stats["win_rate"],
            roi=0.0,  # ROI calculation not implemented yet
            sharpe_ratio=None,
            category_specialization=cat_spec_json,
            avg_bet_timing=None,
            activity_pattern=stats["activity_pattern"],
            total_volume=stats["total_volume"],
            total_resolved_bets=stats["resolved_bets"],
        )

        return {
            "wallet_address": wallet_address,
            "grade": grade,
            "win_rate": stats["win_rate"],
            "resolved_bets": stats["resolved_bets"],
            "wins": stats["wins"],
            "losses": stats["losses"],
            "category_specialization": stats["category_specialization"],
            "activity_pattern": stats["activity_pattern"],
            "total_volume": stats["total_volume"],
        }

    def refresh_all_profiles(self) -> int:
        """
        Identify all whale addresses and refresh their profiles.

        Errors on individual wallets are logged but do not stop
        processing of other wallets.

        Returns:
            Count of profiles successfully updated.
        """
        addresses = self.identify_whale_addresses()
        updated = 0

        for addr in addresses:
            try:
                self.profile_whale(addr)
                updated += 1
            except Exception:
                logger.exception(f"Error profiling whale {addr}")

        logger.info(f"Refreshed {updated}/{len(addresses)} whale profiles")
        return updated
