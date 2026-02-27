"""
Win Rate Analysis Module for Polymarket Tracker.

This module provides comprehensive win rate calculations including:
- Overall win rates on resolved markets only
- Win rates by market category (politics, sports, crypto, etc.)
- Win rates by bet size (small, medium, large)
- Winning/losing streak detection
- Statistical significance analysis with confidence intervals
- Kelly Criterion analysis for optimal bet sizing

Edge cases handled:
- Ties (market resolves with no clear winner)
- Partial fills (incomplete order execution)
- Market cancellations (voided markets)
- Insufficient data flagging
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from .database import Database, db

logger = logging.getLogger(__name__)


# Bet size thresholds in USD
class BetSizeCategory(Enum):
    """Categories for bet sizes."""
    SMALL = "small"       # < $100
    MEDIUM = "medium"     # $100 - $1,000
    LARGE = "large"       # > $1,000

    @classmethod
    def from_amount(cls, amount: float) -> "BetSizeCategory":
        """Determine category from bet amount."""
        if amount < 100:
            return cls.SMALL
        elif amount <= 1000:
            return cls.MEDIUM
        else:
            return cls.LARGE


class MarketStatus(Enum):
    """Status of a market for win rate calculations."""
    RESOLVED_WIN = "win"
    RESOLVED_LOSS = "loss"
    RESOLVED_TIE = "tie"       # Market resolved with refund/no clear winner
    CANCELLED = "cancelled"    # Market was voided
    OPEN = "open"              # Not yet resolved
    PARTIAL = "partial"        # Partial fill - not all shares executed


@dataclass
class BetOutcome:
    """Represents the outcome of a single bet/position."""
    bet_id: str
    market_id: str
    wallet_address: str
    outcome_bet: str
    amount: float
    price: float
    cost_basis: float
    timestamp: datetime
    category: Optional[str] = None
    bet_size_category: BetSizeCategory = BetSizeCategory.SMALL
    status: MarketStatus = MarketStatus.OPEN
    pnl: float = 0.0
    market_outcome: Optional[str] = None
    is_partial_fill: bool = False


@dataclass
class StreakData:
    """Tracks winning and losing streaks."""
    current_streak: int = 0          # Positive = winning, negative = losing
    current_streak_type: str = "none"  # "winning", "losing", "none"
    longest_winning_streak: int = 0
    longest_losing_streak: int = 0
    streak_history: list = field(default_factory=list)  # List of (streak_length, type, end_date)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "current_streak": abs(self.current_streak),
            "current_streak_type": self.current_streak_type,
            "longest_winning_streak": self.longest_winning_streak,
            "longest_losing_streak": self.longest_losing_streak,
            "streak_history": self.streak_history[-10:],  # Last 10 streaks
        }


@dataclass
class WinRateByCategory:
    """Win rate breakdown by market category."""
    category: str
    wins: int = 0
    losses: int = 0
    ties: int = 0
    total_resolved: int = 0
    win_rate: float = 0.0
    total_volume: float = 0.0
    total_pnl: float = 0.0

    def calculate_win_rate(self) -> None:
        """Calculate win rate percentage."""
        # Exclude ties from win rate calculation
        decisive_bets = self.wins + self.losses
        if decisive_bets > 0:
            self.win_rate = (self.wins / decisive_bets) * 100
        self.total_resolved = self.wins + self.losses + self.ties


@dataclass
class WinRateByBetSize:
    """Win rate breakdown by bet size."""
    size_category: BetSizeCategory
    wins: int = 0
    losses: int = 0
    ties: int = 0
    total_resolved: int = 0
    win_rate: float = 0.0
    total_volume: float = 0.0
    avg_bet_size: float = 0.0
    total_pnl: float = 0.0

    def calculate_win_rate(self) -> None:
        """Calculate win rate percentage."""
        decisive_bets = self.wins + self.losses
        if decisive_bets > 0:
            self.win_rate = (self.wins / decisive_bets) * 100
        self.total_resolved = self.wins + self.losses + self.ties


@dataclass
class ConfidenceInterval:
    """Statistical confidence interval for win rate."""
    lower_bound: float
    upper_bound: float
    confidence_level: float  # e.g., 0.95 for 95%
    sample_size: int
    is_significant: bool  # True if sample size >= threshold


@dataclass
class KellyCriterion:
    """Kelly Criterion analysis for optimal bet sizing."""
    optimal_fraction: float      # Optimal bet size as fraction of bankroll
    actual_avg_fraction: float   # Trader's actual average bet fraction
    edge: float                  # Estimated edge (expected return)
    is_overbetting: bool        # True if betting more than Kelly suggests
    kelly_multiple: float       # Actual / Optimal ratio
    recommendation: str         # Text recommendation


@dataclass
class TraderWinRateAnalysis:
    """Comprehensive win rate analysis for a trader."""
    wallet_address: str

    # Overall stats (resolved markets only)
    total_resolved_bets: int = 0
    total_wins: int = 0
    total_losses: int = 0
    total_ties: int = 0
    total_cancelled: int = 0
    overall_win_rate: float = 0.0

    # Breakdowns
    by_category: dict = field(default_factory=dict)  # category -> WinRateByCategory
    by_bet_size: dict = field(default_factory=dict)  # BetSizeCategory -> WinRateByBetSize

    # Streaks
    streaks: StreakData = field(default_factory=StreakData)

    # Statistical analysis
    confidence_interval: Optional[ConfidenceInterval] = None
    has_sufficient_data: bool = False
    min_bets_for_significance: int = 10

    # Kelly Criterion
    kelly_analysis: Optional[KellyCriterion] = None

    # Risk metrics
    sharpe_ratio: Optional[float] = None
    max_drawdown: float = 0.0
    risk_adjusted_return: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "wallet_address": self.wallet_address,
            "overall": {
                "total_resolved_bets": self.total_resolved_bets,
                "wins": self.total_wins,
                "losses": self.total_losses,
                "ties": self.total_ties,
                "cancelled": self.total_cancelled,
                "win_rate": self.overall_win_rate,
                "has_sufficient_data": self.has_sufficient_data,
            },
            "by_category": {
                cat: {
                    "wins": data.wins,
                    "losses": data.losses,
                    "ties": data.ties,
                    "win_rate": data.win_rate,
                    "volume": data.total_volume,
                    "pnl": data.total_pnl,
                }
                for cat, data in self.by_category.items()
            },
            "by_bet_size": {
                size.value: {
                    "wins": data.wins,
                    "losses": data.losses,
                    "ties": data.ties,
                    "win_rate": data.win_rate,
                    "avg_bet_size": data.avg_bet_size,
                    "pnl": data.total_pnl,
                }
                for size, data in self.by_bet_size.items()
            },
            "streaks": self.streaks.to_dict(),
            "confidence_interval": {
                "lower": self.confidence_interval.lower_bound,
                "upper": self.confidence_interval.upper_bound,
                "confidence_level": self.confidence_interval.confidence_level,
                "is_significant": self.confidence_interval.is_significant,
            } if self.confidence_interval else None,
            "kelly_analysis": {
                "optimal_fraction": self.kelly_analysis.optimal_fraction,
                "actual_avg_fraction": self.kelly_analysis.actual_avg_fraction,
                "edge": self.kelly_analysis.edge,
                "is_overbetting": self.kelly_analysis.is_overbetting,
                "kelly_multiple": self.kelly_analysis.kelly_multiple,
                "recommendation": self.kelly_analysis.recommendation,
            } if self.kelly_analysis else None,
            "risk_metrics": {
                "sharpe_ratio": self.sharpe_ratio,
                "max_drawdown": self.max_drawdown,
                "risk_adjusted_return": self.risk_adjusted_return,
            },
        }


class WinRateAnalyzer:
    """
    Analyzes win rates for Polymarket traders.

    Provides comprehensive win rate calculations with:
    - Category breakdowns (politics, sports, crypto, etc.)
    - Bet size breakdowns (small, medium, large)
    - Streak detection
    - Statistical significance testing
    - Kelly Criterion analysis
    """

    # Minimum bets for statistical significance
    MIN_BETS_FOR_SIGNIFICANCE = 10
    MIN_BETS_FOR_LEADERBOARD = 20

    # Confidence level for intervals (95%)
    CONFIDENCE_LEVEL = 0.95

    # Z-score for 95% confidence
    Z_SCORE_95 = 1.96

    def __init__(self, database: Optional[Database] = None):
        """
        Initialize win rate analyzer.

        Args:
            database: Database instance (uses global if not provided).
        """
        self.db = database or db

    def _get_bet_outcomes(self, wallet_address: str) -> list[BetOutcome]:
        """
        Get all bet outcomes for a trader with resolution status.

        Handles edge cases:
        - Ties: Markets with no clear winner
        - Cancellations: Voided markets
        - Partial fills: Incomplete executions

        Args:
            wallet_address: Trader's wallet address.

        Returns:
            List of BetOutcome objects sorted by timestamp.
        """
        outcomes = []

        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT
                    b.bet_id,
                    b.market_id,
                    b.wallet_address,
                    b.amount,
                    b.price,
                    b.side,
                    b.outcome_bet,
                    b.timestamp,
                    b.status as bet_status,
                    m.resolved,
                    m.outcome as market_outcome,
                    m.category,
                    m.active,
                    m.closed
                FROM bets b
                LEFT JOIN markets m ON b.market_id = m.market_id
                WHERE b.wallet_address = ?
                ORDER BY b.timestamp ASC
                """,
                (wallet_address,)
            )

            # Track positions per market to handle buys/sells
            market_positions: dict[str, dict] = {}

            for row in cursor.fetchall():
                market_id = row["market_id"]
                outcome_bet = row["outcome_bet"]
                amount = float(row["amount"] or 0)
                price = float(row["price"] or 0)
                side = row["side"]
                resolved = row["resolved"]
                market_outcome = row["market_outcome"]
                category = row["category"] or "uncategorized"
                bet_status = row["bet_status"]

                # Track net position per market/outcome
                pos_key = f"{market_id}_{outcome_bet}"
                if pos_key not in market_positions:
                    market_positions[pos_key] = {
                        "net_size": 0.0,
                        "total_cost": 0.0,
                        "bets": [],
                    }

                pos = market_positions[pos_key]

                # Update position based on side
                if side.upper() == "BUY":
                    pos["net_size"] += amount
                    pos["total_cost"] += amount * price
                else:  # SELL
                    pos["net_size"] -= amount
                    pos["total_cost"] -= amount * price

                cost_basis = amount * price

                # Determine market status
                status = MarketStatus.OPEN
                pnl = 0.0
                is_partial = bet_status and "partial" in bet_status.lower()

                if resolved:
                    if market_outcome is None or market_outcome.lower() in ["tie", "void", "cancelled", "refund"]:
                        # Tie or cancelled - no winner
                        status = MarketStatus.RESOLVED_TIE
                        pnl = 0.0  # Refund scenario
                    elif market_outcome.lower() == "cancelled" or (not row["active"] and not row["closed"]):
                        status = MarketStatus.CANCELLED
                        pnl = 0.0
                    else:
                        # Check if this bet won or lost
                        bet_won = self._outcome_matches(outcome_bet, market_outcome)
                        if side.upper() == "BUY":
                            if bet_won:
                                status = MarketStatus.RESOLVED_WIN
                                pnl = amount - cost_basis  # $1 payout - cost
                            else:
                                status = MarketStatus.RESOLVED_LOSS
                                pnl = -cost_basis  # Lost entire cost
                        else:  # SELL
                            # Selling is opposite - you win if the outcome you sold doesn't happen
                            if bet_won:
                                status = MarketStatus.RESOLVED_LOSS
                                pnl = -amount + cost_basis  # Sold winner
                            else:
                                status = MarketStatus.RESOLVED_WIN
                                pnl = cost_basis  # Kept premium

                if is_partial:
                    status = MarketStatus.PARTIAL

                # Parse timestamp
                timestamp = row["timestamp"]
                if isinstance(timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp)
                    except ValueError:
                        timestamp = datetime.utcnow()

                outcome = BetOutcome(
                    bet_id=row["bet_id"],
                    market_id=market_id,
                    wallet_address=wallet_address,
                    outcome_bet=outcome_bet,
                    amount=amount,
                    price=price,
                    cost_basis=cost_basis,
                    timestamp=timestamp,
                    category=category,
                    bet_size_category=BetSizeCategory.from_amount(cost_basis),
                    status=status,
                    pnl=pnl,
                    market_outcome=market_outcome,
                    is_partial_fill=is_partial,
                )
                outcomes.append(outcome)

        return outcomes

    def _outcome_matches(self, bet_outcome: str, market_outcome: str) -> bool:
        """
        Check if bet outcome matches market outcome.

        Handles various representations: Yes/No, True/False, 1/0, etc.

        Args:
            bet_outcome: What the trader bet on.
            market_outcome: What the market resolved to.

        Returns:
            True if the bet won.
        """
        if not bet_outcome or not market_outcome:
            return False

        bet_lower = bet_outcome.lower().strip()
        market_lower = market_outcome.lower().strip()

        # Direct match
        if bet_lower == market_lower:
            return True

        # Yes equivalents
        yes_variants = {"yes", "true", "1", "y", "win", "correct"}
        no_variants = {"no", "false", "0", "n", "lose", "incorrect"}

        if bet_lower in yes_variants and market_lower in yes_variants:
            return True
        if bet_lower in no_variants and market_lower in no_variants:
            return True

        return False

    def _calculate_streaks(self, outcomes: list[BetOutcome]) -> StreakData:
        """
        Calculate winning and losing streaks from bet outcomes.

        Only considers resolved bets (excluding ties and cancellations).

        Args:
            outcomes: List of BetOutcome sorted by timestamp.

        Returns:
            StreakData with current and historical streaks.
        """
        streaks = StreakData()

        current_streak = 0
        current_type = "none"
        streak_start_date = None

        for outcome in outcomes:
            # Only consider decisive outcomes
            if outcome.status not in [MarketStatus.RESOLVED_WIN, MarketStatus.RESOLVED_LOSS]:
                continue

            is_win = outcome.status == MarketStatus.RESOLVED_WIN

            if current_type == "none":
                # Start first streak
                current_streak = 1
                current_type = "winning" if is_win else "losing"
                streak_start_date = outcome.timestamp
            elif (current_type == "winning" and is_win) or (current_type == "losing" and not is_win):
                # Continue current streak
                current_streak += 1
            else:
                # Streak broken - record it
                streaks.streak_history.append({
                    "length": current_streak,
                    "type": current_type,
                    "end_date": outcome.timestamp.isoformat() if outcome.timestamp else None,
                })

                # Update longest streaks
                if current_type == "winning":
                    streaks.longest_winning_streak = max(streaks.longest_winning_streak, current_streak)
                else:
                    streaks.longest_losing_streak = max(streaks.longest_losing_streak, current_streak)

                # Start new streak
                current_streak = 1
                current_type = "winning" if is_win else "losing"
                streak_start_date = outcome.timestamp

        # Record final streak
        if current_streak > 0:
            streaks.current_streak = current_streak
            streaks.current_streak_type = current_type

            if current_type == "winning":
                streaks.longest_winning_streak = max(streaks.longest_winning_streak, current_streak)
            else:
                streaks.longest_losing_streak = max(streaks.longest_losing_streak, current_streak)

        return streaks

    def _calculate_confidence_interval(
        self,
        wins: int,
        total: int,
        confidence_level: float = 0.95
    ) -> ConfidenceInterval:
        """
        Calculate confidence interval for win rate using Wilson score interval.

        The Wilson score interval is better for proportions, especially
        with small sample sizes or extreme probabilities.

        Args:
            wins: Number of wins.
            total: Total decisive bets (wins + losses, excluding ties).
            confidence_level: Confidence level (default 0.95 for 95%).

        Returns:
            ConfidenceInterval with bounds and significance flag.
        """
        if total == 0:
            return ConfidenceInterval(
                lower_bound=0.0,
                upper_bound=0.0,
                confidence_level=confidence_level,
                sample_size=0,
                is_significant=False,
            )

        p = wins / total
        z = self.Z_SCORE_95 if confidence_level == 0.95 else 1.645  # 90%

        # Wilson score interval formula
        denominator = 1 + z**2 / total
        center = (p + z**2 / (2 * total)) / denominator
        spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator

        lower = max(0.0, (center - spread) * 100)
        upper = min(100.0, (center + spread) * 100)

        return ConfidenceInterval(
            lower_bound=round(lower, 2),
            upper_bound=round(upper, 2),
            confidence_level=confidence_level,
            sample_size=total,
            is_significant=total >= self.MIN_BETS_FOR_SIGNIFICANCE,
        )

    def _calculate_kelly_criterion(
        self,
        win_rate: float,
        avg_odds: float,
        actual_bet_fraction: float
    ) -> KellyCriterion:
        """
        Calculate Kelly Criterion for optimal bet sizing.

        Kelly formula: f* = (bp - q) / b
        where:
            f* = fraction of bankroll to bet
            b = odds received on the bet (decimal odds - 1)
            p = probability of winning
            q = probability of losing (1 - p)

        Args:
            win_rate: Win rate as percentage (0-100).
            avg_odds: Average decimal odds (e.g., 2.0 for even money).
            actual_bet_fraction: Trader's actual average bet as fraction of volume.

        Returns:
            KellyCriterion analysis.
        """
        if win_rate <= 0 or avg_odds <= 1:
            return KellyCriterion(
                optimal_fraction=0.0,
                actual_avg_fraction=actual_bet_fraction,
                edge=0.0,
                is_overbetting=actual_bet_fraction > 0,
                kelly_multiple=float('inf') if actual_bet_fraction > 0 else 0,
                recommendation="Insufficient edge - consider not betting or improving selection.",
            )

        p = win_rate / 100
        q = 1 - p
        b = avg_odds - 1  # Convert decimal odds to "b" in Kelly formula

        # Kelly formula
        edge = (b * p - q)
        optimal_fraction = max(0, edge / b) if b > 0 else 0

        # Compare to actual betting
        kelly_multiple = actual_bet_fraction / optimal_fraction if optimal_fraction > 0 else float('inf')
        is_overbetting = kelly_multiple > 1.0

        # Generate recommendation
        if optimal_fraction <= 0:
            recommendation = "No positive edge detected. Consider improving bet selection."
        elif kelly_multiple > 2:
            recommendation = f"Significantly overbetting ({kelly_multiple:.1f}x Kelly). Reduce bet sizes to manage risk."
        elif kelly_multiple > 1.5:
            recommendation = f"Moderately overbetting ({kelly_multiple:.1f}x Kelly). Consider reducing position sizes."
        elif kelly_multiple < 0.5:
            recommendation = f"Conservative betting ({kelly_multiple:.1f}x Kelly). Could increase sizes for higher returns."
        else:
            recommendation = f"Near-optimal bet sizing ({kelly_multiple:.1f}x Kelly). Well-managed risk."

        return KellyCriterion(
            optimal_fraction=round(optimal_fraction * 100, 2),  # Convert to percentage
            actual_avg_fraction=round(actual_bet_fraction * 100, 2),
            edge=round(edge * 100, 2),
            is_overbetting=is_overbetting,
            kelly_multiple=round(kelly_multiple, 2) if kelly_multiple != float('inf') else 999.99,
            recommendation=recommendation,
        )

    def _calculate_risk_metrics(
        self,
        outcomes: list[BetOutcome],
        total_volume: float
    ) -> tuple[Optional[float], float, float]:
        """
        Calculate risk-adjusted metrics.

        Args:
            outcomes: List of resolved bet outcomes.
            total_volume: Total trading volume.

        Returns:
            Tuple of (sharpe_ratio, max_drawdown, risk_adjusted_return).
        """
        if not outcomes or total_volume == 0:
            return None, 0.0, 0.0

        # Calculate returns series
        returns = [o.pnl / o.cost_basis if o.cost_basis > 0 else 0 for o in outcomes
                   if o.status in [MarketStatus.RESOLVED_WIN, MarketStatus.RESOLVED_LOSS]]

        if len(returns) < 2:
            return None, 0.0, 0.0

        # Calculate Sharpe Ratio (assuming risk-free rate of 0)
        avg_return = sum(returns) / len(returns)
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance) if variance > 0 else 0

        sharpe = (avg_return / std_dev) if std_dev > 0 else 0

        # Calculate max drawdown
        cumulative = 0
        peak = 0
        max_drawdown = 0
        for r in returns:
            cumulative += r
            if cumulative > peak:
                peak = cumulative
            drawdown = (peak - cumulative) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

        # Risk-adjusted return (return / max_drawdown)
        total_return = sum(returns)
        risk_adjusted = total_return / max_drawdown if max_drawdown > 0 else total_return

        return round(sharpe, 3), round(max_drawdown * 100, 2), round(risk_adjusted, 3)

    def analyze_trader_win_rate(self, wallet_address: str) -> TraderWinRateAnalysis:
        """
        Perform comprehensive win rate analysis for a trader.

        Args:
            wallet_address: Trader's wallet address.

        Returns:
            TraderWinRateAnalysis with all metrics.
        """
        analysis = TraderWinRateAnalysis(wallet_address=wallet_address)

        # Get all bet outcomes
        outcomes = self._get_bet_outcomes(wallet_address)

        if not outcomes:
            return analysis

        # Initialize category and size tracking
        category_stats: dict[str, WinRateByCategory] = {}
        size_stats: dict[BetSizeCategory, WinRateByBetSize] = {
            BetSizeCategory.SMALL: WinRateByBetSize(size_category=BetSizeCategory.SMALL),
            BetSizeCategory.MEDIUM: WinRateByBetSize(size_category=BetSizeCategory.MEDIUM),
            BetSizeCategory.LARGE: WinRateByBetSize(size_category=BetSizeCategory.LARGE),
        }

        total_volume = 0.0
        total_cost_basis = 0.0
        resolved_outcomes = []

        for outcome in outcomes:
            category = outcome.category or "uncategorized"
            size_cat = outcome.bet_size_category
            cost = outcome.cost_basis

            total_volume += cost
            total_cost_basis += cost

            # Initialize category if new
            if category not in category_stats:
                category_stats[category] = WinRateByCategory(category=category)

            cat_stat = category_stats[category]
            size_stat = size_stats[size_cat]

            # Update volume tracking
            cat_stat.total_volume += cost
            size_stat.total_volume += cost

            # Process based on status
            if outcome.status == MarketStatus.RESOLVED_WIN:
                analysis.total_wins += 1
                cat_stat.wins += 1
                size_stat.wins += 1
                cat_stat.total_pnl += outcome.pnl
                size_stat.total_pnl += outcome.pnl
                resolved_outcomes.append(outcome)

            elif outcome.status == MarketStatus.RESOLVED_LOSS:
                analysis.total_losses += 1
                cat_stat.losses += 1
                size_stat.losses += 1
                cat_stat.total_pnl += outcome.pnl
                size_stat.total_pnl += outcome.pnl
                resolved_outcomes.append(outcome)

            elif outcome.status == MarketStatus.RESOLVED_TIE:
                analysis.total_ties += 1
                cat_stat.ties += 1
                size_stat.ties += 1

            elif outcome.status == MarketStatus.CANCELLED:
                analysis.total_cancelled += 1

        # Calculate overall stats
        analysis.total_resolved_bets = analysis.total_wins + analysis.total_losses + analysis.total_ties
        decisive_bets = analysis.total_wins + analysis.total_losses

        if decisive_bets > 0:
            analysis.overall_win_rate = round((analysis.total_wins / decisive_bets) * 100, 2)

        # Calculate category win rates
        for cat, stat in category_stats.items():
            stat.calculate_win_rate()
        analysis.by_category = category_stats

        # Calculate size category win rates
        for size_cat, stat in size_stats.items():
            stat.calculate_win_rate()
            if stat.total_resolved > 0:
                stat.avg_bet_size = stat.total_volume / stat.total_resolved
        analysis.by_bet_size = size_stats

        # Calculate streaks
        analysis.streaks = self._calculate_streaks(outcomes)

        # Statistical significance
        analysis.has_sufficient_data = decisive_bets >= self.MIN_BETS_FOR_SIGNIFICANCE
        analysis.confidence_interval = self._calculate_confidence_interval(
            analysis.total_wins,
            decisive_bets,
            self.CONFIDENCE_LEVEL
        )

        # Kelly Criterion analysis
        if decisive_bets > 0 and total_volume > 0:
            # Estimate average odds from average price (price represents probability)
            avg_price = total_cost_basis / len(outcomes) if outcomes else 0.5
            avg_odds = 1 / avg_price if avg_price > 0 else 2.0  # Convert probability to decimal odds

            # Estimate actual bet fraction (average bet / total volume as proxy)
            avg_bet = total_volume / len(outcomes) if outcomes else 0
            actual_fraction = avg_bet / total_volume if total_volume > 0 else 0

            analysis.kelly_analysis = self._calculate_kelly_criterion(
                analysis.overall_win_rate,
                avg_odds,
                actual_fraction
            )

        # Risk metrics
        sharpe, max_dd, risk_adj = self._calculate_risk_metrics(resolved_outcomes, total_volume)
        analysis.sharpe_ratio = sharpe
        analysis.max_drawdown = max_dd
        analysis.risk_adjusted_return = risk_adj

        return analysis

    def get_win_rate_leaderboard(
        self,
        limit: int = 20,
        min_resolved_bets: int = 20,
        sort_by: str = "win_rate"
    ) -> list[dict]:
        """
        Get leaderboard ranked by win rate metrics.

        Args:
            limit: Maximum traders to return.
            min_resolved_bets: Minimum resolved bets for inclusion.
            sort_by: Sort metric ("win_rate", "risk_adjusted", "sharpe", "streak").

        Returns:
            List of trader analysis dictionaries.
        """
        results = []

        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT DISTINCT wallet_address FROM traders"
            )
            wallets = [row["wallet_address"] for row in cursor.fetchall()]

        for wallet in wallets:
            try:
                analysis = self.analyze_trader_win_rate(wallet)

                # Filter by minimum resolved bets
                decisive = analysis.total_wins + analysis.total_losses
                if decisive < min_resolved_bets:
                    continue

                results.append({
                    "wallet_address": wallet,
                    "display_address": f"{wallet[:6]}...{wallet[-4:]}" if len(wallet) > 12 else wallet,
                    "win_rate": analysis.overall_win_rate,
                    "wins": analysis.total_wins,
                    "losses": analysis.total_losses,
                    "ties": analysis.total_ties,
                    "resolved_bets": decisive,
                    "current_streak": analysis.streaks.current_streak,
                    "current_streak_type": analysis.streaks.current_streak_type,
                    "longest_winning_streak": analysis.streaks.longest_winning_streak,
                    "longest_losing_streak": analysis.streaks.longest_losing_streak,
                    "confidence_lower": analysis.confidence_interval.lower_bound if analysis.confidence_interval else 0,
                    "confidence_upper": analysis.confidence_interval.upper_bound if analysis.confidence_interval else 0,
                    "sharpe_ratio": analysis.sharpe_ratio or 0,
                    "max_drawdown": analysis.max_drawdown,
                    "risk_adjusted_return": analysis.risk_adjusted_return,
                    "has_sufficient_data": analysis.has_sufficient_data,
                    "kelly_optimal": analysis.kelly_analysis.optimal_fraction if analysis.kelly_analysis else 0,
                    "kelly_multiple": analysis.kelly_analysis.kelly_multiple if analysis.kelly_analysis else 0,
                })
            except Exception as e:
                logger.error(f"Error analyzing {wallet}: {e}")
                continue

        # Sort by specified metric
        if sort_by == "win_rate":
            results.sort(key=lambda x: x["win_rate"], reverse=True)
        elif sort_by == "risk_adjusted":
            results.sort(key=lambda x: x["risk_adjusted_return"], reverse=True)
        elif sort_by == "sharpe":
            results.sort(key=lambda x: x["sharpe_ratio"], reverse=True)
        elif sort_by == "streak":
            results.sort(key=lambda x: x["longest_winning_streak"], reverse=True)

        # Assign ranks
        for i, entry in enumerate(results[:limit]):
            entry["rank"] = i + 1

        return results[:limit]


# Global analyzer instance
win_rate_analyzer = WinRateAnalyzer()
