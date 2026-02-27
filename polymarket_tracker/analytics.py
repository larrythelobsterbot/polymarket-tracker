"""
Analytics module for Polymarket Tracker.

This module provides advanced analytics for trader performance:
- PNL (Profit and Loss) calculations for resolved and open positions
- Volume tracking and aggregation
- Leaderboard generation with multiple ranking metrics
- Trader statistics and performance metrics

PNL Calculation Logic:
- Resolved markets: (winning_shares * $1 payout) - (cost_basis) - fees
- Open positions: (current_price - entry_price) * position_size
- Fees: Polymarket takes 2% (200 bps) on net winnings
"""

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Optional

from .database import Database, db

logger = logging.getLogger(__name__)


# Polymarket fee rate (2% on winnings)
POLYMARKET_FEE_RATE = Decimal("0.02")


class LeaderboardMetric(Enum):
    """Available metrics for leaderboard ranking."""
    PNL = "pnl"
    VOLUME = "volume"
    TRADE_COUNT = "trade_count"
    WIN_RATE = "win_rate"
    ROI = "roi"
    RISK_ADJUSTED = "risk_adjusted"
    SHARPE = "sharpe"
    STREAK = "streak"


@dataclass
class TraderPNL:
    """Detailed PNL breakdown for a trader."""
    wallet_address: str
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    total_pnl: Decimal = Decimal("0")
    total_fees_paid: Decimal = Decimal("0")
    winning_trades: int = 0
    losing_trades: int = 0
    total_cost_basis: Decimal = Decimal("0")
    total_proceeds: Decimal = Decimal("0")

    @property
    def win_rate(self) -> float:
        """Calculate win rate as percentage."""
        total = self.winning_trades + self.losing_trades
        if total == 0:
            return 0.0
        return (self.winning_trades / total) * 100

    @property
    def roi(self) -> float:
        """Calculate return on investment."""
        if self.total_cost_basis == 0:
            return 0.0
        return float((self.total_pnl / self.total_cost_basis) * 100)


@dataclass
class TraderStats:
    """Comprehensive trader statistics."""
    wallet_address: str
    total_volume: float = 0.0
    total_trades: int = 0
    markets_participated: int = 0
    avg_bet_size: float = 0.0
    avg_price: float = 0.0
    first_trade: Optional[datetime] = None
    last_trade: Optional[datetime] = None
    pnl: Optional[TraderPNL] = None

    # Breakdown by side
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    buy_count: int = 0
    sell_count: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result = {
            "wallet_address": self.wallet_address,
            "total_volume": self.total_volume,
            "total_trades": self.total_trades,
            "markets_participated": self.markets_participated,
            "avg_bet_size": self.avg_bet_size,
            "avg_price": self.avg_price,
            "first_trade": self.first_trade.isoformat() if self.first_trade else None,
            "last_trade": self.last_trade.isoformat() if self.last_trade else None,
            "buy_volume": self.buy_volume,
            "sell_volume": self.sell_volume,
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
        }
        if self.pnl:
            result["pnl"] = {
                "realized": float(self.pnl.realized_pnl),
                "unrealized": float(self.pnl.unrealized_pnl),
                "total": float(self.pnl.total_pnl),
                "fees_paid": float(self.pnl.total_fees_paid),
                "win_rate": self.pnl.win_rate,
                "roi": self.pnl.roi,
                "winning_trades": self.pnl.winning_trades,
                "losing_trades": self.pnl.losing_trades,
            }
        return result


@dataclass
class LeaderboardEntry:
    """Entry in the leaderboard."""
    rank: int
    wallet_address: str
    display_address: str  # Truncated for display
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    total_volume: float
    total_trades: int
    win_rate: float
    roi: float
    markets_participated: int
    avg_bet_size: float
    last_active: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "rank": self.rank,
            "wallet_address": self.wallet_address,
            "display_address": self.display_address,
            "total_pnl": self.total_pnl,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_volume": self.total_volume,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "roi": self.roi,
            "markets_participated": self.markets_participated,
            "avg_bet_size": self.avg_bet_size,
            "last_active": self.last_active.isoformat() if self.last_active else None,
        }


class PolymarketAnalytics:
    """
    Analytics engine for Polymarket trading data.

    Provides methods for calculating PNL, volume, and generating
    leaderboards with various ranking metrics.
    """

    def __init__(self, database: Optional[Database] = None):
        """
        Initialize analytics engine.

        Args:
            database: Database instance (uses global if not provided).
        """
        self.db = database or db

    def calculate_trader_pnl(self, wallet_address: str) -> TraderPNL:
        """
        Calculate total profit/loss for a trader.

        PNL is calculated as:
        - Resolved markets: (payout if won) - (cost basis) - (fees on winnings)
        - Open positions: (current_price - entry_price) * position_size

        Args:
            wallet_address: Trader's wallet address.

        Returns:
            TraderPNL with detailed breakdown.
        """
        pnl = TraderPNL(wallet_address=wallet_address)

        with self.db.get_connection() as conn:
            # Get all bets for this trader with market resolution info
            cursor = conn.execute(
                """
                SELECT
                    b.bet_id,
                    b.market_id,
                    b.amount,
                    b.price,
                    b.side,
                    b.outcome_bet,
                    b.fee_rate_bps,
                    m.resolved,
                    m.outcome as market_outcome,
                    m.outcome_prices
                FROM bets b
                LEFT JOIN markets m ON b.market_id = m.market_id
                WHERE b.wallet_address = ?
                """,
                (wallet_address,)
            )
            bets = cursor.fetchall()

            # Track positions per market for PNL calculation
            market_positions: dict[str, dict] = {}

            for bet in bets:
                market_id = bet["market_id"]
                amount = Decimal(str(bet["amount"] or 0))
                price = Decimal(str(bet["price"] or 0))
                side = bet["side"]
                outcome_bet = bet["outcome_bet"]
                resolved = bet["resolved"]
                market_outcome = bet["market_outcome"]
                fee_rate_bps = Decimal(str(bet["fee_rate_bps"] or 0))

                # Calculate cost basis for this trade
                cost_basis = amount * price
                pnl.total_cost_basis += cost_basis

                # Initialize market position tracking
                if market_id not in market_positions:
                    market_positions[market_id] = {
                        "positions": {},  # outcome -> {size, cost_basis}
                        "resolved": resolved,
                        "market_outcome": market_outcome,
                    }

                # Track position
                if outcome_bet not in market_positions[market_id]["positions"]:
                    market_positions[market_id]["positions"][outcome_bet] = {
                        "size": Decimal("0"),
                        "cost_basis": Decimal("0"),
                    }

                pos = market_positions[market_id]["positions"][outcome_bet]

                if side.upper() == "BUY":
                    pos["size"] += amount
                    pos["cost_basis"] += cost_basis
                else:  # SELL
                    # Selling reduces position and realizes PNL
                    proceeds = amount * price
                    pnl.total_proceeds += proceeds

                    if pos["size"] > 0:
                        # Calculate realized PNL on this sale
                        avg_entry = pos["cost_basis"] / pos["size"] if pos["size"] > 0 else Decimal("0")
                        sell_pnl = (price - avg_entry) * amount
                        pnl.realized_pnl += sell_pnl

                        # Reduce position
                        pos["size"] -= amount
                        if pos["size"] > 0:
                            pos["cost_basis"] = avg_entry * pos["size"]
                        else:
                            pos["cost_basis"] = Decimal("0")

            # Calculate PNL for resolved markets
            for market_id, market_data in market_positions.items():
                if market_data["resolved"] and market_data["market_outcome"]:
                    winning_outcome = market_data["market_outcome"]

                    for outcome, pos in market_data["positions"].items():
                        if pos["size"] <= 0:
                            continue

                        if outcome.lower() == winning_outcome.lower() or outcome == winning_outcome:
                            # Winner: payout is $1 per share
                            payout = pos["size"] * Decimal("1")
                            gross_pnl = payout - pos["cost_basis"]

                            # Apply 2% fee on winnings (only if profitable)
                            if gross_pnl > 0:
                                fee = gross_pnl * POLYMARKET_FEE_RATE
                                pnl.total_fees_paid += fee
                                net_pnl = gross_pnl - fee
                            else:
                                net_pnl = gross_pnl

                            pnl.realized_pnl += net_pnl
                            pnl.winning_trades += 1
                        else:
                            # Loser: shares worth $0
                            loss = pos["cost_basis"]
                            pnl.realized_pnl -= loss
                            pnl.losing_trades += 1
                else:
                    # Open position - calculate unrealized PNL
                    # Use current market prices from outcome_prices
                    cursor = conn.execute(
                        "SELECT outcome_prices FROM markets WHERE market_id = ?",
                        (market_id,)
                    )
                    market_row = cursor.fetchone()

                    if market_row and market_row["outcome_prices"]:
                        try:
                            import json
                            prices = json.loads(market_row["outcome_prices"])

                            for outcome, pos in market_data["positions"].items():
                                if pos["size"] <= 0:
                                    continue

                                # Try to get current price for this outcome
                                current_price = Decimal("0.5")  # Default to 50%
                                if isinstance(prices, list) and len(prices) > 0:
                                    # Assume Yes is first, No is second
                                    if outcome.lower() in ["yes", "true", "1"]:
                                        current_price = Decimal(str(prices[0]))
                                    elif outcome.lower() in ["no", "false", "0"]:
                                        current_price = Decimal(str(prices[1] if len(prices) > 1 else 1 - float(prices[0])))

                                # Unrealized PNL
                                current_value = pos["size"] * current_price
                                unrealized = current_value - pos["cost_basis"]
                                pnl.unrealized_pnl += unrealized
                        except (json.JSONDecodeError, TypeError, IndexError):
                            pass

        # Calculate total PNL
        pnl.total_pnl = pnl.realized_pnl + pnl.unrealized_pnl

        return pnl

    def calculate_total_volume(self, wallet_address: str) -> float:
        """
        Calculate total USD trading volume for a trader.

        Volume = sum(amount * price) for all trades.

        Args:
            wallet_address: Trader's wallet address.

        Returns:
            Total volume in USD.
        """
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT COALESCE(SUM(amount * price), 0) as total_volume
                FROM bets
                WHERE wallet_address = ?
                """,
                (wallet_address,)
            )
            row = cursor.fetchone()
            return float(row["total_volume"]) if row else 0.0

    def get_trader_stats(self, wallet_address: str) -> TraderStats:
        """
        Get comprehensive statistics for a trader.

        Args:
            wallet_address: Trader's wallet address.

        Returns:
            TraderStats with all metrics.
        """
        stats = TraderStats(wallet_address=wallet_address)

        with self.db.get_connection() as conn:
            # Basic stats
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as trade_count,
                    COALESCE(SUM(amount * price), 0) as total_volume,
                    COALESCE(AVG(amount), 0) as avg_bet_size,
                    COALESCE(AVG(price), 0) as avg_price,
                    MIN(timestamp) as first_trade,
                    MAX(timestamp) as last_trade,
                    COUNT(DISTINCT market_id) as markets_participated
                FROM bets
                WHERE wallet_address = ?
                """,
                (wallet_address,)
            )
            row = cursor.fetchone()

            if row:
                stats.total_trades = row["trade_count"]
                stats.total_volume = float(row["total_volume"])
                stats.avg_bet_size = float(row["avg_bet_size"])
                stats.avg_price = float(row["avg_price"])
                stats.markets_participated = row["markets_participated"]

                if row["first_trade"]:
                    stats.first_trade = row["first_trade"] if isinstance(row["first_trade"], datetime) else datetime.fromisoformat(str(row["first_trade"]))
                if row["last_trade"]:
                    stats.last_trade = row["last_trade"] if isinstance(row["last_trade"], datetime) else datetime.fromisoformat(str(row["last_trade"]))

            # Buy/Sell breakdown
            cursor = conn.execute(
                """
                SELECT
                    side,
                    COUNT(*) as count,
                    COALESCE(SUM(amount * price), 0) as volume
                FROM bets
                WHERE wallet_address = ?
                GROUP BY side
                """,
                (wallet_address,)
            )
            for row in cursor.fetchall():
                if row["side"].upper() == "BUY":
                    stats.buy_count = row["count"]
                    stats.buy_volume = float(row["volume"])
                else:
                    stats.sell_count = row["count"]
                    stats.sell_volume = float(row["volume"])

        # Calculate PNL
        stats.pnl = self.calculate_trader_pnl(wallet_address)

        return stats

    def get_top_traders(
        self,
        metric: LeaderboardMetric = LeaderboardMetric.PNL,
        limit: int = 20,
        min_trades: int = 1,
        min_volume: float = 0.0
    ) -> list[LeaderboardEntry]:
        """
        Get top traders ranked by specified metric.

        Args:
            metric: Ranking metric (PNL, VOLUME, TRADE_COUNT, WIN_RATE, ROI).
            limit: Maximum number of traders to return.
            min_trades: Minimum trades required for inclusion.
            min_volume: Minimum volume required for inclusion.

        Returns:
            List of LeaderboardEntry sorted by metric.
        """
        entries: list[LeaderboardEntry] = []

        with self.db.get_connection() as conn:
            # Get all qualifying traders
            cursor = conn.execute(
                """
                SELECT DISTINCT wallet_address
                FROM traders
                WHERE total_trades >= ? AND total_volume >= ?
                """,
                (min_trades, min_volume)
            )
            traders = [row["wallet_address"] for row in cursor.fetchall()]

        # Calculate stats for each trader
        for wallet in traders:
            try:
                stats = self.get_trader_stats(wallet)

                entry = LeaderboardEntry(
                    rank=0,  # Will be set after sorting
                    wallet_address=wallet,
                    display_address=f"{wallet[:6]}...{wallet[-4:]}" if len(wallet) > 12 else wallet,
                    total_pnl=float(stats.pnl.total_pnl) if stats.pnl else 0.0,
                    realized_pnl=float(stats.pnl.realized_pnl) if stats.pnl else 0.0,
                    unrealized_pnl=float(stats.pnl.unrealized_pnl) if stats.pnl else 0.0,
                    total_volume=stats.total_volume,
                    total_trades=stats.total_trades,
                    win_rate=stats.pnl.win_rate if stats.pnl else 0.0,
                    roi=stats.pnl.roi if stats.pnl else 0.0,
                    markets_participated=stats.markets_participated,
                    avg_bet_size=stats.avg_bet_size,
                    last_active=stats.last_trade,
                )
                entries.append(entry)
            except Exception as e:
                logger.error(f"Error calculating stats for {wallet}: {e}")
                continue

        # Sort by metric
        if metric == LeaderboardMetric.PNL:
            entries.sort(key=lambda x: x.total_pnl, reverse=True)
        elif metric == LeaderboardMetric.VOLUME:
            entries.sort(key=lambda x: x.total_volume, reverse=True)
        elif metric == LeaderboardMetric.TRADE_COUNT:
            entries.sort(key=lambda x: x.total_trades, reverse=True)
        elif metric == LeaderboardMetric.WIN_RATE:
            entries.sort(key=lambda x: x.win_rate, reverse=True)
        elif metric == LeaderboardMetric.ROI:
            entries.sort(key=lambda x: x.roi, reverse=True)

        # Assign ranks and limit
        for i, entry in enumerate(entries[:limit]):
            entry.rank = i + 1

        return entries[:limit]

    def get_leaderboard_summary(self) -> dict:
        """
        Get summary statistics for the leaderboard.

        Returns:
            Dictionary with aggregate metrics.
        """
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT
                    COUNT(DISTINCT wallet_address) as total_traders,
                    COUNT(*) as total_trades,
                    COALESCE(SUM(amount * price), 0) as total_volume,
                    COALESCE(AVG(amount * price), 0) as avg_trade_size,
                    COUNT(DISTINCT market_id) as markets_traded
                FROM bets
                """
            )
            row = cursor.fetchone()

            return {
                "total_traders": row["total_traders"],
                "total_trades": row["total_trades"],
                "total_volume": float(row["total_volume"]),
                "avg_trade_size": float(row["avg_trade_size"]),
                "markets_traded": row["markets_traded"],
                "generated_at": datetime.utcnow().isoformat(),
            }


# Global analytics instance
analytics = PolymarketAnalytics()
