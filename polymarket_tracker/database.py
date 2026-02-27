"""
SQLite database management for Polymarket Tracker.

This module handles all database operations including:
- Schema creation and migrations
- CRUD operations for traders, markets, bets, and positions
- Statistics aggregation and updates
"""

import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional, Generator

from .config import settings

logger = logging.getLogger(__name__)


# SQL schema for all tables
SCHEMA = """
-- Traders table: tracks unique wallet addresses and their activity
CREATE TABLE IF NOT EXISTS traders (
    wallet_address TEXT PRIMARY KEY,
    total_volume REAL DEFAULT 0.0,
    total_trades INTEGER DEFAULT 0,
    total_profit_loss REAL DEFAULT 0.0,
    first_seen TIMESTAMP NOT NULL,
    last_active TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Markets table: stores prediction market information
CREATE TABLE IF NOT EXISTS markets (
    market_id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    description TEXT,
    end_date TIMESTAMP,
    active BOOLEAN DEFAULT 1,
    closed BOOLEAN DEFAULT 0,
    resolved BOOLEAN DEFAULT 0,
    outcome TEXT,
    volume REAL DEFAULT 0.0,
    liquidity REAL,
    category TEXT,
    slug TEXT,
    token_ids TEXT,  -- JSON array of token IDs
    outcome_prices TEXT,  -- JSON array of prices
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Bets table: stores individual trade/bet records
CREATE TABLE IF NOT EXISTS bets (
    bet_id TEXT PRIMARY KEY,
    wallet_address TEXT NOT NULL,
    market_id TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    amount REAL NOT NULL,
    price REAL NOT NULL,
    side TEXT NOT NULL,  -- BUY or SELL
    outcome_bet TEXT NOT NULL,
    fee_rate_bps REAL,
    status TEXT,
    timestamp TIMESTAMP NOT NULL,
    transaction_hash TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (wallet_address) REFERENCES traders(wallet_address),
    FOREIGN KEY (market_id) REFERENCES markets(market_id)
);

-- Positions table: tracks current positions for each wallet/market pair
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    wallet_address TEXT NOT NULL,
    market_id TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    current_position REAL NOT NULL,
    entry_price REAL NOT NULL,
    current_value REAL,
    pnl REAL,
    realized_pnl REAL,
    unrealized_pnl REAL,
    outcome TEXT,
    last_updated TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(wallet_address, market_id, asset_id),
    FOREIGN KEY (wallet_address) REFERENCES traders(wallet_address),
    FOREIGN KEY (market_id) REFERENCES markets(market_id)
);

-- Sync state table: tracks last sync timestamps for incremental updates
CREATE TABLE IF NOT EXISTS sync_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trader statistics cache table: stores computed win rate and streak data
CREATE TABLE IF NOT EXISTS trader_stats_cache (
    wallet_address TEXT PRIMARY KEY,
    total_wins INTEGER DEFAULT 0,
    total_losses INTEGER DEFAULT 0,
    total_ties INTEGER DEFAULT 0,
    win_rate REAL DEFAULT 0.0,
    current_streak INTEGER DEFAULT 0,
    current_streak_type TEXT DEFAULT 'none',
    longest_winning_streak INTEGER DEFAULT 0,
    longest_losing_streak INTEGER DEFAULT 0,
    sharpe_ratio REAL,
    max_drawdown REAL DEFAULT 0.0,
    risk_adjusted_return REAL DEFAULT 0.0,
    last_calculated TIMESTAMP,
    FOREIGN KEY (wallet_address) REFERENCES traders(wallet_address)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_bets_wallet ON bets(wallet_address);
CREATE INDEX IF NOT EXISTS idx_bets_market ON bets(market_id);
CREATE INDEX IF NOT EXISTS idx_bets_timestamp ON bets(timestamp);
CREATE INDEX IF NOT EXISTS idx_positions_wallet ON positions(wallet_address);
CREATE INDEX IF NOT EXISTS idx_positions_market ON positions(market_id);
CREATE INDEX IF NOT EXISTS idx_traders_volume ON traders(total_volume DESC);
CREATE INDEX IF NOT EXISTS idx_traders_last_active ON traders(last_active DESC);
CREATE INDEX IF NOT EXISTS idx_markets_active ON markets(active, closed);
CREATE INDEX IF NOT EXISTS idx_markets_volume ON markets(volume DESC);
CREATE INDEX IF NOT EXISTS idx_markets_resolved ON markets(resolved);
CREATE INDEX IF NOT EXISTS idx_markets_category ON markets(category);
CREATE INDEX IF NOT EXISTS idx_stats_cache_win_rate ON trader_stats_cache(win_rate DESC);
CREATE INDEX IF NOT EXISTS idx_stats_cache_streak ON trader_stats_cache(longest_winning_streak DESC);

-- Whale profiles table: tracks identified whale wallets
CREATE TABLE IF NOT EXISTS whale_profiles (
    wallet_address TEXT PRIMARY KEY,
    grade TEXT DEFAULT 'ungraded',
    win_rate REAL DEFAULT 0.0,
    roi REAL DEFAULT 0.0,
    sharpe_ratio REAL,
    category_specialization TEXT,  -- JSON: {"politics": 65.0, "crypto": 45.0}
    avg_bet_timing REAL,  -- Average hours before resolution
    activity_pattern TEXT DEFAULT 'unknown',  -- steady / dormant_burst / sporadic
    total_volume REAL DEFAULT 0.0,
    total_resolved_bets INTEGER DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (wallet_address) REFERENCES traders(wallet_address)
);

-- Signals table: stores fired trading signals
CREATE TABLE IF NOT EXISTS signals (
    signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    outcome TEXT NOT NULL,
    conviction_score REAL NOT NULL,
    smart_money_score REAL DEFAULT 0.0,
    volume_spike_score REAL DEFAULT 0.0,
    cluster_score REAL DEFAULT 0.0,
    contributing_wallets TEXT,  -- JSON array of wallet addresses
    details TEXT,  -- JSON breakdown
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    market_resolved BOOLEAN DEFAULT 0,
    signal_correct BOOLEAN,
    FOREIGN KEY (market_id) REFERENCES markets(market_id)
);

CREATE INDEX IF NOT EXISTS idx_whale_profiles_grade ON whale_profiles(grade);
CREATE INDEX IF NOT EXISTS idx_whale_profiles_volume ON whale_profiles(total_volume DESC);
CREATE INDEX IF NOT EXISTS idx_signals_market ON signals(market_id);
CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_conviction ON signals(conviction_score DESC);
"""


class Database:
    """SQLite database manager for Polymarket data."""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file. Uses settings default if not provided.
        """
        self.db_path = db_path or settings.database_path
        self._ensure_db_directory()
        self._init_schema()

    def _ensure_db_directory(self) -> None:
        """Ensure the database directory exists."""
        db_dir = Path(self.db_path).parent
        if db_dir and not db_dir.exists():
            db_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Get a database connection with automatic cleanup.

        Yields:
            sqlite3.Connection: Database connection.
        """
        conn = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self.get_connection() as conn:
            conn.executescript(SCHEMA)
        logger.info(f"Database initialized at {self.db_path}")

    # ========== Trader Operations ==========

    def upsert_trader(
        self,
        wallet_address: str,
        volume: float = 0.0,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Insert or update a trader record.

        Args:
            wallet_address: Trader's wallet address.
            volume: Trade volume to add.
            timestamp: Trade timestamp.
        """
        now = timestamp or datetime.utcnow()
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO traders (wallet_address, total_volume, total_trades, first_seen, last_active)
                VALUES (?, ?, 1, ?, ?)
                ON CONFLICT(wallet_address) DO UPDATE SET
                    total_volume = total_volume + excluded.total_volume,
                    total_trades = total_trades + 1,
                    last_active = MAX(last_active, excluded.last_active),
                    updated_at = CURRENT_TIMESTAMP
                """,
                (wallet_address, volume, now, now)
            )

    def get_trader(self, wallet_address: str) -> Optional[dict]:
        """
        Get trader by wallet address.

        Args:
            wallet_address: Trader's wallet address.

        Returns:
            Trader record as dict or None if not found.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM traders WHERE wallet_address = ?",
                (wallet_address,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_top_traders(self, limit: int = 100) -> list[dict]:
        """
        Get top traders by volume.

        Args:
            limit: Maximum number of traders to return.

        Returns:
            List of trader records.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM traders ORDER BY total_volume DESC LIMIT ?",
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]

    # ========== Market Operations ==========

    def upsert_market(
        self,
        market_id: str,
        question: str,
        description: Optional[str] = None,
        end_date: Optional[datetime] = None,
        active: bool = True,
        closed: bool = False,
        resolved: bool = False,
        outcome: Optional[str] = None,
        volume: float = 0.0,
        liquidity: Optional[float] = None,
        category: Optional[str] = None,
        slug: Optional[str] = None,
        token_ids: Optional[str] = None,
        outcome_prices: Optional[str] = None
    ) -> None:
        """
        Insert or update a market record.

        Args:
            market_id: Market's condition ID.
            question: Market question text.
            description: Optional description.
            end_date: Market end date.
            active: Whether market is active.
            closed: Whether market is closed.
            resolved: Whether market is resolved.
            outcome: Resolved outcome.
            volume: Total trading volume.
            liquidity: Current liquidity.
            category: Market category.
            slug: URL slug.
            token_ids: JSON array of token IDs.
            outcome_prices: JSON array of outcome prices.
        """
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO markets (
                    market_id, question, description, end_date, active, closed,
                    resolved, outcome, volume, liquidity, category, slug,
                    token_ids, outcome_prices
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(market_id) DO UPDATE SET
                    question = excluded.question,
                    description = COALESCE(excluded.description, description),
                    end_date = COALESCE(excluded.end_date, end_date),
                    active = excluded.active,
                    closed = excluded.closed,
                    resolved = excluded.resolved,
                    outcome = COALESCE(excluded.outcome, outcome),
                    volume = excluded.volume,
                    liquidity = COALESCE(excluded.liquidity, liquidity),
                    category = COALESCE(excluded.category, category),
                    slug = COALESCE(excluded.slug, slug),
                    token_ids = COALESCE(excluded.token_ids, token_ids),
                    outcome_prices = COALESCE(excluded.outcome_prices, outcome_prices),
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    market_id, question, description, end_date, active, closed,
                    resolved, outcome, volume, liquidity, category, slug,
                    token_ids, outcome_prices
                )
            )

    def get_market(self, market_id: str) -> Optional[dict]:
        """
        Get market by ID.

        Args:
            market_id: Market's condition ID.

        Returns:
            Market record as dict or None if not found.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM markets WHERE market_id = ?",
                (market_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_active_markets(self, limit: int = 1000) -> list[dict]:
        """
        Get all active markets.

        Args:
            limit: Maximum number of markets to return.

        Returns:
            List of active market records.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM markets
                WHERE active = 1 AND closed = 0
                ORDER BY volume DESC
                LIMIT ?
                """,
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_resolved_markets_needing_trades(self, limit: int = 200) -> list[dict]:
        """
        Get resolved markets that have no trades in the bets table yet.

        These are markets we know the outcome for but haven't fetched
        trade history, which is needed for whale win-rate grading.

        Args:
            limit: Maximum number of markets to return.

        Returns:
            List of resolved market records with no associated bets.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT m.* FROM markets m
                LEFT JOIN (
                    SELECT DISTINCT market_id FROM bets
                ) b ON m.market_id = b.market_id
                WHERE m.resolved = 1
                  AND m.outcome IS NOT NULL
                  AND b.market_id IS NULL
                ORDER BY m.volume DESC
                LIMIT ?
                """,
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]

    # ========== Bet Operations ==========

    def insert_bet(
        self,
        bet_id: str,
        wallet_address: str,
        market_id: str,
        asset_id: str,
        amount: float,
        price: float,
        side: str,
        outcome_bet: str,
        timestamp: datetime,
        fee_rate_bps: Optional[float] = None,
        status: Optional[str] = None,
        transaction_hash: Optional[str] = None
    ) -> bool:
        """
        Insert a new bet record.

        Args:
            bet_id: Unique bet/trade ID.
            wallet_address: Trader's wallet address.
            market_id: Market's condition ID.
            asset_id: Asset/token ID.
            amount: Bet amount (size).
            price: Execution price.
            side: BUY or SELL.
            outcome_bet: Outcome being bet on.
            timestamp: Trade timestamp.
            fee_rate_bps: Fee rate in basis points.
            status: Trade status.
            transaction_hash: On-chain transaction hash.

        Returns:
            True if inserted, False if already exists.
        """
        try:
            with self.get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO bets (
                        bet_id, wallet_address, market_id, asset_id, amount,
                        price, side, outcome_bet, timestamp, fee_rate_bps,
                        status, transaction_hash
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        bet_id, wallet_address, market_id, asset_id, amount,
                        price, side, outcome_bet, timestamp, fee_rate_bps,
                        status, transaction_hash
                    )
                )
                return True
        except sqlite3.IntegrityError:
            # Bet already exists
            return False

    def get_bets_for_wallet(
        self,
        wallet_address: str,
        limit: int = 100
    ) -> list[dict]:
        """
        Get bets for a specific wallet.

        Args:
            wallet_address: Trader's wallet address.
            limit: Maximum number of bets to return.

        Returns:
            List of bet records.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM bets
                WHERE wallet_address = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (wallet_address, limit)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_bets_for_market(
        self,
        market_id: str,
        limit: int = 1000
    ) -> list[dict]:
        """
        Get bets for a specific market.

        Args:
            market_id: Market's condition ID.
            limit: Maximum number of bets to return.

        Returns:
            List of bet records.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM bets
                WHERE market_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (market_id, limit)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_recent_bets(self, limit: int = 100) -> list[dict]:
        """
        Get most recent bets across all markets.

        Args:
            limit: Maximum number of bets to return.

        Returns:
            List of bet records.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM bets ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]

    # ========== Position Operations ==========

    def upsert_position(
        self,
        wallet_address: str,
        market_id: str,
        asset_id: str,
        current_position: float,
        entry_price: float,
        current_value: Optional[float] = None,
        pnl: Optional[float] = None,
        realized_pnl: Optional[float] = None,
        unrealized_pnl: Optional[float] = None,
        outcome: Optional[str] = None
    ) -> None:
        """
        Insert or update a position record.

        Args:
            wallet_address: Trader's wallet address.
            market_id: Market's condition ID.
            asset_id: Asset/token ID.
            current_position: Current position size.
            entry_price: Average entry price.
            current_value: Current value of position.
            pnl: Total profit/loss.
            realized_pnl: Realized profit/loss.
            unrealized_pnl: Unrealized profit/loss.
            outcome: Position outcome.
        """
        now = datetime.utcnow()
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO positions (
                    wallet_address, market_id, asset_id, current_position,
                    entry_price, current_value, pnl, realized_pnl,
                    unrealized_pnl, outcome, last_updated
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(wallet_address, market_id, asset_id) DO UPDATE SET
                    current_position = excluded.current_position,
                    entry_price = excluded.entry_price,
                    current_value = excluded.current_value,
                    pnl = excluded.pnl,
                    realized_pnl = excluded.realized_pnl,
                    unrealized_pnl = excluded.unrealized_pnl,
                    outcome = excluded.outcome,
                    last_updated = excluded.last_updated
                """,
                (
                    wallet_address, market_id, asset_id, current_position,
                    entry_price, current_value, pnl, realized_pnl,
                    unrealized_pnl, outcome, now
                )
            )

    def get_positions_for_wallet(self, wallet_address: str) -> list[dict]:
        """
        Get all positions for a wallet.

        Args:
            wallet_address: Trader's wallet address.

        Returns:
            List of position records.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM positions
                WHERE wallet_address = ?
                ORDER BY last_updated DESC
                """,
                (wallet_address,)
            )
            return [dict(row) for row in cursor.fetchall()]

    # ========== Sync State Operations ==========

    def get_sync_state(self, key: str) -> Optional[str]:
        """
        Get sync state value.

        Args:
            key: State key.

        Returns:
            State value or None if not found.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT value FROM sync_state WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()
            return row["value"] if row else None

    def set_sync_state(self, key: str, value: str) -> None:
        """
        Set sync state value.

        Args:
            key: State key.
            value: State value.
        """
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO sync_state (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (key, value)
            )

    # ========== Statistics ==========

    def get_stats(self) -> dict:
        """
        Get database statistics.

        Returns:
            Dictionary with counts and totals.
        """
        with self.get_connection() as conn:
            stats = {}

            cursor = conn.execute("SELECT COUNT(*) as count FROM traders")
            stats["total_traders"] = cursor.fetchone()["count"]

            cursor = conn.execute("SELECT COUNT(*) as count FROM markets")
            stats["total_markets"] = cursor.fetchone()["count"]

            cursor = conn.execute(
                "SELECT COUNT(*) as count FROM markets WHERE active = 1 AND closed = 0"
            )
            stats["active_markets"] = cursor.fetchone()["count"]

            cursor = conn.execute("SELECT COUNT(*) as count FROM bets")
            stats["total_bets"] = cursor.fetchone()["count"]

            cursor = conn.execute("SELECT SUM(amount) as total FROM bets")
            row = cursor.fetchone()
            stats["total_volume"] = row["total"] or 0.0

            cursor = conn.execute("SELECT COUNT(*) as count FROM positions")
            stats["total_positions"] = cursor.fetchone()["count"]

            return stats

    # ========== Whale Profile Operations ==========

    def upsert_whale_profile(
        self,
        wallet_address: str,
        grade: str = "ungraded",
        win_rate: float = 0.0,
        roi: float = 0.0,
        sharpe_ratio: Optional[float] = None,
        category_specialization: Optional[str] = None,
        avg_bet_timing: Optional[float] = None,
        activity_pattern: str = "unknown",
        total_volume: float = 0.0,
        total_resolved_bets: int = 0
    ) -> None:
        """
        Insert or update a whale profile.

        Args:
            wallet_address: Whale's wallet address.
            grade: Performance grade (A/B/C/D/ungraded).
            win_rate: Win rate as a decimal.
            roi: Return on investment.
            sharpe_ratio: Risk-adjusted return ratio.
            category_specialization: JSON string of category weights.
            avg_bet_timing: Average hours before resolution.
            activity_pattern: Trading pattern (steady/dormant_burst/sporadic).
            total_volume: Total trading volume.
            total_resolved_bets: Number of resolved bets.
        """
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO whale_profiles (
                    wallet_address, grade, win_rate, roi, sharpe_ratio,
                    category_specialization, avg_bet_timing, activity_pattern,
                    total_volume, total_resolved_bets
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(wallet_address) DO UPDATE SET
                    grade = excluded.grade,
                    win_rate = excluded.win_rate,
                    roi = excluded.roi,
                    sharpe_ratio = excluded.sharpe_ratio,
                    category_specialization = excluded.category_specialization,
                    avg_bet_timing = excluded.avg_bet_timing,
                    activity_pattern = excluded.activity_pattern,
                    total_volume = excluded.total_volume,
                    total_resolved_bets = excluded.total_resolved_bets,
                    last_updated = CURRENT_TIMESTAMP
                """,
                (
                    wallet_address, grade, win_rate, roi, sharpe_ratio,
                    category_specialization, avg_bet_timing, activity_pattern,
                    total_volume, total_resolved_bets
                )
            )

    def get_whale_profile(self, wallet_address: str) -> Optional[dict]:
        """
        Get a single whale profile by wallet address.

        Args:
            wallet_address: Whale's wallet address.

        Returns:
            Whale profile as dict or None if not found.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM whale_profiles WHERE wallet_address = ?",
                (wallet_address,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_whale_profiles(
        self,
        min_grade: Optional[str] = None,
        limit: int = 100
    ) -> list[dict]:
        """
        Get whale profiles, optionally filtered by minimum grade.

        Grade ordering: A=1, B=2, C=3, D=4, ungraded=5.
        If min_grade="B", returns profiles with grade A or B.

        Args:
            min_grade: Minimum grade to include (e.g. "B" returns A and B).
            limit: Maximum number of profiles to return.

        Returns:
            List of whale profile records ordered by total_volume DESC.
        """
        grade_order = {"A": 1, "B": 2, "C": 3, "D": 4, "ungraded": 5}

        with self.get_connection() as conn:
            if min_grade and min_grade in grade_order:
                max_rank = grade_order[min_grade]
                allowed_grades = [g for g, r in grade_order.items() if r <= max_rank]
                placeholders = ",".join("?" for _ in allowed_grades)
                cursor = conn.execute(
                    f"""
                    SELECT * FROM whale_profiles
                    WHERE grade IN ({placeholders})
                    ORDER BY total_volume DESC
                    LIMIT ?
                    """,
                    (*allowed_grades, limit)
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM whale_profiles ORDER BY total_volume DESC LIMIT ?",
                    (limit,)
                )
            return [dict(row) for row in cursor.fetchall()]

    # ========== Signal Operations ==========

    def insert_signal(
        self,
        market_id: str,
        outcome: str,
        conviction_score: float,
        smart_money_score: float = 0.0,
        volume_spike_score: float = 0.0,
        cluster_score: float = 0.0,
        contributing_wallets: Optional[str] = None,
        details: Optional[str] = None
    ) -> int:
        """
        Insert a new trading signal.

        Args:
            market_id: Market's condition ID.
            outcome: Signal outcome direction.
            conviction_score: Overall conviction score.
            smart_money_score: Smart money component score.
            volume_spike_score: Volume spike component score.
            cluster_score: Cluster activity component score.
            contributing_wallets: JSON array of wallet addresses.
            details: JSON breakdown of signal details.

        Returns:
            The signal_id of the inserted record.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO signals (
                    market_id, outcome, conviction_score, smart_money_score,
                    volume_spike_score, cluster_score, contributing_wallets, details
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    market_id, outcome, conviction_score, smart_money_score,
                    volume_spike_score, cluster_score, contributing_wallets, details
                )
            )
            return cursor.lastrowid

    def get_recent_signals(
        self,
        limit: int = 50,
        min_score: float = 0.0
    ) -> list[dict]:
        """
        Get recent signals with joined market data.

        Args:
            limit: Maximum number of signals to return.
            min_score: Minimum conviction score to include.

        Returns:
            List of signal records with market question/category/end_date/volume.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT s.*, m.question, m.category, m.end_date, m.volume AS market_volume
                FROM signals s
                LEFT JOIN markets m ON s.market_id = m.market_id
                WHERE s.conviction_score >= ?
                ORDER BY s.timestamp DESC
                LIMIT ?
                """,
                (min_score, limit)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_signals_for_market(self, market_id: str) -> list[dict]:
        """
        Get all signals for a specific market.

        Args:
            market_id: Market's condition ID.

        Returns:
            List of signal records ordered by timestamp DESC.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM signals WHERE market_id = ? ORDER BY timestamp DESC",
                (market_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_hot_markets(
        self,
        hours: int = 24,
        limit: int = 20
    ) -> list[dict]:
        """
        Get markets with the most signal activity in the last N hours.

        Args:
            hours: Lookback window in hours.
            limit: Maximum number of markets to return.

        Returns:
            List of dicts with signal_count, max_conviction, avg_conviction,
            latest_signal, plus market question/category/end_date/volume.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT
                    s.market_id,
                    COUNT(*) AS signal_count,
                    MAX(s.conviction_score) AS max_conviction,
                    AVG(s.conviction_score) AS avg_conviction,
                    MAX(s.timestamp) AS latest_signal,
                    m.question,
                    m.category,
                    m.end_date,
                    m.volume AS market_volume
                FROM signals s
                LEFT JOIN markets m ON s.market_id = m.market_id
                WHERE s.timestamp >= datetime('now', ? || ' hours')
                GROUP BY s.market_id
                ORDER BY signal_count DESC, max_conviction DESC
                LIMIT ?
                """,
                (str(-hours), limit)
            )
            return [dict(row) for row in cursor.fetchall()]


# Global database instance
db = Database()
