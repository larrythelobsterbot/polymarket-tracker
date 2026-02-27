# Whale Signal Dashboard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add whale wallet profiling, a composite signal scoring engine, and a live web dashboard to surface unusual large bets as forward-looking trading signals.

**Architecture:** Three new modules (`whale_profiler.py`, `signal_engine.py`, `web/`) layered on top of the existing SQLite database, collector, and insider detection system. The signal engine runs inline after each trade sync. The web dashboard is a separate FastAPI process reading the same database.

**Tech Stack:** Python 3.10+, FastAPI, Jinja2, HTMX, WebSocket (via `starlette`), existing SQLite database

---

### Task 1: Add Dependencies

**Files:**
- Modify: `requirements.txt`

**Step 1: Add FastAPI and related packages to requirements.txt**

Add these lines to the end of `requirements.txt` (before the dev dependencies section):

```
# Web dashboard dependencies
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
jinja2>=3.1.0
python-multipart>=0.0.9
```

**Step 2: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: All packages install successfully

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "feat: add FastAPI dependencies for web dashboard"
```

---

### Task 2: Database Schema — whale_profiles and signals tables

**Files:**
- Modify: `polymarket_tracker/database.py`
- Test: `tests/test_database.py`

**Step 1: Write failing tests for new tables**

Add to `tests/test_database.py`:

```python
def test_whale_profiles_table_exists(tmp_db):
    """whale_profiles table should be created on init."""
    with tmp_db.get_connection() as conn:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='whale_profiles'"
        )
        assert cursor.fetchone() is not None


def test_upsert_whale_profile(tmp_db):
    """Should insert and update whale profiles."""
    # First, create the trader record (FK dependency)
    tmp_db.upsert_trader("0xwhale1", volume=100000.0)

    tmp_db.upsert_whale_profile(
        wallet_address="0xwhale1",
        grade="B",
        win_rate=58.5,
        roi=12.3,
        sharpe_ratio=1.4,
        category_specialization='{"politics": 65.0, "crypto": 45.0}',
        avg_bet_timing=18.5,
        activity_pattern="steady",
        total_volume=100000.0,
        total_resolved_bets=25,
    )
    profile = tmp_db.get_whale_profile("0xwhale1")
    assert profile is not None
    assert profile["grade"] == "B"
    assert profile["win_rate"] == 58.5
    assert profile["total_resolved_bets"] == 25

    # Update should work
    tmp_db.upsert_whale_profile(
        wallet_address="0xwhale1",
        grade="A",
        win_rate=67.0,
        roi=15.0,
        sharpe_ratio=1.8,
        total_volume=150000.0,
        total_resolved_bets=30,
    )
    profile = tmp_db.get_whale_profile("0xwhale1")
    assert profile["grade"] == "A"
    assert profile["win_rate"] == 67.0


def test_get_whale_profiles(tmp_db):
    """Should return whale profiles filtered by grade."""
    tmp_db.upsert_trader("0xwhaleA", volume=200000.0)
    tmp_db.upsert_trader("0xwhaleB", volume=100000.0)
    tmp_db.upsert_trader("0xwhaleC", volume=80000.0)

    tmp_db.upsert_whale_profile("0xwhaleA", grade="A", win_rate=70.0, total_volume=200000.0, total_resolved_bets=30)
    tmp_db.upsert_whale_profile("0xwhaleB", grade="B", win_rate=58.0, total_volume=100000.0, total_resolved_bets=25)
    tmp_db.upsert_whale_profile("0xwhaleC", grade="C", win_rate=48.0, total_volume=80000.0, total_resolved_bets=22)

    # All whales
    all_whales = tmp_db.get_whale_profiles()
    assert len(all_whales) == 3

    # Only grade B+
    good_whales = tmp_db.get_whale_profiles(min_grade="B")
    assert len(good_whales) == 2


def test_signals_table_exists(tmp_db):
    """signals table should be created on init."""
    with tmp_db.get_connection() as conn:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='signals'"
        )
        assert cursor.fetchone() is not None


def test_insert_signal(tmp_db):
    """Should insert and retrieve signals."""
    tmp_db.upsert_trader("0xwhale1", volume=100000.0)
    tmp_db.upsert_market(market_id="mkt1", question="Will X happen?")

    signal_id = tmp_db.insert_signal(
        market_id="mkt1",
        outcome="Yes",
        conviction_score=75.0,
        smart_money_score=80.0,
        volume_spike_score=60.0,
        cluster_score=0.0,
        contributing_wallets='["0xwhale1"]',
        details='{"reason": "A-grade whale bet $50k on Yes"}',
    )
    assert signal_id is not None

    signals = tmp_db.get_recent_signals(limit=10)
    assert len(signals) == 1
    assert signals[0]["conviction_score"] == 75.0
    assert signals[0]["outcome"] == "Yes"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_database.py::test_whale_profiles_table_exists tests/test_database.py::test_signals_table_exists -v`
Expected: FAIL — tables don't exist yet

**Step 3: Add schema and CRUD methods to database.py**

In `polymarket_tracker/database.py`, append to the `SCHEMA` string (before the closing `"""`), after the existing indexes:

```sql
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
```

Then add these methods to the `Database` class, after the existing `get_stats` method:

```python
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
        total_resolved_bets: int = 0,
    ) -> None:
        """Insert or update a whale profile."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO whale_profiles (
                    wallet_address, grade, win_rate, roi, sharpe_ratio,
                    category_specialization, avg_bet_timing, activity_pattern,
                    total_volume, total_resolved_bets, last_updated
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(wallet_address) DO UPDATE SET
                    grade = excluded.grade,
                    win_rate = excluded.win_rate,
                    roi = excluded.roi,
                    sharpe_ratio = COALESCE(excluded.sharpe_ratio, sharpe_ratio),
                    category_specialization = COALESCE(excluded.category_specialization, category_specialization),
                    avg_bet_timing = COALESCE(excluded.avg_bet_timing, avg_bet_timing),
                    activity_pattern = COALESCE(excluded.activity_pattern, activity_pattern),
                    total_volume = excluded.total_volume,
                    total_resolved_bets = excluded.total_resolved_bets,
                    last_updated = CURRENT_TIMESTAMP
                """,
                (
                    wallet_address, grade, win_rate, roi, sharpe_ratio,
                    category_specialization, avg_bet_timing, activity_pattern,
                    total_volume, total_resolved_bets,
                ),
            )

    def get_whale_profile(self, wallet_address: str) -> Optional[dict]:
        """Get whale profile by wallet address."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM whale_profiles WHERE wallet_address = ?",
                (wallet_address,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_whale_profiles(
        self,
        min_grade: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Get whale profiles, optionally filtered by minimum grade."""
        grade_order = {"A": 1, "B": 2, "C": 3, "D": 4, "ungraded": 5}
        with self.get_connection() as conn:
            if min_grade and min_grade in grade_order:
                max_rank = grade_order[min_grade]
                allowed = [g for g, r in grade_order.items() if r <= max_rank]
                placeholders = ",".join("?" for _ in allowed)
                cursor = conn.execute(
                    f"""
                    SELECT * FROM whale_profiles
                    WHERE grade IN ({placeholders})
                    ORDER BY total_volume DESC
                    LIMIT ?
                    """,
                    (*allowed, limit),
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM whale_profiles ORDER BY total_volume DESC LIMIT ?",
                    (limit,),
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
        details: Optional[str] = None,
    ) -> int:
        """Insert a new signal and return its ID."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO signals (
                    market_id, outcome, conviction_score,
                    smart_money_score, volume_spike_score, cluster_score,
                    contributing_wallets, details
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    market_id, outcome, conviction_score,
                    smart_money_score, volume_spike_score, cluster_score,
                    contributing_wallets, details,
                ),
            )
            return cursor.lastrowid

    def get_recent_signals(
        self,
        limit: int = 50,
        min_score: float = 0.0,
    ) -> list[dict]:
        """Get recent signals, optionally filtered by minimum conviction score."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT s.*, m.question as market_question, m.category, m.end_date, m.volume as market_volume
                FROM signals s
                LEFT JOIN markets m ON s.market_id = m.market_id
                WHERE s.conviction_score >= ?
                ORDER BY s.timestamp DESC
                LIMIT ?
                """,
                (min_score, limit),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_signals_for_market(self, market_id: str) -> list[dict]:
        """Get all signals for a specific market."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM signals
                WHERE market_id = ?
                ORDER BY timestamp DESC
                """,
                (market_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_hot_markets(self, hours: int = 24, limit: int = 20) -> list[dict]:
        """Get markets with the most signal activity in the last N hours."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT
                    s.market_id,
                    m.question,
                    m.category,
                    m.end_date,
                    m.volume,
                    COUNT(*) as signal_count,
                    MAX(s.conviction_score) as max_conviction,
                    AVG(s.conviction_score) as avg_conviction,
                    MAX(s.timestamp) as latest_signal
                FROM signals s
                LEFT JOIN markets m ON s.market_id = m.market_id
                WHERE s.timestamp >= datetime('now', ? || ' hours')
                GROUP BY s.market_id
                ORDER BY signal_count DESC, max_conviction DESC
                LIMIT ?
                """,
                (f"-{hours}", limit),
            )
            return [dict(row) for row in cursor.fetchall()]
```

**Step 4: Run all database tests**

Run: `pytest tests/test_database.py -v`
Expected: All tests PASS (including the new ones)

**Step 5: Commit**

```bash
git add polymarket_tracker/database.py tests/test_database.py
git commit -m "feat: add whale_profiles and signals tables with CRUD operations"
```

---

### Task 3: Whale Profiler Module

**Files:**
- Create: `polymarket_tracker/whale_profiler.py`
- Test: `tests/test_whale_profiler.py`

**Step 1: Write failing tests**

Create `tests/test_whale_profiler.py`:

```python
"""Tests for whale profiler module."""

import json
import pytest
from datetime import datetime, timedelta
from polymarket_tracker.whale_profiler import WhaleProfiler, WhaleConfig


@pytest.fixture
def profiler(tmp_db):
    """Create a WhaleProfiler with test database."""
    return WhaleProfiler(database=tmp_db)


@pytest.fixture
def populated_db(tmp_db):
    """Database with sample whale data."""
    # Create a whale with high volume and good win rate
    tmp_db.upsert_trader("0xwhale_good", volume=100000.0)

    # Add a resolved market that the whale won
    tmp_db.upsert_market(
        market_id="mkt_resolved_1",
        question="Will BTC hit 100k?",
        resolved=True,
        outcome="Yes",
        volume=500000.0,
        category="crypto",
    )

    # Whale bought Yes at 0.40 — market resolved Yes — whale won
    for i in range(15):
        tmp_db.insert_bet(
            bet_id=f"bet_win_{i}",
            wallet_address="0xwhale_good",
            market_id="mkt_resolved_1",
            asset_id="asset1",
            amount=500.0,
            price=0.40,
            side="BUY",
            outcome_bet="Yes",
            timestamp=datetime.utcnow() - timedelta(days=30),
        )

    # Add resolved markets the whale lost
    tmp_db.upsert_market(
        market_id="mkt_resolved_2",
        question="Will ETH flip BTC?",
        resolved=True,
        outcome="No",
        volume=200000.0,
        category="crypto",
    )
    for i in range(5):
        tmp_db.insert_bet(
            bet_id=f"bet_lose_{i}",
            wallet_address="0xwhale_good",
            market_id="mkt_resolved_2",
            asset_id="asset2",
            amount=300.0,
            price=0.60,
            side="BUY",
            outcome_bet="Yes",
            timestamp=datetime.utcnow() - timedelta(days=20),
        )

    return tmp_db


def test_classify_grade_a():
    """65%+ win rate with 20+ bets = grade A."""
    grade = WhaleProfiler.classify_grade(win_rate=70.0, resolved_bets=25)
    assert grade == "A"


def test_classify_grade_ungraded():
    """Less than 20 resolved bets = ungraded."""
    grade = WhaleProfiler.classify_grade(win_rate=90.0, resolved_bets=10)
    assert grade == "ungraded"


def test_identify_whales_by_volume(profiler, tmp_db):
    """Wallets above volume threshold should be identified as whales."""
    tmp_db.upsert_trader("0xbig", volume=60000.0)
    tmp_db.upsert_trader("0xsmall", volume=5000.0)

    whale_addresses = profiler.identify_whale_addresses()
    assert "0xbig" in whale_addresses
    assert "0xsmall" not in whale_addresses


def test_calculate_win_rate_resolved_only(profiler, populated_db):
    """Win rate should only count resolved decisive markets."""
    stats = profiler.calculate_wallet_stats("0xwhale_good")
    # Won mkt_resolved_1, lost mkt_resolved_2 = 1 win, 1 loss = 50%
    assert stats["win_rate"] == 50.0
    assert stats["resolved_bets"] == 2
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_whale_profiler.py -v`
Expected: FAIL — module doesn't exist

**Step 3: Implement whale_profiler.py**

Create `polymarket_tracker/whale_profiler.py`:

```python
"""
Whale Wallet Profiler for Polymarket Tracker.

Identifies high-volume wallets, calculates their historical accuracy
on resolved markets, and assigns performance grades.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from .database import Database, db

logger = logging.getLogger(__name__)


@dataclass
class WhaleConfig:
    """Configuration for whale identification."""
    VOLUME_THRESHOLD: float = 50000.0       # Cumulative USD volume to be a whale
    SINGLE_BET_THRESHOLD: float = 5000.0    # Single bet size to flag wallet
    MIN_RESOLVED_BETS: int = 20             # Minimum resolved bets for grading
    STALE_DAYS: int = 90                    # Days inactive to flag as stale

    # Grading thresholds (win rate %)
    GRADE_A_THRESHOLD: float = 65.0
    GRADE_B_THRESHOLD: float = 55.0
    GRADE_C_THRESHOLD: float = 45.0


class WhaleProfiler:
    """Identifies and profiles whale wallets."""

    def __init__(
        self,
        database: Optional[Database] = None,
        config: Optional[WhaleConfig] = None,
    ):
        self.db = database or db
        self.config = config or WhaleConfig()

    @staticmethod
    def classify_grade(win_rate: float, resolved_bets: int, min_bets: int = 20) -> str:
        """Assign a letter grade based on win rate and sample size."""
        if resolved_bets < min_bets:
            return "ungraded"
        if win_rate >= 65.0:
            return "A"
        elif win_rate >= 55.0:
            return "B"
        elif win_rate >= 45.0:
            return "C"
        else:
            return "D"

    def identify_whale_addresses(self) -> list[str]:
        """Find wallet addresses that qualify as whales."""
        addresses = []
        with self.db.get_connection() as conn:
            # Wallets above cumulative volume threshold
            cursor = conn.execute(
                "SELECT wallet_address FROM traders WHERE total_volume >= ?",
                (self.config.VOLUME_THRESHOLD,),
            )
            volume_whales = {row["wallet_address"] for row in cursor.fetchall()}
            addresses.extend(volume_whales)

            # Wallets with any single bet above threshold
            cursor = conn.execute(
                """
                SELECT DISTINCT wallet_address FROM bets
                WHERE (amount * price) >= ?
                """,
                (self.config.SINGLE_BET_THRESHOLD,),
            )
            big_bettors = {row["wallet_address"] for row in cursor.fetchall()}
            addresses.extend(a for a in big_bettors if a not in volume_whales)

        return addresses

    def calculate_wallet_stats(self, wallet_address: str) -> dict:
        """Calculate resolved-market win rate and stats for a wallet."""
        with self.db.get_connection() as conn:
            # Get all resolved markets this wallet bet on
            cursor = conn.execute(
                """
                SELECT
                    b.market_id,
                    b.outcome_bet,
                    b.side,
                    SUM(b.amount * b.price) as total_cost,
                    SUM(b.amount) as total_size,
                    m.outcome as market_outcome,
                    m.resolved,
                    m.category,
                    m.volume as market_volume
                FROM bets b
                JOIN markets m ON b.market_id = m.market_id
                WHERE b.wallet_address = ? AND m.resolved = 1 AND m.outcome IS NOT NULL
                GROUP BY b.market_id, b.outcome_bet
                """,
                (wallet_address,),
            )
            resolved_bets = cursor.fetchall()

            wins = 0
            losses = 0
            category_wins = {}
            category_totals = {}
            timing_hours = []

            # Track per-market to avoid double-counting
            market_results = {}
            for bet in resolved_bets:
                mid = bet["market_id"]
                if mid in market_results:
                    continue  # Already counted this market

                won = (
                    bet["outcome_bet"].lower() == bet["market_outcome"].lower()
                    if bet["side"].upper() == "BUY"
                    else bet["outcome_bet"].lower() != bet["market_outcome"].lower()
                )
                market_results[mid] = won

                cat = bet["category"] or "other"
                category_totals[cat] = category_totals.get(cat, 0) + 1

                if won:
                    wins += 1
                    category_wins[cat] = category_wins.get(cat, 0) + 1
                else:
                    losses += 1

            total_decisive = wins + losses
            win_rate = (wins / total_decisive * 100) if total_decisive > 0 else 0.0

            # Category specialization
            cat_spec = {}
            for cat, total in category_totals.items():
                cat_wins = category_wins.get(cat, 0)
                cat_spec[cat] = round(cat_wins / total * 100, 1) if total > 0 else 0.0

            # Volume
            cursor = conn.execute(
                "SELECT total_volume FROM traders WHERE wallet_address = ?",
                (wallet_address,),
            )
            row = cursor.fetchone()
            total_volume = float(row["total_volume"]) if row else 0.0

            # Activity pattern
            cursor = conn.execute(
                """
                SELECT MIN(timestamp) as first, MAX(timestamp) as last
                FROM bets WHERE wallet_address = ?
                """,
                (wallet_address,),
            )
            row = cursor.fetchone()
            activity_pattern = "unknown"
            if row and row["first"] and row["last"]:
                # Check for dormant gaps (30+ day gaps between bets)
                cursor2 = conn.execute(
                    """
                    SELECT timestamp FROM bets
                    WHERE wallet_address = ?
                    ORDER BY timestamp
                    """,
                    (wallet_address,),
                )
                timestamps = [r["timestamp"] for r in cursor2.fetchall()]
                if len(timestamps) >= 2:
                    max_gap_days = 0
                    for i in range(1, len(timestamps)):
                        t1 = timestamps[i - 1]
                        t2 = timestamps[i]
                        if isinstance(t1, str):
                            t1 = datetime.fromisoformat(t1)
                        if isinstance(t2, str):
                            t2 = datetime.fromisoformat(t2)
                        gap = (t2 - t1).days
                        max_gap_days = max(max_gap_days, gap)

                    if max_gap_days >= 30:
                        activity_pattern = "dormant_burst"
                    elif len(timestamps) >= 10:
                        activity_pattern = "steady"
                    else:
                        activity_pattern = "sporadic"

        return {
            "win_rate": round(win_rate, 1),
            "resolved_bets": total_decisive,
            "wins": wins,
            "losses": losses,
            "total_volume": total_volume,
            "category_specialization": cat_spec,
            "activity_pattern": activity_pattern,
        }

    def profile_whale(self, wallet_address: str) -> dict:
        """Build a complete whale profile for a wallet."""
        stats = self.calculate_wallet_stats(wallet_address)
        grade = self.classify_grade(
            win_rate=stats["win_rate"],
            resolved_bets=stats["resolved_bets"],
            min_bets=self.config.MIN_RESOLVED_BETS,
        )

        self.db.upsert_whale_profile(
            wallet_address=wallet_address,
            grade=grade,
            win_rate=stats["win_rate"],
            roi=0.0,  # Calculated separately via analytics
            category_specialization=json.dumps(stats["category_specialization"]),
            activity_pattern=stats["activity_pattern"],
            total_volume=stats["total_volume"],
            total_resolved_bets=stats["resolved_bets"],
        )

        return {
            "wallet_address": wallet_address,
            "grade": grade,
            **stats,
        }

    def refresh_all_profiles(self) -> int:
        """Re-profile all identified whales. Returns count of profiles updated."""
        addresses = self.identify_whale_addresses()
        updated = 0
        for addr in addresses:
            try:
                self.profile_whale(addr)
                updated += 1
            except Exception as e:
                logger.error(f"Error profiling {addr}: {e}")
        logger.info(f"Refreshed {updated} whale profiles")
        return updated
```

**Step 4: Run tests**

Run: `pytest tests/test_whale_profiler.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add polymarket_tracker/whale_profiler.py tests/test_whale_profiler.py
git commit -m "feat: add whale profiler with grade classification and stats"
```

---

### Task 4: Signal Engine Module

**Files:**
- Create: `polymarket_tracker/signal_engine.py`
- Test: `tests/test_signal_engine.py`

**Step 1: Write failing tests**

Create `tests/test_signal_engine.py`:

```python
"""Tests for signal engine module."""

import json
import pytest
from datetime import datetime, timedelta
from polymarket_tracker.signal_engine import SignalEngine, SignalConfig


@pytest.fixture
def engine(tmp_db):
    """Create a SignalEngine with test database."""
    return SignalEngine(database=tmp_db)


def test_smart_money_score_a_grade(engine, tmp_db):
    """A-grade whale bet should produce high smart money score."""
    tmp_db.upsert_trader("0xwhaleA", volume=200000.0)
    tmp_db.upsert_whale_profile(
        "0xwhaleA", grade="A", win_rate=70.0,
        total_volume=200000.0, total_resolved_bets=30,
    )

    score = engine.score_smart_money(
        wallet_address="0xwhaleA",
        bet_amount=10000.0,
        market_id="mkt1",
    )
    assert score >= 70  # A-grade whale with large bet = high score


def test_smart_money_score_no_profile(engine, tmp_db):
    """Non-whale should produce zero smart money score."""
    tmp_db.upsert_trader("0xnobody", volume=1000.0)

    score = engine.score_smart_money(
        wallet_address="0xnobody",
        bet_amount=100.0,
        market_id="mkt1",
    )
    assert score == 0


def test_volume_spike_detection(engine, tmp_db):
    """Bet 3x above rolling average should trigger volume spike."""
    tmp_db.upsert_market(market_id="mkt1", question="Test?", volume=500000.0)

    # Insert baseline bets (average ~$200)
    for i in range(20):
        tmp_db.insert_bet(
            bet_id=f"baseline_{i}",
            wallet_address="0xtrader",
            market_id="mkt1",
            asset_id="a1",
            amount=200.0,
            price=0.50,
            side="BUY",
            outcome_bet="Yes",
            timestamp=datetime.utcnow() - timedelta(days=3, hours=i),
        )

    # Big bet = 3x+ average
    score = engine.score_volume_spike(
        market_id="mkt1",
        bet_amount=800.0,
    )
    assert score >= 50  # Well above rolling average


def test_composite_score_calculation(engine):
    """Composite score should weight components correctly."""
    score = engine.calculate_composite_score(
        smart_money=80.0,
        volume_spike=60.0,
        cluster=0.0,
    )
    # (80*0.4) + (60*0.3) + (0*0.3) = 32 + 18 + 0 = 50
    assert score == 50.0


def test_signal_threshold(engine):
    """Signals below threshold should not fire."""
    score = engine.calculate_composite_score(
        smart_money=30.0,
        volume_spike=20.0,
        cluster=0.0,
    )
    assert score < 60  # Below default threshold
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_signal_engine.py -v`
Expected: FAIL — module doesn't exist

**Step 3: Implement signal_engine.py**

Create `polymarket_tracker/signal_engine.py`:

```python
"""
Signal Engine for Polymarket Tracker.

Evaluates incoming bets against three signal types:
1. Smart Money — graded whale taking a position
2. Volume Spike — bet size well above market's rolling average
3. Coordinated Cluster — multiple wallets same outcome in short window

Produces a composite conviction score (0-100) per market outcome.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from .database import Database, db

logger = logging.getLogger(__name__)


@dataclass
class SignalConfig:
    """Configuration for signal thresholds."""
    # Composite weights (must sum to 1.0)
    WEIGHT_SMART_MONEY: float = 0.4
    WEIGHT_VOLUME_SPIKE: float = 0.3
    WEIGHT_CLUSTER: float = 0.3

    # Signal fire threshold
    CONVICTION_THRESHOLD: float = 60.0

    # Smart money config
    GRADE_SCORES: dict = None  # Set in __post_init__

    # Volume spike config
    SPIKE_MULTIPLIER: float = 3.0       # Bet must be Nx above rolling avg
    ROLLING_WINDOW_DAYS: int = 7

    # Cluster config
    CLUSTER_WINDOW_MINUTES: int = 30
    CLUSTER_MIN_WALLETS: int = 3

    def __post_init__(self):
        if self.GRADE_SCORES is None:
            self.GRADE_SCORES = {"A": 90, "B": 70, "C": 40, "D": 20, "ungraded": 0}


class SignalEngine:
    """Evaluates bets and generates trading signals."""

    def __init__(
        self,
        database: Optional[Database] = None,
        config: Optional[SignalConfig] = None,
    ):
        self.db = database or db
        self.config = config or SignalConfig()

    def score_smart_money(
        self,
        wallet_address: str,
        bet_amount: float,
        market_id: str,
    ) -> float:
        """Score based on whale grade and bet characteristics. Returns 0-100."""
        profile = self.db.get_whale_profile(wallet_address)
        if not profile or profile["grade"] == "ungraded":
            return 0.0

        grade = profile["grade"]
        base_score = self.config.GRADE_SCORES.get(grade, 0)

        # Boost for category specialization
        market = self.db.get_market(market_id)
        if market and profile.get("category_specialization"):
            try:
                cat_spec = json.loads(profile["category_specialization"])
                market_cat = market.get("category", "")
                if market_cat and market_cat in cat_spec:
                    cat_wr = cat_spec[market_cat]
                    if cat_wr >= 65:
                        base_score = min(100, base_score + 10)
            except (json.JSONDecodeError, TypeError):
                pass

        return float(base_score)

    def score_volume_spike(
        self,
        market_id: str,
        bet_amount: float,
    ) -> float:
        """Score based on how much this bet exceeds the market's rolling average. Returns 0-100."""
        with self.db.get_connection() as conn:
            cutoff = datetime.utcnow() - timedelta(days=self.config.ROLLING_WINDOW_DAYS)
            cursor = conn.execute(
                """
                SELECT AVG(amount) as avg_amount, COUNT(*) as bet_count
                FROM bets
                WHERE market_id = ? AND timestamp >= ?
                """,
                (market_id, cutoff.isoformat()),
            )
            row = cursor.fetchone()

        if not row or not row["avg_amount"] or row["bet_count"] < 5:
            return 0.0

        avg = float(row["avg_amount"])
        if avg <= 0:
            return 0.0

        ratio = bet_amount / avg
        if ratio < self.config.SPIKE_MULTIPLIER:
            return 0.0

        # Scale: 3x = 50, 5x = 75, 10x+ = 100
        score = min(100, 50 + (ratio - self.config.SPIKE_MULTIPLIER) * 10)
        return round(score, 1)

    def score_cluster(
        self,
        market_id: str,
        outcome: str,
        timestamp: datetime,
    ) -> float:
        """Score based on correlated betting — multiple wallets same outcome in short window. Returns 0-100."""
        window_start = timestamp - timedelta(minutes=self.config.CLUSTER_WINDOW_MINUTES)
        window_end = timestamp + timedelta(minutes=self.config.CLUSTER_WINDOW_MINUTES)

        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT COUNT(DISTINCT wallet_address) as wallet_count
                FROM bets
                WHERE market_id = ?
                  AND outcome_bet = ?
                  AND timestamp BETWEEN ? AND ?
                  AND side = 'BUY'
                """,
                (market_id, outcome, window_start.isoformat(), window_end.isoformat()),
            )
            row = cursor.fetchone()

        if not row:
            return 0.0

        wallet_count = row["wallet_count"]
        if wallet_count < self.config.CLUSTER_MIN_WALLETS:
            return 0.0

        # Scale: 3 wallets = 50, 5 = 75, 8+ = 100
        score = min(100, 50 + (wallet_count - self.config.CLUSTER_MIN_WALLETS) * 12.5)
        return round(score, 1)

    def calculate_composite_score(
        self,
        smart_money: float,
        volume_spike: float,
        cluster: float,
    ) -> float:
        """Calculate weighted composite conviction score."""
        score = (
            smart_money * self.config.WEIGHT_SMART_MONEY
            + volume_spike * self.config.WEIGHT_VOLUME_SPIKE
            + cluster * self.config.WEIGHT_CLUSTER
        )
        return round(score, 1)

    def evaluate_bet(
        self,
        wallet_address: str,
        market_id: str,
        outcome: str,
        amount: float,
        timestamp: datetime,
    ) -> Optional[dict]:
        """
        Evaluate a single bet across all signal types.
        Returns signal dict if conviction >= threshold, else None.
        """
        sm_score = self.score_smart_money(wallet_address, amount, market_id)
        vs_score = self.score_volume_spike(market_id, amount)
        cl_score = self.score_cluster(market_id, outcome, timestamp)

        conviction = self.calculate_composite_score(sm_score, vs_score, cl_score)

        if conviction < self.config.CONVICTION_THRESHOLD:
            return None

        # Build signal details
        details = {
            "wallet_address": wallet_address,
            "bet_amount": amount,
            "smart_money": {"score": sm_score},
            "volume_spike": {"score": vs_score},
            "cluster": {"score": cl_score},
        }

        # Add whale profile info if available
        profile = self.db.get_whale_profile(wallet_address)
        if profile:
            details["smart_money"]["grade"] = profile["grade"]
            details["smart_money"]["win_rate"] = profile["win_rate"]

        signal_id = self.db.insert_signal(
            market_id=market_id,
            outcome=outcome,
            conviction_score=conviction,
            smart_money_score=sm_score,
            volume_spike_score=vs_score,
            cluster_score=cl_score,
            contributing_wallets=json.dumps([wallet_address]),
            details=json.dumps(details),
        )

        logger.info(
            f"Signal fired: market={market_id} outcome={outcome} "
            f"conviction={conviction} (SM={sm_score}, VS={vs_score}, CL={cl_score})"
        )

        return {
            "signal_id": signal_id,
            "market_id": market_id,
            "outcome": outcome,
            "conviction_score": conviction,
            "smart_money_score": sm_score,
            "volume_spike_score": vs_score,
            "cluster_score": cl_score,
            "details": details,
        }

    def evaluate_recent_bets(self, since_minutes: int = 5) -> list[dict]:
        """Evaluate all bets from the last N minutes. Returns list of fired signals."""
        cutoff = datetime.utcnow() - timedelta(minutes=since_minutes)
        signals = []

        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM bets
                WHERE timestamp >= ? AND side = 'BUY'
                ORDER BY timestamp DESC
                """,
                (cutoff.isoformat(),),
            )
            recent_bets = [dict(row) for row in cursor.fetchall()]

        for bet in recent_bets:
            ts = bet["timestamp"]
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)

            result = self.evaluate_bet(
                wallet_address=bet["wallet_address"],
                market_id=bet["market_id"],
                outcome=bet["outcome_bet"],
                amount=float(bet["amount"]),
                timestamp=ts,
            )
            if result:
                signals.append(result)

        return signals
```

**Step 4: Run tests**

Run: `pytest tests/test_signal_engine.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add polymarket_tracker/signal_engine.py tests/test_signal_engine.py
git commit -m "feat: add signal engine with smart money, volume spike, and cluster scoring"
```

---

### Task 5: Integrate Signal Engine into Collection Cycle

**Files:**
- Modify: `polymarket_tracker/collector.py:312-367`

**Step 1: Write a failing test**

Add to a new file `tests/test_collector_signals.py`:

```python
"""Test that the collector triggers signal evaluation after trade sync."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from polymarket_tracker.collector import DataCollector


@pytest.mark.asyncio
async def test_collect_data_runs_signal_evaluation(tmp_db):
    """collect_data should call signal engine after syncing trades."""
    collector = DataCollector(database=tmp_db)
    collector.client = AsyncMock()
    collector.client.get_all_active_markets = AsyncMock(return_value=[])
    collector.client.health_check = AsyncMock(return_value=True)
    collector.client.close = AsyncMock()

    with patch("polymarket_tracker.collector.SignalEngine") as MockEngine:
        mock_engine = MagicMock()
        mock_engine.evaluate_recent_bets.return_value = []
        MockEngine.return_value = mock_engine

        await collector.collect_data()

        mock_engine.evaluate_recent_bets.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_collector_signals.py -v`
Expected: FAIL — `SignalEngine` not imported in collector

**Step 3: Add signal evaluation to collect_data**

In `polymarket_tracker/collector.py`, add the import at the top (after existing imports):

```python
from .signal_engine import SignalEngine
```

Then in the `collect_data` method, after `stats["trades_synced"] = await self.sync_all_trades()` (around line 350), add:

```python
            # Step 3: Evaluate recent bets for signals
            try:
                signal_engine = SignalEngine(database=self.db)
                signals = signal_engine.evaluate_recent_bets(
                    since_minutes=settings.fetch_interval_minutes
                )
                stats["signals_fired"] = len(signals)
                if signals:
                    logger.info(f"Signal engine: {len(signals)} signals fired")
            except Exception as e:
                logger.error(f"Signal evaluation error: {e}")
                stats["signals_fired"] = 0
```

**Step 4: Run test**

Run: `pytest tests/test_collector_signals.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add polymarket_tracker/collector.py tests/test_collector_signals.py
git commit -m "feat: integrate signal engine into collection cycle"
```

---

### Task 6: Integrate Whale Profiler into Collection Cycle

**Files:**
- Modify: `polymarket_tracker/collector.py`

**Step 1: Add whale profile refresh to collect_data**

In `polymarket_tracker/collector.py`, add import:

```python
from .whale_profiler import WhaleProfiler
```

In `collect_data`, right after the market sync block (after `stats["markets_synced"] = await self.sync_markets()`), add:

```python
                # Refresh whale profiles alongside market sync (every 30 min)
                try:
                    profiler = WhaleProfiler(database=self.db)
                    stats["whales_profiled"] = profiler.refresh_all_profiles()
                except Exception as e:
                    logger.error(f"Whale profiling error: {e}")
                    stats["whales_profiled"] = 0
```

**Step 2: Run existing tests to ensure nothing broke**

Run: `pytest tests/ -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add polymarket_tracker/collector.py
git commit -m "feat: integrate whale profiler refresh into collection cycle"
```

---

### Task 7: Web Dashboard — FastAPI App Structure

**Files:**
- Create: `polymarket_tracker/web/__init__.py`
- Create: `polymarket_tracker/web/app.py`
- Create: `polymarket_tracker/web/templates/base.html`
- Create: `polymarket_tracker/web/templates/index.html`
- Create: `polymarket_tracker/web/static/style.css`

**Step 1: Create the web package directory**

Run:
```bash
mkdir -p polymarket_tracker/web/templates polymarket_tracker/web/static
```

**Step 2: Create `polymarket_tracker/web/__init__.py`**

```python
"""Web dashboard for Polymarket Tracker."""
```

**Step 3: Create `polymarket_tracker/web/app.py`**

```python
"""
FastAPI web dashboard for Polymarket Tracker.

Serves a live signal feed, hot markets, and whale directory.
Run: python -m polymarket_tracker.web
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..database import Database, db

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Polymarket Whale Tracker")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ========== WebSocket Manager ==========

class ConnectionManager:
    """Manages WebSocket connections for live signal push."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


manager = ConnectionManager()


# ========== Template Filters ==========

def format_usd(value):
    """Format a number as USD."""
    if value is None:
        return "$0"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.1f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:,.1f}k"
    return f"${value:,.0f}"


def format_address(addr):
    """Truncate wallet address for display."""
    if not addr or len(addr) < 12:
        return addr or ""
    return f"{addr[:6]}...{addr[-4:]}"


def format_time_ago(dt_value):
    """Format a datetime as 'X ago'."""
    if not dt_value:
        return "—"
    if isinstance(dt_value, str):
        try:
            dt_value = datetime.fromisoformat(dt_value)
        except ValueError:
            return dt_value
    delta = datetime.utcnow() - dt_value
    if delta.days > 0:
        return f"{delta.days}d ago"
    hours = delta.seconds // 3600
    if hours > 0:
        return f"{hours}h ago"
    minutes = delta.seconds // 60
    return f"{minutes}m ago"


templates.env.filters["usd"] = format_usd
templates.env.filters["address"] = format_address
templates.env.filters["time_ago"] = format_time_ago


# ========== Routes ==========

@app.get("/", response_class=HTMLResponse)
async def live_feed(request: Request, min_score: float = Query(0, alias="min_score")):
    """Live signal feed page."""
    signals = db.get_recent_signals(limit=50, min_score=min_score)
    stats = db.get_stats()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "signals": signals,
        "stats": stats,
        "min_score": min_score,
        "page": "feed",
    })


@app.get("/markets", response_class=HTMLResponse)
async def hot_markets(request: Request, hours: int = Query(24)):
    """Hot markets page — markets with most signal activity."""
    markets = db.get_hot_markets(hours=hours)
    return templates.TemplateResponse("markets.html", {
        "request": request,
        "markets": markets,
        "hours": hours,
        "page": "markets",
    })


@app.get("/whales", response_class=HTMLResponse)
async def whale_directory(request: Request, grade: Optional[str] = None):
    """Whale directory page."""
    whales = db.get_whale_profiles(min_grade=grade)
    return templates.TemplateResponse("whales.html", {
        "request": request,
        "whales": whales,
        "grade_filter": grade,
        "page": "whales",
    })


@app.get("/whales/{address}", response_class=HTMLResponse)
async def whale_profile(request: Request, address: str):
    """Individual whale profile page."""
    profile = db.get_whale_profile(address)
    bets = db.get_bets_for_wallet(address, limit=50)
    return templates.TemplateResponse("whale_detail.html", {
        "request": request,
        "profile": profile,
        "bets": bets,
        "address": address,
        "page": "whales",
    })


@app.get("/markets/{market_id}", response_class=HTMLResponse)
async def market_detail(request: Request, market_id: str):
    """Market detail page with signal history."""
    market = db.get_market(market_id)
    bets = db.get_bets_for_market(market_id, limit=100)
    signals = db.get_signals_for_market(market_id)
    return templates.TemplateResponse("market_detail.html", {
        "request": request,
        "market": market,
        "bets": bets,
        "signals": signals,
        "page": "markets",
    })


# ========== API Endpoints (for HTMX) ==========

@app.get("/api/signals")
async def api_signals(min_score: float = 0, limit: int = 20):
    """JSON API for signals (used by HTMX polling)."""
    return db.get_recent_signals(limit=limit, min_score=min_score)


# ========== WebSocket ==========

@app.websocket("/ws/signals")
async def websocket_signals(websocket: WebSocket):
    """WebSocket endpoint for live signal updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, signals pushed via broadcast
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

**Step 4: Create `polymarket_tracker/web/__main__.py`**

```python
"""Entry point: python -m polymarket_tracker.web"""

import uvicorn


def main():
    uvicorn.run(
        "polymarket_tracker.web.app:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
    )


if __name__ == "__main__":
    main()
```

**Step 5: Commit**

```bash
git add polymarket_tracker/web/
git commit -m "feat: add FastAPI web dashboard app structure with routes"
```

---

### Task 8: Web Dashboard — HTML Templates

**Files:**
- Create: `polymarket_tracker/web/templates/base.html`
- Create: `polymarket_tracker/web/templates/index.html`
- Create: `polymarket_tracker/web/templates/markets.html`
- Create: `polymarket_tracker/web/templates/whales.html`
- Create: `polymarket_tracker/web/templates/whale_detail.html`
- Create: `polymarket_tracker/web/templates/market_detail.html`
- Create: `polymarket_tracker/web/static/style.css`

**Step 1: Create base.html**

This is the layout shell. Uses HTMX for interactivity, no build step needed.

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Polymarket Whale Tracker{% endblock %}</title>
    <link rel="stylesheet" href="/static/style.css">
    <script src="https://unpkg.com/htmx.org@2.0.4"></script>
</head>
<body>
    <nav class="navbar">
        <div class="nav-brand">Whale Tracker</div>
        <div class="nav-links">
            <a href="/" class="{% if page == 'feed' %}active{% endif %}">Live Feed</a>
            <a href="/markets" class="{% if page == 'markets' %}active{% endif %}">Hot Markets</a>
            <a href="/whales" class="{% if page == 'whales' %}active{% endif %}">Whales</a>
        </div>
    </nav>
    <main class="container">
        {% block content %}{% endblock %}
    </main>
</body>
</html>
```

**Step 2: Create index.html (Live Feed)**

```html
{% extends "base.html" %}
{% block title %}Live Signal Feed — Whale Tracker{% endblock %}
{% block content %}
<div class="page-header">
    <h1>Live Signal Feed</h1>
    <div class="stats-bar">
        <span>Traders: {{ stats.total_traders | default(0) }}</span>
        <span>Markets: {{ stats.total_markets | default(0) }}</span>
        <span>Bets: {{ stats.total_bets | default(0) }}</span>
    </div>
</div>

<div class="filter-bar">
    <form method="get" action="/">
        <label>Min Score:
            <input type="number" name="min_score" value="{{ min_score }}" min="0" max="100" step="5">
        </label>
        <button type="submit">Filter</button>
    </form>
</div>

<div id="signal-feed" hx-get="/api/signals?min_score={{ min_score }}&limit=20" hx-trigger="every 30s" hx-swap="innerHTML">
    {% if signals %}
    <div class="signal-list">
        {% for s in signals %}
        <div class="signal-card score-{{ 'high' if s.conviction_score >= 80 else ('med' if s.conviction_score >= 60 else 'low') }}">
            <div class="signal-header">
                <span class="conviction-badge">{{ s.conviction_score | round(0) }}</span>
                <span class="signal-time">{{ s.timestamp | time_ago }}</span>
            </div>
            <div class="signal-body">
                <a href="/markets/{{ s.market_id }}" class="market-link">
                    {{ s.market_question | default(s.market_id) }}
                </a>
                <div class="signal-outcome">Outcome: <strong>{{ s.outcome }}</strong></div>
                <div class="signal-scores">
                    <span title="Smart Money">SM: {{ s.smart_money_score | round(0) }}</span>
                    <span title="Volume Spike">VS: {{ s.volume_spike_score | round(0) }}</span>
                    <span title="Cluster">CL: {{ s.cluster_score | round(0) }}</span>
                </div>
            </div>
        </div>
        {% endfor %}
    </div>
    {% else %}
    <div class="empty-state">No signals yet. Signals appear when unusual betting activity is detected.</div>
    {% endif %}
</div>
{% endblock %}
```

**Step 3: Create markets.html, whales.html, whale_detail.html, market_detail.html**

These follow the same pattern — table/card layouts with data from the route context. Each template extends `base.html` and renders the relevant data. Keep them simple — data tables with links between pages.

*(Each template follows the same pattern as index.html — extend base, render data in tables/cards. Implementation should follow the exact same structure.)*

**Step 4: Create static/style.css**

Minimal dark-theme CSS:

```css
:root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --border: #2a2d37;
    --text: #e4e4e7;
    --text-muted: #71717a;
    --accent: #3b82f6;
    --green: #22c55e;
    --yellow: #eab308;
    --red: #ef4444;
}

* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }

.navbar { display: flex; align-items: center; padding: 1rem 2rem; background: var(--surface); border-bottom: 1px solid var(--border); }
.nav-brand { font-size: 1.25rem; font-weight: 700; margin-right: 2rem; }
.nav-links a { color: var(--text-muted); text-decoration: none; margin-right: 1.5rem; padding: 0.5rem 0; }
.nav-links a.active, .nav-links a:hover { color: var(--text); border-bottom: 2px solid var(--accent); }

.container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
.page-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem; }
.page-header h1 { font-size: 1.5rem; }
.stats-bar span { color: var(--text-muted); margin-left: 1.5rem; font-size: 0.875rem; }

.filter-bar { margin-bottom: 1.5rem; padding: 1rem; background: var(--surface); border-radius: 8px; }
.filter-bar input { background: var(--bg); border: 1px solid var(--border); color: var(--text); padding: 0.5rem; border-radius: 4px; width: 80px; }
.filter-bar button { background: var(--accent); color: white; border: none; padding: 0.5rem 1rem; border-radius: 4px; cursor: pointer; margin-left: 0.5rem; }

.signal-list { display: flex; flex-direction: column; gap: 1rem; }
.signal-card { background: var(--surface); border-radius: 8px; padding: 1rem; border-left: 4px solid var(--border); }
.signal-card.score-high { border-left-color: var(--red); }
.signal-card.score-med { border-left-color: var(--yellow); }
.signal-card.score-low { border-left-color: var(--green); }

.signal-header { display: flex; justify-content: space-between; margin-bottom: 0.5rem; }
.conviction-badge { background: var(--bg); padding: 0.25rem 0.75rem; border-radius: 12px; font-weight: 700; font-size: 0.875rem; }
.signal-time { color: var(--text-muted); font-size: 0.875rem; }
.market-link { color: var(--accent); text-decoration: none; font-weight: 500; }
.market-link:hover { text-decoration: underline; }
.signal-outcome { margin: 0.5rem 0; }
.signal-scores { display: flex; gap: 1rem; color: var(--text-muted); font-size: 0.875rem; }

.empty-state { text-align: center; color: var(--text-muted); padding: 4rem; }

/* Tables */
table { width: 100%; border-collapse: collapse; background: var(--surface); border-radius: 8px; overflow: hidden; }
th, td { padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid var(--border); }
th { color: var(--text-muted); font-weight: 500; font-size: 0.875rem; text-transform: uppercase; }
tr:hover { background: rgba(59, 130, 246, 0.05); }

/* Grades */
.grade { font-weight: 700; padding: 0.25rem 0.5rem; border-radius: 4px; }
.grade-A { color: var(--green); }
.grade-B { color: var(--accent); }
.grade-C { color: var(--yellow); }
.grade-D { color: var(--red); }
```

**Step 5: Commit**

```bash
git add polymarket_tracker/web/
git commit -m "feat: add HTML templates and CSS for web dashboard"
```

---

### Task 9: Remaining Template Pages

**Files:**
- Create: `polymarket_tracker/web/templates/markets.html`
- Create: `polymarket_tracker/web/templates/whales.html`
- Create: `polymarket_tracker/web/templates/whale_detail.html`
- Create: `polymarket_tracker/web/templates/market_detail.html`

**Step 1: Create markets.html**

```html
{% extends "base.html" %}
{% block title %}Hot Markets — Whale Tracker{% endblock %}
{% block content %}
<div class="page-header">
    <h1>Hot Markets (Last {{ hours }}h)</h1>
</div>
<table>
    <thead>
        <tr>
            <th>Market</th>
            <th>Category</th>
            <th>Signals</th>
            <th>Max Conviction</th>
            <th>Volume</th>
            <th>Latest Signal</th>
        </tr>
    </thead>
    <tbody>
        {% for m in markets %}
        <tr>
            <td><a href="/markets/{{ m.market_id }}" class="market-link">{{ m.question | default(m.market_id) }}</a></td>
            <td>{{ m.category | default('—') }}</td>
            <td>{{ m.signal_count }}</td>
            <td><span class="conviction-badge">{{ m.max_conviction | round(0) }}</span></td>
            <td>{{ m.volume | usd }}</td>
            <td>{{ m.latest_signal | time_ago }}</td>
        </tr>
        {% endfor %}
    </tbody>
</table>
{% if not markets %}
<div class="empty-state">No hot markets in the last {{ hours }} hours.</div>
{% endif %}
{% endblock %}
```

**Step 2: Create whales.html**

```html
{% extends "base.html" %}
{% block title %}Whale Directory — Whale Tracker{% endblock %}
{% block content %}
<div class="page-header">
    <h1>Whale Directory</h1>
    <div class="filter-bar" style="margin:0; padding:0.5rem; background:none;">
        <a href="/whales" class="{% if not grade_filter %}active{% endif %}">All</a>
        <a href="/whales?grade=A" class="{% if grade_filter == 'A' %}active{% endif %}">A</a>
        <a href="/whales?grade=B" class="{% if grade_filter == 'B' %}active{% endif %}">B+</a>
    </div>
</div>
<table>
    <thead>
        <tr>
            <th>Wallet</th>
            <th>Grade</th>
            <th>Win Rate</th>
            <th>Volume</th>
            <th>Resolved Bets</th>
            <th>Pattern</th>
        </tr>
    </thead>
    <tbody>
        {% for w in whales %}
        <tr>
            <td><a href="/whales/{{ w.wallet_address }}" class="market-link">{{ w.wallet_address | address }}</a></td>
            <td><span class="grade grade-{{ w.grade }}">{{ w.grade }}</span></td>
            <td>{{ w.win_rate | round(1) }}%</td>
            <td>{{ w.total_volume | usd }}</td>
            <td>{{ w.total_resolved_bets }}</td>
            <td>{{ w.activity_pattern | default('—') }}</td>
        </tr>
        {% endfor %}
    </tbody>
</table>
{% if not whales %}
<div class="empty-state">No whale profiles yet. Run the collector to identify whales.</div>
{% endif %}
{% endblock %}
```

**Step 3: Create whale_detail.html**

```html
{% extends "base.html" %}
{% block title %}{{ address | address }} — Whale Tracker{% endblock %}
{% block content %}
<h1>Whale Profile: {{ address | address }}</h1>
{% if profile %}
<div class="profile-grid" style="display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin:1.5rem 0;">
    <div class="stat-card" style="background:var(--surface); padding:1rem; border-radius:8px;">
        <div style="color:var(--text-muted); font-size:0.875rem;">Grade</div>
        <div class="grade grade-{{ profile.grade }}" style="font-size:2rem;">{{ profile.grade }}</div>
    </div>
    <div class="stat-card" style="background:var(--surface); padding:1rem; border-radius:8px;">
        <div style="color:var(--text-muted); font-size:0.875rem;">Win Rate</div>
        <div style="font-size:1.5rem;">{{ profile.win_rate | round(1) }}%</div>
    </div>
    <div class="stat-card" style="background:var(--surface); padding:1rem; border-radius:8px;">
        <div style="color:var(--text-muted); font-size:0.875rem;">Volume</div>
        <div style="font-size:1.5rem;">{{ profile.total_volume | usd }}</div>
    </div>
    <div class="stat-card" style="background:var(--surface); padding:1rem; border-radius:8px;">
        <div style="color:var(--text-muted); font-size:0.875rem;">Resolved Bets</div>
        <div style="font-size:1.5rem;">{{ profile.total_resolved_bets }}</div>
    </div>
</div>
{% else %}
<div class="empty-state">No whale profile found for this address.</div>
{% endif %}

<h2 style="margin:2rem 0 1rem;">Recent Bets</h2>
<table>
    <thead><tr><th>Time</th><th>Market</th><th>Side</th><th>Amount</th><th>Price</th><th>Outcome</th></tr></thead>
    <tbody>
        {% for b in bets %}
        <tr>
            <td>{{ b.timestamp | time_ago }}</td>
            <td><a href="/markets/{{ b.market_id }}" class="market-link">{{ b.market_id | address }}</a></td>
            <td>{{ b.side }}</td>
            <td>{{ b.amount | usd }}</td>
            <td>{{ "%.2f" | format(b.price) }}</td>
            <td>{{ b.outcome_bet }}</td>
        </tr>
        {% endfor %}
    </tbody>
</table>
{% endblock %}
```

**Step 4: Create market_detail.html**

```html
{% extends "base.html" %}
{% block title %}{{ market.question | default('Market') }} — Whale Tracker{% endblock %}
{% block content %}
{% if market %}
<h1>{{ market.question }}</h1>
<div style="color:var(--text-muted); margin-bottom:1.5rem;">
    Category: {{ market.category | default('—') }} |
    Volume: {{ market.volume | usd }} |
    End: {{ market.end_date | default('—') }}
</div>

{% if signals %}
<h2 style="margin-bottom:1rem;">Signals</h2>
<div class="signal-list" style="margin-bottom:2rem;">
    {% for s in signals %}
    <div class="signal-card score-{{ 'high' if s.conviction_score >= 80 else ('med' if s.conviction_score >= 60 else 'low') }}">
        <div class="signal-header">
            <span class="conviction-badge">{{ s.conviction_score | round(0) }}</span>
            <span>{{ s.outcome }}</span>
            <span class="signal-time">{{ s.timestamp | time_ago }}</span>
        </div>
        <div class="signal-scores">
            <span>SM: {{ s.smart_money_score | round(0) }}</span>
            <span>VS: {{ s.volume_spike_score | round(0) }}</span>
            <span>CL: {{ s.cluster_score | round(0) }}</span>
        </div>
    </div>
    {% endfor %}
</div>
{% endif %}

<h2 style="margin-bottom:1rem;">Recent Bets</h2>
<table>
    <thead><tr><th>Time</th><th>Wallet</th><th>Side</th><th>Amount</th><th>Price</th><th>Outcome</th></tr></thead>
    <tbody>
        {% for b in bets %}
        <tr>
            <td>{{ b.timestamp | time_ago }}</td>
            <td><a href="/whales/{{ b.wallet_address }}" class="market-link">{{ b.wallet_address | address }}</a></td>
            <td>{{ b.side }}</td>
            <td>{{ b.amount | usd }}</td>
            <td>{{ "%.2f" | format(b.price) }}</td>
            <td>{{ b.outcome_bet }}</td>
        </tr>
        {% endfor %}
    </tbody>
</table>
{% else %}
<div class="empty-state">Market not found.</div>
{% endif %}
{% endblock %}
```

**Step 5: Commit**

```bash
git add polymarket_tracker/web/templates/
git commit -m "feat: add remaining template pages for markets, whales, and details"
```

---

### Task 10: Smoke Test & Manual Verification

**Step 1: Run the full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

**Step 2: Start the web dashboard**

Run: `python -m polymarket_tracker.web`
Expected: Server starts on http://localhost:8080

**Step 3: Verify pages load**

- Visit `http://localhost:8080/` — should show empty signal feed
- Visit `http://localhost:8080/markets` — should show empty hot markets
- Visit `http://localhost:8080/whales` — should show empty whale directory
- All pages should render with the dark theme, no errors

**Step 4: Run a collection cycle to populate data**

Run: `python -m polymarket_tracker.main --once`
Expected: Markets synced, trades collected, whale profiles generated, signals evaluated

**Step 5: Verify dashboard shows data**

Refresh `http://localhost:8080/whales` — should now show identified whales with grades

**Step 6: Commit**

```bash
git add -A
git commit -m "feat: complete whale signal dashboard v1"
```
