"""
Tests for the database module.
"""

import json
import os
import tempfile
from datetime import datetime, timedelta

import pytest

from polymarket_tracker.database import Database


@pytest.fixture
def test_db():
    """Create a temporary test database."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = Database(db_path=path)
    yield db
    os.unlink(path)


class TestDatabase:
    """Tests for Database class."""

    def test_init_creates_tables(self, test_db):
        """Test that initialization creates all required tables."""
        stats = test_db.get_stats()
        assert stats["total_traders"] == 0
        assert stats["total_markets"] == 0
        assert stats["total_bets"] == 0

    def test_upsert_trader(self, test_db):
        """Test inserting and updating traders."""
        test_db.upsert_trader(
            wallet_address="0x1234567890abcdef",
            volume=100.0,
            timestamp=datetime.utcnow()
        )

        trader = test_db.get_trader("0x1234567890abcdef")
        assert trader is not None
        assert trader["total_volume"] == 100.0
        assert trader["total_trades"] == 1

        # Update with more volume
        test_db.upsert_trader(
            wallet_address="0x1234567890abcdef",
            volume=50.0,
            timestamp=datetime.utcnow()
        )

        trader = test_db.get_trader("0x1234567890abcdef")
        assert trader["total_volume"] == 150.0
        assert trader["total_trades"] == 2

    def test_upsert_market(self, test_db):
        """Test inserting and updating markets."""
        test_db.upsert_market(
            market_id="test_market_1",
            question="Will it rain tomorrow?",
            description="Test description",
            active=True,
            volume=1000.0
        )

        market = test_db.get_market("test_market_1")
        assert market is not None
        assert market["question"] == "Will it rain tomorrow?"
        assert market["volume"] == 1000.0

        # Update market
        test_db.upsert_market(
            market_id="test_market_1",
            question="Will it rain tomorrow?",
            volume=2000.0
        )

        market = test_db.get_market("test_market_1")
        assert market["volume"] == 2000.0

    def test_insert_bet(self, test_db):
        """Test inserting bets."""
        # First create a trader and market
        test_db.upsert_trader(
            wallet_address="0xtrader1",
            volume=0,
            timestamp=datetime.utcnow()
        )
        test_db.upsert_market(
            market_id="market_1",
            question="Test market?"
        )

        # Insert a bet
        result = test_db.insert_bet(
            bet_id="bet_1",
            wallet_address="0xtrader1",
            market_id="market_1",
            asset_id="asset_1",
            amount=100.0,
            price=0.5,
            side="BUY",
            outcome_bet="Yes",
            timestamp=datetime.utcnow()
        )
        assert result is True

        # Try to insert duplicate
        result = test_db.insert_bet(
            bet_id="bet_1",
            wallet_address="0xtrader1",
            market_id="market_1",
            asset_id="asset_1",
            amount=100.0,
            price=0.5,
            side="BUY",
            outcome_bet="Yes",
            timestamp=datetime.utcnow()
        )
        assert result is False

    def test_get_active_markets(self, test_db):
        """Test getting active markets."""
        # Insert active and inactive markets
        test_db.upsert_market(
            market_id="active_1",
            question="Active market?",
            active=True,
            closed=False
        )
        test_db.upsert_market(
            market_id="closed_1",
            question="Closed market?",
            active=True,
            closed=True
        )
        test_db.upsert_market(
            market_id="inactive_1",
            question="Inactive market?",
            active=False,
            closed=False
        )

        active_markets = test_db.get_active_markets()
        assert len(active_markets) == 1
        assert active_markets[0]["market_id"] == "active_1"

    def test_sync_state(self, test_db):
        """Test sync state operations."""
        # Initially empty
        assert test_db.get_sync_state("test_key") is None

        # Set state
        test_db.set_sync_state("test_key", "test_value")
        assert test_db.get_sync_state("test_key") == "test_value"

        # Update state
        test_db.set_sync_state("test_key", "new_value")
        assert test_db.get_sync_state("test_key") == "new_value"

    def test_get_top_traders(self, test_db):
        """Test getting top traders by volume."""
        # Insert traders with different volumes
        for i in range(5):
            test_db.upsert_trader(
                wallet_address=f"0xtrader{i}",
                volume=(i + 1) * 100.0,
                timestamp=datetime.utcnow()
            )

        top_traders = test_db.get_top_traders(limit=3)
        assert len(top_traders) == 3
        assert top_traders[0]["total_volume"] == 500.0  # Highest volume first
        assert top_traders[1]["total_volume"] == 400.0
        assert top_traders[2]["total_volume"] == 300.0

    # ========== Whale Profile Tests ==========

    def test_whale_profiles_table_exists(self, test_db):
        """Test that whale_profiles table exists after init."""
        with test_db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='whale_profiles'"
            )
            row = cursor.fetchone()
            assert row is not None
            assert row["name"] == "whale_profiles"

    def test_upsert_and_get_whale_profile(self, test_db):
        """Test inserting and retrieving a whale profile."""
        # Create the referenced trader first
        test_db.upsert_trader(
            wallet_address="0xwhale1",
            volume=50000.0,
            timestamp=datetime.utcnow()
        )

        test_db.upsert_whale_profile(
            wallet_address="0xwhale1",
            grade="A",
            win_rate=0.72,
            roi=1.35,
            sharpe_ratio=2.1,
            category_specialization=json.dumps({"politics": 65.0, "crypto": 35.0}),
            avg_bet_timing=48.5,
            activity_pattern="steady",
            total_volume=50000.0,
            total_resolved_bets=120
        )

        profile = test_db.get_whale_profile("0xwhale1")
        assert profile is not None
        assert profile["wallet_address"] == "0xwhale1"
        assert profile["grade"] == "A"
        assert profile["win_rate"] == 0.72
        assert profile["roi"] == 1.35
        assert profile["sharpe_ratio"] == 2.1
        assert profile["activity_pattern"] == "steady"
        assert profile["total_volume"] == 50000.0
        assert profile["total_resolved_bets"] == 120

        # Verify category_specialization is valid JSON
        cat_spec = json.loads(profile["category_specialization"])
        assert cat_spec["politics"] == 65.0

        # Update the profile
        test_db.upsert_whale_profile(
            wallet_address="0xwhale1",
            grade="B",
            win_rate=0.65,
            roi=0.9,
            total_volume=55000.0,
            total_resolved_bets=130
        )

        updated = test_db.get_whale_profile("0xwhale1")
        assert updated["grade"] == "B"
        assert updated["win_rate"] == 0.65
        assert updated["total_volume"] == 55000.0

    def test_get_whale_profile_not_found(self, test_db):
        """Test that get_whale_profile returns None for unknown address."""
        assert test_db.get_whale_profile("0xnonexistent") is None

    def test_get_whale_profiles_with_grade_filter(self, test_db):
        """Test getting whale profiles filtered by minimum grade."""
        # Create traders and whale profiles with different grades
        for addr, grade, volume in [
            ("0xwhaleA", "A", 100000.0),
            ("0xwhaleB", "B", 80000.0),
            ("0xwhaleC", "C", 60000.0),
            ("0xwhaleD", "D", 40000.0),
            ("0xwhaleU", "ungraded", 20000.0),
        ]:
            test_db.upsert_trader(
                wallet_address=addr, volume=volume, timestamp=datetime.utcnow()
            )
            test_db.upsert_whale_profile(
                wallet_address=addr, grade=grade, total_volume=volume
            )

        # min_grade="B" should return A and B only
        profiles = test_db.get_whale_profiles(min_grade="B")
        assert len(profiles) == 2
        grades = {p["grade"] for p in profiles}
        assert grades == {"A", "B"}

        # min_grade="C" should return A, B, and C
        profiles = test_db.get_whale_profiles(min_grade="C")
        assert len(profiles) == 3
        grades = {p["grade"] for p in profiles}
        assert grades == {"A", "B", "C"}

        # No filter returns all
        profiles = test_db.get_whale_profiles()
        assert len(profiles) == 5

        # Verify ordering by total_volume DESC
        volumes = [p["total_volume"] for p in profiles]
        assert volumes == sorted(volumes, reverse=True)

    # ========== Signals Tests ==========

    def test_signals_table_exists(self, test_db):
        """Test that signals table exists after init."""
        with test_db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='signals'"
            )
            row = cursor.fetchone()
            assert row is not None
            assert row["name"] == "signals"

    def test_insert_and_get_signal(self, test_db):
        """Test inserting and retrieving a signal."""
        test_db.upsert_market(
            market_id="market_sig_1",
            question="Will BTC hit 100k?",
            category="crypto",
            volume=500000.0
        )

        signal_id = test_db.insert_signal(
            market_id="market_sig_1",
            outcome="Yes",
            conviction_score=0.85,
            smart_money_score=0.9,
            volume_spike_score=0.7,
            cluster_score=0.8,
            contributing_wallets=json.dumps(["0xwallet1", "0xwallet2"]),
            details=json.dumps({"reason": "whale cluster detected"})
        )

        assert signal_id is not None
        assert isinstance(signal_id, int)
        assert signal_id > 0

        # Retrieve by market
        signals = test_db.get_signals_for_market("market_sig_1")
        assert len(signals) == 1
        s = signals[0]
        assert s["signal_id"] == signal_id
        assert s["market_id"] == "market_sig_1"
        assert s["outcome"] == "Yes"
        assert s["conviction_score"] == 0.85
        assert s["smart_money_score"] == 0.9
        assert s["volume_spike_score"] == 0.7
        assert s["cluster_score"] == 0.8

        wallets = json.loads(s["contributing_wallets"])
        assert "0xwallet1" in wallets

    def test_get_recent_signals_with_min_score(self, test_db):
        """Test get_recent_signals with min_score filter."""
        test_db.upsert_market(
            market_id="market_filter",
            question="Filter test market?",
            category="politics",
            volume=100000.0
        )

        # Insert signals with different conviction scores
        test_db.insert_signal(
            market_id="market_filter", outcome="Yes",
            conviction_score=0.3
        )
        test_db.insert_signal(
            market_id="market_filter", outcome="Yes",
            conviction_score=0.6
        )
        test_db.insert_signal(
            market_id="market_filter", outcome="No",
            conviction_score=0.9
        )

        # No filter: all 3
        all_signals = test_db.get_recent_signals()
        assert len(all_signals) == 3

        # min_score=0.5: should get 2 signals
        filtered = test_db.get_recent_signals(min_score=0.5)
        assert len(filtered) == 2
        for s in filtered:
            assert s["conviction_score"] >= 0.5

        # Verify JOIN: market data should be present
        assert filtered[0]["question"] == "Filter test market?"
        assert filtered[0]["category"] == "politics"
        assert filtered[0]["market_volume"] == 100000.0

        # min_score=0.8: should get 1 signal
        high_only = test_db.get_recent_signals(min_score=0.8)
        assert len(high_only) == 1
        assert high_only[0]["conviction_score"] == 0.9

    def test_get_hot_markets(self, test_db):
        """Test get_hot_markets returns correct aggregation."""
        # Create two markets
        test_db.upsert_market(
            market_id="hot_market_1",
            question="Hot market one?",
            category="politics",
            volume=200000.0
        )
        test_db.upsert_market(
            market_id="hot_market_2",
            question="Hot market two?",
            category="crypto",
            volume=150000.0
        )

        # Insert multiple signals for market 1 (more signals = hotter)
        for score in [0.7, 0.8, 0.9]:
            test_db.insert_signal(
                market_id="hot_market_1", outcome="Yes",
                conviction_score=score
            )

        # Insert one signal for market 2
        test_db.insert_signal(
            market_id="hot_market_2", outcome="No",
            conviction_score=0.6
        )

        hot = test_db.get_hot_markets(hours=24, limit=10)
        assert len(hot) == 2

        # Market 1 should be first (more signals)
        assert hot[0]["market_id"] == "hot_market_1"
        assert hot[0]["signal_count"] == 3
        assert hot[0]["max_conviction"] == 0.9
        assert abs(hot[0]["avg_conviction"] - 0.8) < 0.01
        assert hot[0]["question"] == "Hot market one?"
        assert hot[0]["category"] == "politics"
        assert hot[0]["market_volume"] == 200000.0

        # Market 2 should be second
        assert hot[1]["market_id"] == "hot_market_2"
        assert hot[1]["signal_count"] == 1
        assert hot[1]["max_conviction"] == 0.6
