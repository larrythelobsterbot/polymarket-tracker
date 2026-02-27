"""
Tests for the signal engine module.
"""

import json
import os
import tempfile
from datetime import datetime, timedelta

import pytest

from polymarket_tracker.database import Database
from polymarket_tracker.signal_engine import SignalConfig, SignalEngine


@pytest.fixture
def test_db():
    """Create a temporary test database."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = Database(db_path=path)
    yield db
    os.unlink(path)


@pytest.fixture
def engine(test_db):
    """Create a SignalEngine backed by the test database."""
    return SignalEngine(database=test_db, config=SignalConfig())


def _seed_trader_and_market(db, wallet="0xwhale1", market_id="market_1",
                            category="politics", volume=100000.0):
    """Helper: insert a trader and a market."""
    db.upsert_trader(wallet_address=wallet, volume=volume, timestamp=datetime.utcnow())
    db.upsert_market(
        market_id=market_id,
        question="Test market?",
        category=category,
        volume=volume,
    )


def _seed_whale(db, wallet="0xwhale1", grade="A", win_rate=0.72,
                category_specialization=None):
    """Helper: insert a whale profile (trader must already exist)."""
    cat_spec = category_specialization or json.dumps({"politics": 70.0, "crypto": 30.0})
    db.upsert_whale_profile(
        wallet_address=wallet,
        grade=grade,
        win_rate=win_rate,
        category_specialization=cat_spec,
        total_volume=50000.0,
        total_resolved_bets=100,
    )


def _insert_bets(db, market_id, count, amount, minutes_ago_start=60):
    """Helper: insert N bets for a market spread across time."""
    now = datetime.utcnow()
    for i in range(count):
        ts = now - timedelta(minutes=minutes_ago_start - i)
        db.upsert_trader(wallet_address=f"0xfiller{i}", volume=amount, timestamp=ts)
        db.insert_bet(
            bet_id=f"filler_bet_{market_id}_{i}",
            wallet_address=f"0xfiller{i}",
            market_id=market_id,
            asset_id="asset_1",
            amount=amount,
            price=0.5,
            side="BUY",
            outcome_bet="Yes",
            timestamp=ts,
        )


class TestSmartMoney:
    """Tests for score_smart_money."""

    def test_smart_money_score_a_grade(self, test_db, engine):
        """A-grade whale should score >= 70."""
        _seed_trader_and_market(test_db)
        _seed_whale(test_db, grade="A")

        score = engine.score_smart_money("0xwhale1", 5000.0, "market_1")
        assert score >= 70

    def test_smart_money_score_no_profile(self, test_db, engine):
        """Non-whale (no profile) should score 0."""
        _seed_trader_and_market(test_db, wallet="0xnobody")

        score = engine.score_smart_money("0xnobody", 5000.0, "market_1")
        assert score == 0

    def test_smart_money_score_category_boost(self, test_db, engine):
        """Whale with 70% in matching category gets boosted above base."""
        _seed_trader_and_market(test_db, category="politics")
        # B-grade base is 70; with 70% win rate in politics (>= 65%), gets +10 = 80
        _seed_whale(
            test_db,
            grade="B",
            category_specialization=json.dumps({"politics": 70.0, "crypto": 30.0}),
        )

        score = engine.score_smart_money("0xwhale1", 5000.0, "market_1")
        # B-grade base = 70, +10 boost = 80
        assert score == 80.0


class TestVolumeSpike:
    """Tests for score_volume_spike.

    Note: _insert_bets uses amount=100, price=0.5, so avg USD = 50.0.
    The score_volume_spike method now uses AVG(amount * price) for comparison.
    """

    def test_volume_spike_above_threshold(self, test_db, engine):
        """Bet 4x above avg USD should score >= 50."""
        _seed_trader_and_market(test_db)
        # Insert 10 bets at amount=100, price=0.5 => avg USD = 50
        _insert_bets(test_db, "market_1", count=10, amount=100.0)

        # A bet worth $200 USD is 4x the avg $50 => above 3x threshold
        score = engine.score_volume_spike("market_1", 200.0)
        assert score >= 50

    def test_volume_spike_below_threshold(self, test_db, engine):
        """Bet below 3x avg USD should score 0."""
        _seed_trader_and_market(test_db)
        # Insert 10 bets at amount=100, price=0.5 => avg USD = 50
        _insert_bets(test_db, "market_1", count=10, amount=100.0)

        # A bet worth $100 USD is 2x the avg $50 => below 3x threshold
        score = engine.score_volume_spike("market_1", 100.0)
        assert score == 0

    def test_volume_spike_insufficient_data(self, test_db, engine):
        """Fewer than 5 bets should score 0."""
        _seed_trader_and_market(test_db)
        # Insert only 3 bets
        _insert_bets(test_db, "market_1", count=3, amount=100.0)

        score = engine.score_volume_spike("market_1", 500.0)
        assert score == 0


class TestCluster:
    """Tests for score_cluster."""

    def test_cluster_score_three_wallets(self, test_db, engine):
        """Exactly 3 wallets buying same outcome should score 50."""
        _seed_trader_and_market(test_db)
        now = datetime.utcnow()

        # Insert 3 BUY bets from different wallets within the cluster window
        for i in range(3):
            wallet = f"0xcluster{i}"
            test_db.upsert_trader(wallet_address=wallet, volume=100.0, timestamp=now)
            test_db.insert_bet(
                bet_id=f"cluster_bet_{i}",
                wallet_address=wallet,
                market_id="market_1",
                asset_id="asset_1",
                amount=100.0,
                price=0.5,
                side="BUY",
                outcome_bet="Yes",
                timestamp=now - timedelta(minutes=i * 5),  # All within 30-min window
            )

        score = engine.score_cluster("market_1", "Yes", now)
        assert score == 50.0

    def test_cluster_score_below_minimum(self, test_db, engine):
        """Only 2 wallets should score 0."""
        _seed_trader_and_market(test_db)
        now = datetime.utcnow()

        for i in range(2):
            wallet = f"0xcluster{i}"
            test_db.upsert_trader(wallet_address=wallet, volume=100.0, timestamp=now)
            test_db.insert_bet(
                bet_id=f"cluster_bet_{i}",
                wallet_address=wallet,
                market_id="market_1",
                asset_id="asset_1",
                amount=100.0,
                price=0.5,
                side="BUY",
                outcome_bet="Yes",
                timestamp=now - timedelta(minutes=i * 5),
            )

        score = engine.score_cluster("market_1", "Yes", now)
        assert score == 0


class TestComposite:
    """Tests for calculate_composite_score."""

    def test_composite_score_calculation(self, engine):
        """Verify (80*0.4) + (60*0.3) + (0*0.3) = 50."""
        score = engine.calculate_composite_score(80, 60, 0)
        assert score == pytest.approx(50.0)


class TestEvaluateBet:
    """Tests for evaluate_bet end-to-end."""

    def test_evaluate_bet_above_threshold(self, test_db, engine):
        """A bet that triggers strong signals should fire and persist."""
        _seed_trader_and_market(test_db, category="politics")
        _seed_whale(test_db, grade="A",
                     category_specialization=json.dumps({"politics": 70.0}))
        now = datetime.utcnow()

        # Build volume baseline (10 bets at amount=100, price=0.5 => avg USD = 50)
        _insert_bets(test_db, "market_1", count=10, amount=100.0)

        # Create cluster: 3 other wallets buying "Yes" within the window
        for i in range(3):
            w = f"0xcluster{i}"
            test_db.upsert_trader(wallet_address=w, volume=100.0, timestamp=now)
            test_db.insert_bet(
                bet_id=f"eval_cluster_{i}",
                wallet_address=w,
                market_id="market_1",
                asset_id="asset_1",
                amount=100.0,
                price=0.5,
                side="BUY",
                outcome_bet="Yes",
                timestamp=now - timedelta(minutes=i * 2),
            )

        # The whale's large bet: amount=2000 shares at price=0.5 => $1000 USD
        # avg USD baseline = $50, ratio = $1000/$50 = 20x spike
        # Smart money: A-grade(90) + category boost(+10) = 100
        # Volume spike: 20x => 50 + (20-3)*10 = 220, capped at 100
        # Cluster: 3 wallets in small market = 50
        # Composite: 100*0.4 + 100*0.3 + 50*0.3 = 40 + 30 + 15 = 85
        result = engine.evaluate_bet(
            wallet_address="0xwhale1",
            market_id="market_1",
            outcome="Yes",
            amount=2000.0,
            price=0.5,
            timestamp=now,
        )

        assert result is not None
        assert result["signal_id"] > 0
        assert result["market_id"] == "market_1"
        assert result["outcome"] == "Yes"
        assert result["conviction_score"] >= 60.0

        # Verify it was persisted in the DB
        signals = test_db.get_signals_for_market("market_1")
        assert len(signals) >= 1
        stored = signals[0]
        assert stored["conviction_score"] == result["conviction_score"]

    def test_evaluate_bet_below_threshold(self, test_db, engine):
        """A weak bet should return None and not persist anything."""
        _seed_trader_and_market(test_db, wallet="0xnobody")
        now = datetime.utcnow()

        # No whale profile, no volume spike, no cluster => all zeros
        result = engine.evaluate_bet(
            wallet_address="0xnobody",
            market_id="market_1",
            outcome="Yes",
            amount=10.0,
            price=0.5,
            timestamp=now,
        )

        assert result is None

        # Nothing should be in the signals table
        signals = test_db.get_signals_for_market("market_1")
        assert len(signals) == 0
