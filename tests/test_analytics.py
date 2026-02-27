"""
Tests for the analytics module.
"""

import os
import tempfile
from datetime import datetime
from decimal import Decimal

import pytest

from polymarket_tracker.database import Database
from polymarket_tracker.analytics import (
    PolymarketAnalytics,
    LeaderboardMetric,
    TraderPNL,
    POLYMARKET_FEE_RATE,
)


@pytest.fixture
def test_db():
    """Create a temporary test database."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = Database(db_path=path)
    yield db
    os.unlink(path)


@pytest.fixture
def analytics_with_data(test_db):
    """Create analytics instance with sample data."""
    # Create markets
    test_db.upsert_market(
        market_id="market_1",
        question="Test Market 1?",
        resolved=True,
        outcome="Yes",
        outcome_prices='[0.7, 0.3]'
    )
    test_db.upsert_market(
        market_id="market_2",
        question="Test Market 2?",
        resolved=False,
        outcome_prices='[0.6, 0.4]'
    )

    # Create traders
    for i in range(5):
        test_db.upsert_trader(
            wallet_address=f"0xtrader{i}",
            volume=(i + 1) * 1000,
            timestamp=datetime.utcnow()
        )

    # Create bets
    # Trader 0: Won on market_1 (bought Yes at 0.5, market resolved Yes)
    test_db.insert_bet(
        bet_id="bet_1",
        wallet_address="0xtrader0",
        market_id="market_1",
        asset_id="token_yes_1",
        amount=100,  # 100 shares
        price=0.5,   # Cost: $50
        side="BUY",
        outcome_bet="Yes",
        timestamp=datetime.utcnow()
    )

    # Trader 1: Lost on market_1 (bought No at 0.4, market resolved Yes)
    test_db.insert_bet(
        bet_id="bet_2",
        wallet_address="0xtrader1",
        market_id="market_1",
        asset_id="token_no_1",
        amount=100,
        price=0.4,
        side="BUY",
        outcome_bet="No",
        timestamp=datetime.utcnow()
    )

    # Trader 2: Has open position on market_2
    test_db.insert_bet(
        bet_id="bet_3",
        wallet_address="0xtrader2",
        market_id="market_2",
        asset_id="token_yes_2",
        amount=200,
        price=0.4,
        side="BUY",
        outcome_bet="Yes",
        timestamp=datetime.utcnow()
    )

    # Trader 3: Multiple trades
    test_db.insert_bet(
        bet_id="bet_4",
        wallet_address="0xtrader3",
        market_id="market_1",
        asset_id="token_yes_1",
        amount=50,
        price=0.6,
        side="BUY",
        outcome_bet="Yes",
        timestamp=datetime.utcnow()
    )
    test_db.insert_bet(
        bet_id="bet_5",
        wallet_address="0xtrader3",
        market_id="market_2",
        asset_id="token_no_2",
        amount=100,
        price=0.3,
        side="BUY",
        outcome_bet="No",
        timestamp=datetime.utcnow()
    )

    return PolymarketAnalytics(database=test_db)


class TestTraderPNL:
    """Tests for TraderPNL dataclass."""

    def test_win_rate_calculation(self):
        pnl = TraderPNL(
            wallet_address="0xtest",
            winning_trades=7,
            losing_trades=3
        )
        assert pnl.win_rate == 70.0

    def test_win_rate_no_trades(self):
        pnl = TraderPNL(wallet_address="0xtest")
        assert pnl.win_rate == 0.0

    def test_roi_calculation(self):
        pnl = TraderPNL(
            wallet_address="0xtest",
            total_pnl=Decimal("50"),
            total_cost_basis=Decimal("100")
        )
        assert pnl.roi == 50.0

    def test_roi_zero_cost_basis(self):
        pnl = TraderPNL(wallet_address="0xtest")
        assert pnl.roi == 0.0


class TestPolymarketAnalytics:
    """Tests for PolymarketAnalytics class."""

    def test_calculate_total_volume(self, analytics_with_data):
        """Test volume calculation."""
        # Trader 0: 100 * 0.5 = 50
        volume = analytics_with_data.calculate_total_volume("0xtrader0")
        assert volume == 50.0

        # Trader 3: (50 * 0.6) + (100 * 0.3) = 30 + 30 = 60
        volume = analytics_with_data.calculate_total_volume("0xtrader3")
        assert volume == 60.0

    def test_calculate_trader_pnl_winner(self, analytics_with_data):
        """Test PNL calculation for a winning trader."""
        pnl = analytics_with_data.calculate_trader_pnl("0xtrader0")

        # Trader 0 bought 100 Yes shares at 0.5 = $50 cost
        # Market resolved Yes, payout = 100 * $1 = $100
        # Gross profit = $100 - $50 = $50
        # Fee = $50 * 0.02 = $1
        # Net profit = $50 - $1 = $49
        assert pnl.winning_trades == 1
        assert pnl.losing_trades == 0
        assert float(pnl.realized_pnl) == pytest.approx(49.0, rel=0.01)

    def test_calculate_trader_pnl_loser(self, analytics_with_data):
        """Test PNL calculation for a losing trader."""
        pnl = analytics_with_data.calculate_trader_pnl("0xtrader1")

        # Trader 1 bought 100 No shares at 0.4 = $40 cost
        # Market resolved Yes, No shares worth $0
        # Loss = -$40
        assert pnl.winning_trades == 0
        assert pnl.losing_trades == 1
        assert float(pnl.realized_pnl) == pytest.approx(-40.0, rel=0.01)

    def test_get_trader_stats(self, analytics_with_data):
        """Test comprehensive trader stats."""
        stats = analytics_with_data.get_trader_stats("0xtrader3")

        assert stats.total_trades == 2
        assert stats.markets_participated == 2
        assert stats.total_volume == pytest.approx(60.0, rel=0.01)
        assert stats.pnl is not None

    def test_get_top_traders_by_volume(self, analytics_with_data):
        """Test leaderboard by volume."""
        entries = analytics_with_data.get_top_traders(
            metric=LeaderboardMetric.VOLUME,
            limit=3
        )

        assert len(entries) <= 3
        # Should be sorted by volume descending
        for i in range(len(entries) - 1):
            assert entries[i].total_volume >= entries[i + 1].total_volume

    def test_get_top_traders_by_pnl(self, analytics_with_data):
        """Test leaderboard by PNL."""
        entries = analytics_with_data.get_top_traders(
            metric=LeaderboardMetric.PNL,
            limit=5
        )

        assert len(entries) <= 5
        # Should be sorted by PNL descending
        for i in range(len(entries) - 1):
            assert entries[i].total_pnl >= entries[i + 1].total_pnl

    def test_get_leaderboard_summary(self, analytics_with_data):
        """Test summary statistics."""
        summary = analytics_with_data.get_leaderboard_summary()

        assert "total_traders" in summary
        assert "total_trades" in summary
        assert "total_volume" in summary
        assert summary["total_traders"] > 0
        assert summary["total_trades"] > 0

    def test_leaderboard_entry_ranks(self, analytics_with_data):
        """Test that leaderboard entries have correct ranks."""
        entries = analytics_with_data.get_top_traders(limit=5)

        for i, entry in enumerate(entries):
            assert entry.rank == i + 1
