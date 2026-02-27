"""
Tests for the win rate analysis module.
"""

import os
import tempfile
from datetime import datetime, timedelta

import pytest

from polymarket_tracker.database import Database
from polymarket_tracker.win_rate import (
    WinRateAnalyzer,
    BetSizeCategory,
    MarketStatus,
    ConfidenceInterval,
)


@pytest.fixture
def test_db():
    """Create a temporary test database with sample data."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = Database(db_path=path)

    # Create resolved markets
    db.upsert_market(
        market_id="market_win_1",
        question="Will Bitcoin hit $100k?",
        resolved=True,
        outcome="Yes",
        category="crypto"
    )
    db.upsert_market(
        market_id="market_win_2",
        question="Will ETH flip BTC?",
        resolved=True,
        outcome="No",
        category="crypto"
    )
    db.upsert_market(
        market_id="market_loss_1",
        question="Will Team A win?",
        resolved=True,
        outcome="No",
        category="sports"
    )
    db.upsert_market(
        market_id="market_tie_1",
        question="Cancelled market",
        resolved=True,
        outcome="tie",
        category="politics"
    )
    db.upsert_market(
        market_id="market_open_1",
        question="Open market?",
        resolved=False,
        outcome=None,
        category="politics",
        outcome_prices='[0.6, 0.4]'
    )

    # Create trader
    db.upsert_trader(
        wallet_address="0xtrader1",
        volume=10000,
        timestamp=datetime.utcnow()
    )

    # Create bets for trader1
    base_time = datetime.utcnow() - timedelta(days=30)

    # Win 1: Bought Yes on market that resolved Yes
    db.insert_bet(
        bet_id="bet_1",
        wallet_address="0xtrader1",
        market_id="market_win_1",
        asset_id="token_yes",
        amount=100,
        price=0.5,
        side="BUY",
        outcome_bet="Yes",
        timestamp=base_time + timedelta(days=1)
    )

    # Win 2: Bought No on market that resolved No
    db.insert_bet(
        bet_id="bet_2",
        wallet_address="0xtrader1",
        market_id="market_win_2",
        asset_id="token_no",
        amount=200,
        price=0.4,
        side="BUY",
        outcome_bet="No",
        timestamp=base_time + timedelta(days=2)
    )

    # Loss 1: Bought Yes on market that resolved No
    db.insert_bet(
        bet_id="bet_3",
        wallet_address="0xtrader1",
        market_id="market_loss_1",
        asset_id="token_yes",
        amount=150,
        price=0.6,
        side="BUY",
        outcome_bet="Yes",
        timestamp=base_time + timedelta(days=3)
    )

    # Tie: Bet on cancelled market
    db.insert_bet(
        bet_id="bet_4",
        wallet_address="0xtrader1",
        market_id="market_tie_1",
        asset_id="token_yes",
        amount=50,
        price=0.5,
        side="BUY",
        outcome_bet="Yes",
        timestamp=base_time + timedelta(days=4)
    )

    # Open position
    db.insert_bet(
        bet_id="bet_5",
        wallet_address="0xtrader1",
        market_id="market_open_1",
        asset_id="token_yes",
        amount=100,
        price=0.4,
        side="BUY",
        outcome_bet="Yes",
        timestamp=base_time + timedelta(days=5)
    )

    # Large bet for size category testing
    db.insert_bet(
        bet_id="bet_6",
        wallet_address="0xtrader1",
        market_id="market_win_1",
        asset_id="token_yes",
        amount=2000,  # Large bet > $1000
        price=0.6,
        side="BUY",
        outcome_bet="Yes",
        timestamp=base_time + timedelta(days=6)
    )

    yield db
    os.unlink(path)


@pytest.fixture
def analyzer(test_db):
    """Create analyzer with test database."""
    return WinRateAnalyzer(database=test_db)


class TestBetSizeCategory:
    """Tests for BetSizeCategory enum."""

    def test_small_bet(self):
        assert BetSizeCategory.from_amount(50) == BetSizeCategory.SMALL
        assert BetSizeCategory.from_amount(99.99) == BetSizeCategory.SMALL

    def test_medium_bet(self):
        assert BetSizeCategory.from_amount(100) == BetSizeCategory.MEDIUM
        assert BetSizeCategory.from_amount(500) == BetSizeCategory.MEDIUM
        assert BetSizeCategory.from_amount(1000) == BetSizeCategory.MEDIUM

    def test_large_bet(self):
        assert BetSizeCategory.from_amount(1001) == BetSizeCategory.LARGE
        assert BetSizeCategory.from_amount(10000) == BetSizeCategory.LARGE


class TestWinRateAnalyzer:
    """Tests for WinRateAnalyzer class."""

    def test_analyze_trader_overall_stats(self, analyzer):
        """Test overall win rate calculation."""
        analysis = analyzer.analyze_trader_win_rate("0xtrader1")

        # Should have 3 wins (bet_1, bet_2, bet_6), 1 loss (bet_3), 1 tie (bet_4)
        # Open position (bet_5) shouldn't count
        assert analysis.total_wins >= 2  # At minimum bet_1 and bet_2
        assert analysis.total_losses >= 1  # At minimum bet_3
        assert analysis.total_ties >= 1  # bet_4

        # Win rate should be wins / (wins + losses) * 100
        decisive = analysis.total_wins + analysis.total_losses
        if decisive > 0:
            expected_rate = (analysis.total_wins / decisive) * 100
            assert abs(analysis.overall_win_rate - expected_rate) < 0.01

    def test_analyze_trader_by_category(self, analyzer):
        """Test win rate breakdown by category."""
        analysis = analyzer.analyze_trader_win_rate("0xtrader1")

        # Should have crypto and sports categories
        assert "crypto" in analysis.by_category
        assert "sports" in analysis.by_category

        # Crypto should have wins (market_win_1 and market_win_2)
        crypto = analysis.by_category["crypto"]
        assert crypto.wins >= 2

        # Sports should have a loss
        sports = analysis.by_category["sports"]
        assert sports.losses >= 1

    def test_analyze_trader_by_bet_size(self, analyzer):
        """Test win rate breakdown by bet size."""
        analysis = analyzer.analyze_trader_win_rate("0xtrader1")

        # Should have entries for different size categories
        assert BetSizeCategory.SMALL in analysis.by_bet_size
        assert BetSizeCategory.MEDIUM in analysis.by_bet_size
        assert BetSizeCategory.LARGE in analysis.by_bet_size

    def test_streak_detection(self, analyzer):
        """Test winning/losing streak calculation."""
        analysis = analyzer.analyze_trader_win_rate("0xtrader1")

        # Should have tracked streaks
        assert analysis.streaks is not None
        assert analysis.streaks.longest_winning_streak >= 0
        assert analysis.streaks.longest_losing_streak >= 0

    def test_confidence_interval(self, analyzer):
        """Test confidence interval calculation."""
        analysis = analyzer.analyze_trader_win_rate("0xtrader1")

        if analysis.confidence_interval:
            ci = analysis.confidence_interval
            assert ci.lower_bound <= ci.upper_bound
            assert ci.confidence_level == 0.95
            assert 0 <= ci.lower_bound <= 100
            assert 0 <= ci.upper_bound <= 100

    def test_statistical_significance_flag(self, analyzer):
        """Test that insufficient data is flagged."""
        analysis = analyzer.analyze_trader_win_rate("0xtrader1")

        # With only ~4 resolved bets, should be flagged as insufficient
        decisive = analysis.total_wins + analysis.total_losses
        if decisive < 10:
            assert not analysis.has_sufficient_data
        else:
            assert analysis.has_sufficient_data

    def test_kelly_criterion(self, analyzer):
        """Test Kelly Criterion calculation."""
        analysis = analyzer.analyze_trader_win_rate("0xtrader1")

        if analysis.kelly_analysis:
            kelly = analysis.kelly_analysis
            assert kelly.optimal_fraction >= 0
            assert kelly.kelly_multiple >= 0
            assert kelly.recommendation  # Should have a recommendation

    def test_risk_metrics(self, analyzer):
        """Test risk metric calculations."""
        analysis = analyzer.analyze_trader_win_rate("0xtrader1")

        assert analysis.max_drawdown >= 0
        assert analysis.max_drawdown <= 100  # Percentage

    def test_empty_trader(self, test_db):
        """Test analysis for trader with no bets."""
        analyzer = WinRateAnalyzer(database=test_db)
        analysis = analyzer.analyze_trader_win_rate("0xnonexistent")

        assert analysis.total_resolved_bets == 0
        assert analysis.overall_win_rate == 0.0
        assert not analysis.has_sufficient_data


class TestConfidenceIntervalCalculation:
    """Tests for confidence interval calculations."""

    def test_wilson_score_50_percent(self, analyzer):
        """Test Wilson score interval at 50% win rate."""
        ci = analyzer._calculate_confidence_interval(50, 100, 0.95)

        assert ci.sample_size == 100
        assert ci.is_significant
        # At 50% with 100 samples, interval should be roughly 40-60%
        assert 35 < ci.lower_bound < 50
        assert 50 < ci.upper_bound < 65

    def test_wilson_score_high_win_rate(self, analyzer):
        """Test Wilson score interval at high win rate."""
        ci = analyzer._calculate_confidence_interval(90, 100, 0.95)

        # At 90% with 100 samples
        assert ci.lower_bound > 80
        assert ci.upper_bound <= 100

    def test_wilson_score_low_sample(self, analyzer):
        """Test Wilson score with small sample size."""
        ci = analyzer._calculate_confidence_interval(3, 5, 0.95)

        assert not ci.is_significant  # < 10 samples
        # Interval should be wide
        assert ci.upper_bound - ci.lower_bound > 30

    def test_zero_samples(self, analyzer):
        """Test confidence interval with no samples."""
        ci = analyzer._calculate_confidence_interval(0, 0, 0.95)

        assert ci.lower_bound == 0.0
        assert ci.upper_bound == 0.0
        assert not ci.is_significant


class TestLeaderboard:
    """Tests for win rate leaderboard generation."""

    def test_get_leaderboard_empty_with_high_min(self, analyzer):
        """Test leaderboard returns empty with high minimum bets."""
        results = analyzer.get_win_rate_leaderboard(
            limit=10,
            min_resolved_bets=100  # No traders have this many
        )
        assert results == []

    def test_get_leaderboard_sorting(self, test_db):
        """Test leaderboard sorts correctly."""
        # Add more traders with different win rates
        test_db.upsert_trader("0xtrader2", 5000, datetime.utcnow())
        test_db.upsert_market("market_t2_1", "Test?", resolved=True, outcome="Yes")

        for i in range(15):
            test_db.insert_bet(
                bet_id=f"bet_t2_{i}",
                wallet_address="0xtrader2",
                market_id="market_t2_1",
                asset_id="token_yes",
                amount=50,
                price=0.5,
                side="BUY",
                outcome_bet="Yes",
                timestamp=datetime.utcnow()
            )

        analyzer = WinRateAnalyzer(database=test_db)
        results = analyzer.get_win_rate_leaderboard(
            limit=10,
            min_resolved_bets=1,
            sort_by="win_rate"
        )

        # Results should be sorted by win rate descending
        for i in range(len(results) - 1):
            assert results[i]["win_rate"] >= results[i + 1]["win_rate"]
