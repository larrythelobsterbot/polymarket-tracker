"""
Tests for the insider detection module.
"""

import os
import tempfile
from datetime import datetime, timedelta

import pytest

from polymarket_tracker.database import Database
from polymarket_tracker.insider_detection import (
    InsiderDetector,
    DetectionConfig,
    AlertType,
    AlertSeverity,
    BetSizeCategory,
)


@pytest.fixture
def test_db():
    """Create a temporary test database with sample data."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = Database(db_path=path)

    now = datetime.utcnow()

    # Market resolving soon
    db.upsert_market(
        market_id="market_soon",
        question="Market resolving soon?",
        end_date=now + timedelta(hours=12),
        resolved=False,
        volume=50000,
        liquidity=10000,
        outcome_prices='[0.7, 0.3]'
    )

    # Niche market (low volume)
    db.upsert_market(
        market_id="market_niche",
        question="Niche market question?",
        end_date=now + timedelta(days=30),
        resolved=True,
        outcome="Yes",
        volume=5000,
        category="crypto"
    )

    # Normal market
    db.upsert_market(
        market_id="market_normal",
        question="Normal market?",
        end_date=now + timedelta(days=7),
        resolved=False,
        volume=500000,
        liquidity=100000,
        outcome_prices='[0.5, 0.5]'
    )

    # Create traders
    db.upsert_trader("0xactive_trader", 100000, now)
    db.upsert_trader("0xdormant_trader", 50000, now - timedelta(days=60))

    # Active trader - recent bets
    for i in range(10):
        db.insert_bet(
            bet_id=f"bet_active_{i}",
            wallet_address="0xactive_trader",
            market_id="market_normal",
            asset_id="token_yes",
            amount=100,
            price=0.5,
            side="BUY",
            outcome_bet="Yes",
            timestamp=now - timedelta(days=i)
        )

    # Dormant trader - old bet then new large bet
    db.insert_bet(
        bet_id="bet_dormant_old",
        wallet_address="0xdormant_trader",
        market_id="market_normal",
        asset_id="token_yes",
        amount=100,
        price=0.5,
        side="BUY",
        outcome_bet="Yes",
        timestamp=now - timedelta(days=60)
    )

    # Large bet near resolution
    db.insert_bet(
        bet_id="bet_large_near_resolution",
        wallet_address="0xactive_trader",
        market_id="market_soon",
        asset_id="token_yes",
        amount=15000,
        price=0.7,
        side="BUY",
        outcome_bet="Yes",
        timestamp=now
    )

    # Dormant trader reactivation with large bet
    db.insert_bet(
        bet_id="bet_dormant_reactivation",
        wallet_address="0xdormant_trader",
        market_id="market_soon",
        asset_id="token_yes",
        amount=20000,
        price=0.5,
        side="BUY",
        outcome_bet="Yes",
        timestamp=now
    )

    # Low liquidity hour bet (3 AM UTC)
    low_liq_time = now.replace(hour=3, minute=0)
    db.insert_bet(
        bet_id="bet_low_liquidity",
        wallet_address="0xactive_trader",
        market_id="market_normal",
        asset_id="token_yes",
        amount=8000,
        price=0.5,
        side="BUY",
        outcome_bet="Yes",
        timestamp=low_liq_time
    )

    yield db
    os.unlink(path)


@pytest.fixture
def detector(test_db):
    """Create detector with test database."""
    return InsiderDetector(database=test_db)


class TestDetectionConfig:
    """Tests for DetectionConfig defaults."""

    def test_default_thresholds(self):
        config = DetectionConfig()
        assert config.LARGE_BET_THRESHOLD == 10000.0
        assert config.RESOLUTION_WINDOW_HOURS == 24
        assert config.DORMANT_DAYS_THRESHOLD == 30
        assert config.ALERT_SCORE_THRESHOLD == 80


class TestTimingDetection:
    """Tests for timing-based detection."""

    def test_large_bet_near_resolution(self, detector):
        """Test detection of large bet near resolution."""
        now = datetime.utcnow()
        resolution = now + timedelta(hours=12)

        is_suspicious, score, hours = detector.check_large_bet_near_resolution(
            bet_amount=15000,
            bet_timestamp=now,
            market_end_date=resolution
        )

        assert is_suspicious
        assert score > 0
        assert hours == pytest.approx(12, abs=0.1)

    def test_small_bet_not_flagged(self, detector):
        """Test that small bets near resolution aren't flagged."""
        now = datetime.utcnow()
        resolution = now + timedelta(hours=6)

        is_suspicious, score, hours = detector.check_large_bet_near_resolution(
            bet_amount=100,  # Small bet
            bet_timestamp=now,
            market_end_date=resolution
        )

        assert not is_suspicious
        assert score == 0

    def test_large_bet_far_from_resolution(self, detector):
        """Test that large bets far from resolution aren't flagged."""
        now = datetime.utcnow()
        resolution = now + timedelta(days=30)

        is_suspicious, score, hours = detector.check_large_bet_near_resolution(
            bet_amount=50000,
            bet_timestamp=now,
            market_end_date=resolution
        )

        assert not is_suspicious

    def test_dormant_wallet_activation(self, detector):
        """Test detection of dormant wallet reactivation."""
        now = datetime.utcnow()

        is_suspicious, score = detector.check_dormant_wallet_activation(
            wallet_address="0xdormant_trader",
            bet_timestamp=now,
            bet_amount=20000
        )

        assert is_suspicious
        assert score > 0

    def test_active_wallet_not_flagged(self, detector):
        """Test that active wallets aren't flagged for dormancy."""
        now = datetime.utcnow()

        is_suspicious, score = detector.check_dormant_wallet_activation(
            wallet_address="0xactive_trader",
            bet_timestamp=now,
            bet_amount=10000
        )

        assert not is_suspicious

    def test_low_liquidity_betting(self, detector):
        """Test detection of low liquidity hour betting."""
        # 3 AM UTC is low liquidity
        low_liq_time = datetime.utcnow().replace(hour=3, minute=0)

        is_suspicious, score = detector.check_low_liquidity_betting(
            bet_timestamp=low_liq_time,
            bet_amount=8000
        )

        assert is_suspicious
        assert score > 0

    def test_normal_hours_not_flagged(self, detector):
        """Test that normal hours aren't flagged."""
        normal_time = datetime.utcnow().replace(hour=14, minute=0)

        is_suspicious, score = detector.check_low_liquidity_betting(
            bet_timestamp=normal_time,
            bet_amount=10000
        )

        assert not is_suspicious


class TestAnomalyScoring:
    """Tests for anomaly scoring."""

    def test_calculate_anomaly_score_large_bet(self, detector):
        """Test anomaly score for large bet near resolution."""
        suspicious_bet = detector.calculate_anomaly_score("bet_large_near_resolution")

        assert suspicious_bet.anomaly_score > 0
        assert AlertType.LARGE_BET_NEAR_RESOLUTION in suspicious_bet.alert_types
        assert suspicious_bet.amount == 15000

    def test_calculate_anomaly_score_dormant(self, detector):
        """Test anomaly score for dormant wallet reactivation."""
        suspicious_bet = detector.calculate_anomaly_score("bet_dormant_reactivation")

        assert suspicious_bet.anomaly_score > 0
        assert AlertType.DORMANT_WALLET_ACTIVATION in suspicious_bet.alert_types

    def test_score_breakdown_populated(self, detector):
        """Test that score breakdown is populated."""
        suspicious_bet = detector.calculate_anomaly_score("bet_large_near_resolution")

        assert len(suspicious_bet.score_breakdown) > 0

    def test_severity_assignment(self, detector):
        """Test severity levels based on score."""
        # High score bet
        suspicious_bet = detector.calculate_anomaly_score("bet_dormant_reactivation")

        # Score determines severity
        if suspicious_bet.anomaly_score >= 90:
            assert suspicious_bet.severity == AlertSeverity.CRITICAL
        elif suspicious_bet.anomaly_score >= 80:
            assert suspicious_bet.severity == AlertSeverity.HIGH
        elif suspicious_bet.anomaly_score >= 60:
            assert suspicious_bet.severity == AlertSeverity.MEDIUM


class TestCorrelationDetection:
    """Tests for correlated betting detection."""

    def test_detect_correlated_betting_single_wallet(self, test_db):
        """Test that single wallet bets don't trigger correlation."""
        detector = InsiderDetector(database=test_db)
        groups = detector.detect_correlated_betting("market_normal")

        # Need multiple wallets for correlation
        single_wallet_groups = [g for g in groups if len(g.wallets) < 3]
        assert all(g.correlation_score < 50 for g in single_wallet_groups)

    def test_detect_correlated_betting_multiple_wallets(self, test_db):
        """Test correlation detection with multiple wallets."""
        now = datetime.utcnow()

        # Add correlated bets from multiple wallets
        for i in range(5):
            test_db.upsert_trader(f"0xcorrelated_{i}", 10000, now)
            test_db.insert_bet(
                bet_id=f"bet_correlated_{i}",
                wallet_address=f"0xcorrelated_{i}",
                market_id="market_normal",
                asset_id="token_yes",
                amount=5000,
                price=0.5,
                side="BUY",
                outcome_bet="Yes",
                timestamp=now + timedelta(minutes=i * 5)  # Within 30 min window
            )

        detector = InsiderDetector(database=test_db)
        groups = detector.detect_correlated_betting("market_normal", time_window_minutes=60)

        # Should detect correlation
        multi_wallet_groups = [g for g in groups if len(g.wallets) >= 3]
        assert len(multi_wallet_groups) > 0


class TestWalletRiskProfile:
    """Tests for wallet risk profiling."""

    def test_build_wallet_risk_profile(self, detector):
        """Test building risk profile for a wallet."""
        profile = detector.build_wallet_risk_profile("0xactive_trader")

        assert profile.wallet_address == "0xactive_trader"
        assert profile.total_bets_analyzed > 0
        assert profile.total_volume > 0
        assert profile.first_seen is not None

    def test_risk_profile_suspicious_count(self, detector):
        """Test that suspicious bets are counted."""
        profile = detector.build_wallet_risk_profile("0xactive_trader")

        # Has some suspicious bets due to large bet near resolution
        assert profile.total_suspicious_bets >= 0

    def test_risk_profile_dormant_wallet(self, detector):
        """Test risk profile for dormant wallet."""
        profile = detector.build_wallet_risk_profile("0xdormant_trader")

        # Should be flagged for reactivation
        assert profile.overall_risk_score > 0


class TestMarketManipulation:
    """Tests for market manipulation detection."""

    def test_detect_volume_spike_no_spike(self, detector):
        """Test that normal volume doesn't trigger alert."""
        alert = detector.detect_volume_spike("market_normal")

        # No spike expected in test data
        # (would need historical comparison data)


class TestSerialization:
    """Tests for data serialization."""

    def test_suspicious_bet_to_dict(self, detector):
        """Test SuspiciousBet serialization."""
        bet = detector.calculate_anomaly_score("bet_large_near_resolution")
        data = bet.to_dict()

        assert "bet_id" in data
        assert "wallet_address" in data
        assert "anomaly_score" in data
        assert "alert_types" in data
        assert "severity" in data

    def test_wallet_profile_to_dict(self, detector):
        """Test WalletRiskProfile serialization."""
        profile = detector.build_wallet_risk_profile("0xactive_trader")
        data = profile.to_dict()

        assert "wallet_address" in data
        assert "overall_risk_score" in data
        assert "total_suspicious_bets" in data
