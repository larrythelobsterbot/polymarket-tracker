"""
Tests for the whale profiler module.
"""

import json
import os
import tempfile
from datetime import datetime, timedelta

import pytest

from polymarket_tracker.database import Database
from polymarket_tracker.whale_profiler import WhaleConfig, WhaleProfiler


@pytest.fixture
def test_db():
    """Create a temporary test database."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = Database(db_path=path)
    yield db
    os.unlink(path)


@pytest.fixture
def profiler(test_db):
    """Create a WhaleProfiler with the test database."""
    return WhaleProfiler(database=test_db)


class TestClassifyGrade:
    """Tests for the classify_grade static method."""

    def test_classify_grade_a(self):
        """70% win rate with 25 bets should be grade A."""
        grade = WhaleProfiler.classify_grade(win_rate=70.0, resolved_bets=25)
        assert grade == "A"

    def test_classify_grade_b(self):
        """58% win rate with 25 bets should be grade B."""
        grade = WhaleProfiler.classify_grade(win_rate=58.0, resolved_bets=25)
        assert grade == "B"

    def test_classify_grade_c(self):
        """48% win rate with 25 bets should be grade C."""
        grade = WhaleProfiler.classify_grade(win_rate=48.0, resolved_bets=25)
        assert grade == "C"

    def test_classify_grade_d(self):
        """30% win rate with 25 bets should be grade D."""
        grade = WhaleProfiler.classify_grade(win_rate=30.0, resolved_bets=25)
        assert grade == "D"

    def test_classify_grade_ungraded(self):
        """90% win rate with only 10 bets should be ungraded (below min_bets)."""
        grade = WhaleProfiler.classify_grade(win_rate=90.0, resolved_bets=10)
        assert grade == "ungraded"

    def test_classify_grade_exact_thresholds(self):
        """Test exact boundary values for each grade."""
        assert WhaleProfiler.classify_grade(65.0, 20) == "A"
        assert WhaleProfiler.classify_grade(64.9, 20) == "B"
        assert WhaleProfiler.classify_grade(55.0, 20) == "B"
        assert WhaleProfiler.classify_grade(54.9, 20) == "C"
        assert WhaleProfiler.classify_grade(45.0, 20) == "C"
        assert WhaleProfiler.classify_grade(44.9, 20) == "D"

    def test_classify_grade_custom_min_bets(self):
        """Test with a custom min_bets threshold."""
        # With min_bets=5, 10 bets is enough
        grade = WhaleProfiler.classify_grade(win_rate=70.0, resolved_bets=10, min_bets=5)
        assert grade == "A"


class TestIdentifyWhaleAddresses:
    """Tests for whale identification by volume and single bet size."""

    def test_identify_whales_by_volume(self, test_db, profiler):
        """Wallet with 60k volume is a whale, 5k volume is not."""
        now = datetime.utcnow()

        # High-volume whale
        test_db.upsert_trader("0xwhale_big", volume=60000.0, timestamp=now)
        # Small fish
        test_db.upsert_trader("0xsmall_fish", volume=5000.0, timestamp=now)

        whales = profiler.identify_whale_addresses()
        assert "0xwhale_big" in whales
        assert "0xsmall_fish" not in whales

    def test_identify_whales_by_single_bet(self, test_db, profiler):
        """Wallet with a $6k single bet is a whale even if total volume is low."""
        now = datetime.utcnow()

        # Low total volume, but one big bet
        test_db.upsert_trader("0xbig_bettor", volume=10000.0, timestamp=now)
        test_db.upsert_market(
            market_id="market_whale_test",
            question="Whale test market?",
        )
        # amount=6000, price=1.0 -> amount*price = 6000 >= 5000 threshold
        test_db.insert_bet(
            bet_id="bet_whale_big",
            wallet_address="0xbig_bettor",
            market_id="market_whale_test",
            asset_id="asset_1",
            amount=6000.0,
            price=1.0,
            side="BUY",
            outcome_bet="Yes",
            timestamp=now,
        )

        # Another wallet with only small bets
        test_db.upsert_trader("0xsmall_bettor", volume=10000.0, timestamp=now)
        test_db.insert_bet(
            bet_id="bet_small_1",
            wallet_address="0xsmall_bettor",
            market_id="market_whale_test",
            asset_id="asset_1",
            amount=100.0,
            price=0.5,
            side="BUY",
            outcome_bet="Yes",
            timestamp=now,
        )

        whales = profiler.identify_whale_addresses()
        assert "0xbig_bettor" in whales
        assert "0xsmall_bettor" not in whales

    def test_identify_whales_combined(self, test_db, profiler):
        """Both volume-based and bet-size-based whales appear in the result."""
        now = datetime.utcnow()

        # Volume whale
        test_db.upsert_trader("0xvolume_whale", volume=80000.0, timestamp=now)

        # Bet-size whale
        test_db.upsert_trader("0xbet_whale", volume=3000.0, timestamp=now)
        test_db.upsert_market(market_id="market_combined", question="Combined test?")
        test_db.insert_bet(
            bet_id="bet_combined_big",
            wallet_address="0xbet_whale",
            market_id="market_combined",
            asset_id="asset_1",
            amount=10000.0,
            price=0.6,
            side="BUY",
            outcome_bet="Yes",
            timestamp=now,
        )

        whales = profiler.identify_whale_addresses()
        assert "0xvolume_whale" in whales
        assert "0xbet_whale" in whales


class TestCalculateWalletStats:
    """Tests for wallet stats calculation on resolved markets."""

    def test_calculate_win_rate_resolved_only(self, test_db, profiler):
        """Verify correct win rate from resolved markets with wins and losses."""
        now = datetime.utcnow()
        test_db.upsert_trader("0xstats_wallet", volume=100000.0, timestamp=now)

        # Create resolved markets
        # Market 1: resolved Yes
        test_db.upsert_market(
            market_id="m_resolved_yes",
            question="Will it rain?",
            resolved=True,
            outcome="Yes",
            category="weather",
        )
        # Market 2: resolved No
        test_db.upsert_market(
            market_id="m_resolved_no",
            question="Will it snow?",
            resolved=True,
            outcome="No",
            category="weather",
        )
        # Market 3: resolved Yes
        test_db.upsert_market(
            market_id="m_resolved_yes2",
            question="Will BTC go up?",
            resolved=True,
            outcome="Yes",
            category="crypto",
        )
        # Market 4: unresolved (should be ignored)
        test_db.upsert_market(
            market_id="m_unresolved",
            question="Open market?",
            resolved=False,
            outcome=None,
        )
        # Market 5: resolved as tie (should be skipped)
        test_db.upsert_market(
            market_id="m_tie",
            question="Tied market?",
            resolved=True,
            outcome="tie",
            category="sports",
        )

        base = now - timedelta(days=10)

        # WIN: BUY Yes on market that resolved Yes
        test_db.insert_bet(
            bet_id="stat_bet_1", wallet_address="0xstats_wallet",
            market_id="m_resolved_yes", asset_id="a1",
            amount=100, price=0.5, side="BUY", outcome_bet="Yes",
            timestamp=base + timedelta(days=1),
        )
        # WIN: BUY No on market that resolved No
        test_db.insert_bet(
            bet_id="stat_bet_2", wallet_address="0xstats_wallet",
            market_id="m_resolved_no", asset_id="a2",
            amount=200, price=0.4, side="BUY", outcome_bet="No",
            timestamp=base + timedelta(days=2),
        )
        # LOSS: BUY Yes on market that resolved Yes (crypto) - actually this is a win
        # Let's make a loss: BUY No on market that resolved Yes
        test_db.insert_bet(
            bet_id="stat_bet_3", wallet_address="0xstats_wallet",
            market_id="m_resolved_yes2", asset_id="a3",
            amount=150, price=0.6, side="BUY", outcome_bet="No",
            timestamp=base + timedelta(days=3),
        )
        # Bet on unresolved market (should be ignored)
        test_db.insert_bet(
            bet_id="stat_bet_4", wallet_address="0xstats_wallet",
            market_id="m_unresolved", asset_id="a4",
            amount=100, price=0.5, side="BUY", outcome_bet="Yes",
            timestamp=base + timedelta(days=4),
        )
        # Bet on tie market (should be skipped in win/loss count)
        test_db.insert_bet(
            bet_id="stat_bet_5", wallet_address="0xstats_wallet",
            market_id="m_tie", asset_id="a5",
            amount=50, price=0.5, side="BUY", outcome_bet="Yes",
            timestamp=base + timedelta(days=5),
        )

        stats = profiler.calculate_wallet_stats("0xstats_wallet")

        # 2 wins (m_resolved_yes, m_resolved_no), 1 loss (m_resolved_yes2)
        assert stats["wins"] == 2
        assert stats["losses"] == 1
        assert stats["resolved_bets"] == 3
        # win_rate = 2/3 * 100 = 66.67%
        assert abs(stats["win_rate"] - 66.67) < 0.1
        assert stats["total_volume"] == 100000.0

    def test_no_double_counting_per_market(self, test_db, profiler):
        """Multiple bets on same market should only count once."""
        now = datetime.utcnow()
        test_db.upsert_trader("0xdouble_wallet", volume=50000.0, timestamp=now)

        test_db.upsert_market(
            market_id="m_double",
            question="Double-bet market?",
            resolved=True,
            outcome="Yes",
            category="politics",
        )

        # Two BUY bets on same market
        for i in range(3):
            test_db.insert_bet(
                bet_id=f"double_bet_{i}",
                wallet_address="0xdouble_wallet",
                market_id="m_double",
                asset_id="a1",
                amount=100, price=0.5, side="BUY", outcome_bet="Yes",
                timestamp=now - timedelta(hours=i),
            )

        stats = profiler.calculate_wallet_stats("0xdouble_wallet")

        # Should count as 1 resolved bet, not 3
        assert stats["resolved_bets"] == 1
        assert stats["wins"] == 1

    def test_category_specialization(self, test_db, profiler):
        """Verify category-level win rate tracking."""
        now = datetime.utcnow()
        test_db.upsert_trader("0xcat_wallet", volume=80000.0, timestamp=now)

        # 2 crypto wins
        for i in range(2):
            test_db.upsert_market(
                market_id=f"m_crypto_win_{i}",
                question=f"Crypto win {i}?",
                resolved=True, outcome="Yes", category="crypto",
            )
            test_db.insert_bet(
                bet_id=f"cat_crypto_win_{i}",
                wallet_address="0xcat_wallet",
                market_id=f"m_crypto_win_{i}",
                asset_id="a1", amount=100, price=0.5,
                side="BUY", outcome_bet="Yes",
                timestamp=now - timedelta(days=i),
            )

        # 1 crypto loss
        test_db.upsert_market(
            market_id="m_crypto_loss",
            question="Crypto loss?",
            resolved=True, outcome="No", category="crypto",
        )
        test_db.insert_bet(
            bet_id="cat_crypto_loss",
            wallet_address="0xcat_wallet",
            market_id="m_crypto_loss",
            asset_id="a1", amount=100, price=0.5,
            side="BUY", outcome_bet="Yes",
            timestamp=now - timedelta(days=3),
        )

        # 1 politics win
        test_db.upsert_market(
            market_id="m_politics_win",
            question="Politics win?",
            resolved=True, outcome="Yes", category="politics",
        )
        test_db.insert_bet(
            bet_id="cat_politics_win",
            wallet_address="0xcat_wallet",
            market_id="m_politics_win",
            asset_id="a1", amount=100, price=0.5,
            side="BUY", outcome_bet="Yes",
            timestamp=now - timedelta(days=4),
        )

        stats = profiler.calculate_wallet_stats("0xcat_wallet")

        cat_spec = stats["category_specialization"]
        # Crypto: 2 wins / 3 total = 66.67%
        assert abs(cat_spec["crypto"] - 66.67) < 0.1
        # Politics: 1 win / 1 total = 100%
        assert cat_spec["politics"] == 100.0

    def test_sell_side_logic(self, test_db, profiler):
        """SELL on winning outcome = loss, SELL on losing outcome = win."""
        now = datetime.utcnow()
        test_db.upsert_trader("0xsell_wallet", volume=70000.0, timestamp=now)

        # Market resolves Yes
        test_db.upsert_market(
            market_id="m_sell_test",
            question="Sell test?",
            resolved=True, outcome="Yes", category="test",
        )

        # SELL on Yes (the winning outcome) = LOSS
        test_db.insert_bet(
            bet_id="sell_bet_1",
            wallet_address="0xsell_wallet",
            market_id="m_sell_test",
            asset_id="a1", amount=100, price=0.5,
            side="SELL", outcome_bet="Yes",
            timestamp=now,
        )

        stats = profiler.calculate_wallet_stats("0xsell_wallet")
        assert stats["losses"] == 1
        assert stats["wins"] == 0

    def test_activity_pattern_steady(self, test_db, profiler):
        """10+ bets with no big gaps should be 'steady'."""
        now = datetime.utcnow()
        test_db.upsert_trader("0xsteady_wallet", volume=60000.0, timestamp=now)

        # Create 12 resolved markets with daily bets (no gap > 30 days)
        for i in range(12):
            mid = f"m_steady_{i}"
            test_db.upsert_market(
                market_id=mid, question=f"Steady {i}?",
                resolved=True, outcome="Yes", category="crypto",
            )
            test_db.insert_bet(
                bet_id=f"steady_bet_{i}",
                wallet_address="0xsteady_wallet",
                market_id=mid, asset_id="a1",
                amount=100, price=0.5, side="BUY", outcome_bet="Yes",
                timestamp=now - timedelta(days=12 - i),
            )

        stats = profiler.calculate_wallet_stats("0xsteady_wallet")
        assert stats["activity_pattern"] == "steady"

    def test_activity_pattern_dormant_burst(self, test_db, profiler):
        """A gap of 30+ days between bets should be 'dormant_burst'."""
        now = datetime.utcnow()
        test_db.upsert_trader("0xdormant_wallet", volume=60000.0, timestamp=now)

        # Old bet
        test_db.upsert_market(
            market_id="m_dormant_old", question="Old?",
            resolved=True, outcome="Yes", category="crypto",
        )
        test_db.insert_bet(
            bet_id="dormant_old_bet",
            wallet_address="0xdormant_wallet",
            market_id="m_dormant_old", asset_id="a1",
            amount=100, price=0.5, side="BUY", outcome_bet="Yes",
            timestamp=now - timedelta(days=60),
        )

        # Recent bet (gap > 30 days)
        test_db.upsert_market(
            market_id="m_dormant_new", question="New?",
            resolved=True, outcome="Yes", category="crypto",
        )
        test_db.insert_bet(
            bet_id="dormant_new_bet",
            wallet_address="0xdormant_wallet",
            market_id="m_dormant_new", asset_id="a1",
            amount=100, price=0.5, side="BUY", outcome_bet="Yes",
            timestamp=now,
        )

        stats = profiler.calculate_wallet_stats("0xdormant_wallet")
        assert stats["activity_pattern"] == "dormant_burst"


class TestProfileWhale:
    """Tests for the profile_whale method."""

    def test_profile_whale_writes_to_db(self, test_db, profiler):
        """Verify profile_whale creates a whale_profiles record in the database."""
        now = datetime.utcnow()
        test_db.upsert_trader("0xprofile_whale", volume=100000.0, timestamp=now)

        # Create enough resolved markets for grading (25 markets)
        for i in range(25):
            mid = f"m_profile_{i}"
            # 18 wins, 7 losses -> 72% win rate -> grade A
            outcome = "Yes" if i < 18 else "No"
            test_db.upsert_market(
                market_id=mid, question=f"Profile test {i}?",
                resolved=True, outcome="Yes", category="crypto",
            )
            test_db.insert_bet(
                bet_id=f"profile_bet_{i}",
                wallet_address="0xprofile_whale",
                market_id=mid, asset_id="a1",
                amount=500, price=0.5, side="BUY",
                outcome_bet=outcome,
                timestamp=now - timedelta(days=25 - i),
            )

        result = profiler.profile_whale("0xprofile_whale")

        # Verify returned dict
        assert result["wallet_address"] == "0xprofile_whale"
        assert result["wins"] == 18
        assert result["losses"] == 7
        assert result["resolved_bets"] == 25
        assert result["grade"] == "A"
        assert result["total_volume"] == 100000.0

        # Verify database record was created
        db_profile = test_db.get_whale_profile("0xprofile_whale")
        assert db_profile is not None
        assert db_profile["wallet_address"] == "0xprofile_whale"
        assert db_profile["grade"] == "A"
        assert db_profile["total_resolved_bets"] == 25
        assert db_profile["total_volume"] == 100000.0
        assert abs(db_profile["win_rate"] - 72.0) < 0.1

    def test_profile_whale_ungraded_few_bets(self, test_db, profiler):
        """Wallet with fewer resolved bets than min gets 'ungraded'."""
        now = datetime.utcnow()
        test_db.upsert_trader("0xfew_bets", volume=60000.0, timestamp=now)

        # Only 5 resolved markets
        for i in range(5):
            mid = f"m_few_{i}"
            test_db.upsert_market(
                market_id=mid, question=f"Few {i}?",
                resolved=True, outcome="Yes", category="crypto",
            )
            test_db.insert_bet(
                bet_id=f"few_bet_{i}",
                wallet_address="0xfew_bets",
                market_id=mid, asset_id="a1",
                amount=100, price=0.5, side="BUY", outcome_bet="Yes",
                timestamp=now - timedelta(days=i),
            )

        result = profiler.profile_whale("0xfew_bets")
        assert result["grade"] == "ungraded"


class TestRefreshAllProfiles:
    """Tests for the refresh_all_profiles method."""

    def test_refresh_all_profiles(self, test_db, profiler):
        """Verify refresh_all_profiles processes all identified whales."""
        now = datetime.utcnow()

        # Create two whale-qualifying wallets
        test_db.upsert_trader("0xwhale_refresh_1", volume=80000.0, timestamp=now)
        test_db.upsert_trader("0xwhale_refresh_2", volume=60000.0, timestamp=now)
        # One non-whale
        test_db.upsert_trader("0xnon_whale", volume=1000.0, timestamp=now)

        # Add some resolved markets and bets for the whales
        for whale_idx, whale_addr in enumerate(["0xwhale_refresh_1", "0xwhale_refresh_2"]):
            for i in range(3):
                mid = f"m_refresh_{whale_idx}_{i}"
                test_db.upsert_market(
                    market_id=mid, question=f"Refresh {whale_idx} {i}?",
                    resolved=True, outcome="Yes", category="crypto",
                )
                test_db.insert_bet(
                    bet_id=f"refresh_bet_{whale_idx}_{i}",
                    wallet_address=whale_addr,
                    market_id=mid, asset_id="a1",
                    amount=100, price=0.5, side="BUY", outcome_bet="Yes",
                    timestamp=now - timedelta(days=i),
                )

        count = profiler.refresh_all_profiles()

        # Should have updated 2 whale profiles
        assert count == 2

        # Both should have profiles in the database
        p1 = test_db.get_whale_profile("0xwhale_refresh_1")
        p2 = test_db.get_whale_profile("0xwhale_refresh_2")
        assert p1 is not None
        assert p2 is not None

        # Non-whale should not have a profile
        p3 = test_db.get_whale_profile("0xnon_whale")
        assert p3 is None

    def test_refresh_continues_on_error(self, test_db, monkeypatch):
        """Verify refresh_all_profiles logs errors but continues."""
        now = datetime.utcnow()

        test_db.upsert_trader("0xerror_whale_1", volume=80000.0, timestamp=now)
        test_db.upsert_trader("0xerror_whale_2", volume=70000.0, timestamp=now)

        profiler = WhaleProfiler(database=test_db)

        # Make calculate_wallet_stats fail for the first wallet only
        original_calculate = profiler.calculate_wallet_stats
        call_count = {"n": 0}

        def patched_calculate(wallet_address):
            call_count["n"] += 1
            if wallet_address == "0xerror_whale_1":
                raise RuntimeError("Simulated failure")
            return original_calculate(wallet_address)

        monkeypatch.setattr(profiler, "calculate_wallet_stats", patched_calculate)

        count = profiler.refresh_all_profiles()

        # One failed, one succeeded
        assert count == 1
        # Both were attempted
        assert call_count["n"] == 2
