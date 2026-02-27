"""
Tests for the export module.
"""

import csv
import json
import os
import tempfile
from datetime import datetime

import pytest

from polymarket_tracker.database import Database
from polymarket_tracker.analytics import PolymarketAnalytics, LeaderboardMetric
from polymarket_tracker.export import LeaderboardExporter


@pytest.fixture
def test_db():
    """Create a temporary test database with sample data."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = Database(db_path=path)

    # Add sample data
    for i in range(5):
        db.upsert_trader(
            wallet_address=f"0xtrader{i}",
            volume=(i + 1) * 1000,
            timestamp=datetime.utcnow()
        )
        db.upsert_market(
            market_id=f"market_{i}",
            question=f"Test Market {i}?",
            volume=(i + 1) * 10000
        )
        db.insert_bet(
            bet_id=f"bet_{i}",
            wallet_address=f"0xtrader{i}",
            market_id=f"market_{i}",
            asset_id=f"token_{i}",
            amount=100 * (i + 1),
            price=0.5,
            side="BUY",
            outcome_bet="Yes",
            timestamp=datetime.utcnow()
        )

    yield db
    os.unlink(path)


@pytest.fixture
def exporter(test_db):
    """Create exporter with test database."""
    analytics = PolymarketAnalytics(database=test_db)
    return LeaderboardExporter(analytics_engine=analytics)


class TestLeaderboardExporter:
    """Tests for LeaderboardExporter class."""

    def test_export_json(self, exporter):
        """Test JSON export."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            result = exporter.export_leaderboard_json(filepath, limit=5)
            assert os.path.exists(result)

            with open(result) as f:
                data = json.load(f)

            assert "metadata" in data
            assert "leaderboard" in data
            assert "summary" in data
            assert len(data["leaderboard"]) <= 5
            assert data["metadata"]["metric"] == "pnl"
        finally:
            os.unlink(filepath)

    def test_export_json_without_summary(self, exporter):
        """Test JSON export without summary."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            exporter.export_leaderboard_json(filepath, include_summary=False)

            with open(filepath) as f:
                data = json.load(f)

            assert "summary" not in data
        finally:
            os.unlink(filepath)

    def test_export_csv(self, exporter):
        """Test CSV export."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            filepath = f.name

        try:
            result = exporter.export_leaderboard_csv(filepath, limit=5)
            assert os.path.exists(result)

            with open(result) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) <= 5
            assert "rank" in rows[0]
            assert "wallet_address" in rows[0]
            assert "total_pnl" in rows[0]
            assert "total_volume" in rows[0]
        finally:
            os.unlink(filepath)

    def test_export_different_metrics(self, exporter):
        """Test export with different metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Export by volume
            filepath = os.path.join(tmpdir, "volume.json")
            exporter.export_leaderboard_json(
                filepath,
                metric=LeaderboardMetric.VOLUME
            )

            with open(filepath) as f:
                data = json.load(f)

            assert data["metadata"]["metric"] == "volume"

    def test_export_trader_report(self, exporter):
        """Test individual trader report export."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            result = exporter.export_trader_report_json(
                "0xtrader0",
                filepath
            )
            assert os.path.exists(result)

            with open(result) as f:
                data = json.load(f)

            assert "metadata" in data
            assert "trader" in data
            assert data["metadata"]["wallet_address"] == "0xtrader0"
        finally:
            os.unlink(filepath)

    def test_export_combined_leaderboards(self, exporter):
        """Test exporting all leaderboard types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = exporter.export_combined_leaderboards(tmpdir, limit=3)

            # Should have both JSON and CSV for each metric
            assert len(files) > 0
            for filepath in files.values():
                assert os.path.exists(filepath)

    def test_export_creates_directory(self, exporter):
        """Test that export creates directory if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "nested", "dir")
            filepath = os.path.join(subdir, "export.json")

            result = exporter.export_leaderboard_json(filepath)
            assert os.path.exists(result)
