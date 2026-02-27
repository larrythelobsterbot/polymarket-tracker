"""
Export functionality for Polymarket Tracker.

This module provides functions to export leaderboard and analytics
data to various formats:
- JSON: Full data export with metadata
- CSV: Tabular format for spreadsheets
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .analytics import (
    PolymarketAnalytics,
    LeaderboardEntry,
    LeaderboardMetric,
    TraderStats,
    analytics,
)

logger = logging.getLogger(__name__)


class LeaderboardExporter:
    """
    Export leaderboard data to various formats.

    Supports JSON and CSV exports with configurable options.
    """

    def __init__(self, analytics_engine: Optional[PolymarketAnalytics] = None):
        """
        Initialize exporter.

        Args:
            analytics_engine: Analytics instance (uses global if not provided).
        """
        self.analytics = analytics_engine or analytics

    def export_leaderboard_json(
        self,
        filepath: str,
        metric: LeaderboardMetric = LeaderboardMetric.PNL,
        limit: int = 20,
        include_summary: bool = True,
        pretty: bool = True
    ) -> str:
        """
        Export leaderboard to JSON file.

        Args:
            filepath: Output file path.
            metric: Ranking metric.
            limit: Number of traders to include.
            include_summary: Include aggregate statistics.
            pretty: Pretty-print JSON output.

        Returns:
            Path to created file.
        """
        # Get leaderboard data
        entries = self.analytics.get_top_traders(metric=metric, limit=limit)

        # Build export structure
        export_data = {
            "metadata": {
                "exported_at": datetime.utcnow().isoformat(),
                "metric": metric.value,
                "limit": limit,
                "total_entries": len(entries),
            },
            "leaderboard": [entry.to_dict() for entry in entries],
        }

        if include_summary:
            export_data["summary"] = self.analytics.get_leaderboard_summary()

        # Write to file
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            if pretty:
                json.dump(export_data, f, indent=2, default=str)
            else:
                json.dump(export_data, f, default=str)

        logger.info(f"Exported leaderboard to {filepath}")
        return str(filepath)

    def export_leaderboard_csv(
        self,
        filepath: str,
        metric: LeaderboardMetric = LeaderboardMetric.PNL,
        limit: int = 20
    ) -> str:
        """
        Export leaderboard to CSV file.

        Args:
            filepath: Output file path.
            metric: Ranking metric.
            limit: Number of traders to include.

        Returns:
            Path to created file.
        """
        # Get leaderboard data
        entries = self.analytics.get_top_traders(metric=metric, limit=limit)

        # Define columns
        fieldnames = [
            "rank",
            "wallet_address",
            "total_pnl",
            "realized_pnl",
            "unrealized_pnl",
            "total_volume",
            "total_trades",
            "win_rate",
            "roi",
            "markets_participated",
            "avg_bet_size",
            "last_active",
        ]

        # Write to file
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for entry in entries:
                row = entry.to_dict()
                # Remove display_address for CSV (redundant with wallet_address)
                row.pop("display_address", None)
                writer.writerow(row)

        logger.info(f"Exported leaderboard to {filepath}")
        return str(filepath)

    def export_trader_report_json(
        self,
        wallet_address: str,
        filepath: str,
        pretty: bool = True
    ) -> str:
        """
        Export detailed report for a single trader.

        Args:
            wallet_address: Trader's wallet address.
            filepath: Output file path.
            pretty: Pretty-print JSON output.

        Returns:
            Path to created file.
        """
        stats = self.analytics.get_trader_stats(wallet_address)

        export_data = {
            "metadata": {
                "exported_at": datetime.utcnow().isoformat(),
                "wallet_address": wallet_address,
            },
            "trader": stats.to_dict(),
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            if pretty:
                json.dump(export_data, f, indent=2, default=str)
            else:
                json.dump(export_data, f, default=str)

        logger.info(f"Exported trader report to {filepath}")
        return str(filepath)

    def export_all_traders_csv(
        self,
        filepath: str,
        min_trades: int = 1
    ) -> str:
        """
        Export all traders to CSV file.

        Args:
            filepath: Output file path.
            min_trades: Minimum trades for inclusion.

        Returns:
            Path to created file.
        """
        # Get all traders (using large limit)
        entries = self.analytics.get_top_traders(
            metric=LeaderboardMetric.VOLUME,
            limit=10000,
            min_trades=min_trades
        )

        fieldnames = [
            "wallet_address",
            "total_pnl",
            "realized_pnl",
            "unrealized_pnl",
            "total_volume",
            "total_trades",
            "win_rate",
            "roi",
            "markets_participated",
            "avg_bet_size",
            "last_active",
        ]

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for entry in entries:
                row = entry.to_dict()
                row.pop("rank", None)
                row.pop("display_address", None)
                writer.writerow(row)

        logger.info(f"Exported {len(entries)} traders to {filepath}")
        return str(filepath)

    def export_combined_leaderboards(
        self,
        output_dir: str,
        limit: int = 20
    ) -> dict[str, str]:
        """
        Export leaderboards for all metrics.

        Args:
            output_dir: Output directory.
            limit: Number of traders per leaderboard.

        Returns:
            Dictionary mapping metric to filepath.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        files = {}

        for metric in LeaderboardMetric:
            # JSON export
            json_path = output_dir / f"leaderboard_{metric.value}_{timestamp}.json"
            self.export_leaderboard_json(str(json_path), metric=metric, limit=limit)
            files[f"{metric.value}_json"] = str(json_path)

            # CSV export
            csv_path = output_dir / f"leaderboard_{metric.value}_{timestamp}.csv"
            self.export_leaderboard_csv(str(csv_path), metric=metric, limit=limit)
            files[f"{metric.value}_csv"] = str(csv_path)

        logger.info(f"Exported all leaderboards to {output_dir}")
        return files


# Global exporter instance
exporter = LeaderboardExporter()
