"""
CLI Dashboard for Polymarket Tracker Leaderboard.

This module provides an interactive terminal dashboard displaying:
- Top 20 traders by PNL
- Top 20 traders by Volume
- Trader details on demand
- Auto-refresh every 30 seconds

Usage:
    python -m polymarket_tracker.dashboard
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Optional

from .analytics import PolymarketAnalytics, LeaderboardMetric, analytics
from .export import LeaderboardExporter, exporter
from .database import db


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal styling."""
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"
    DIM = "\033[2m"


def supports_color() -> bool:
    """Check if the terminal supports color output."""
    if os.getenv("NO_COLOR"):
        return False
    if os.getenv("FORCE_COLOR"):
        return True
    if not hasattr(sys.stdout, "isatty"):
        return False
    if not sys.stdout.isatty():
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            return True
        except Exception:
            return False
    return True


# Disable colors if not supported
if not supports_color():
    for attr in dir(Colors):
        if not attr.startswith("_"):
            setattr(Colors, attr, "")


def clear_screen():
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def format_currency(value: float, include_sign: bool = True) -> str:
    """Format a value as currency with color based on sign."""
    if value >= 0:
        color = Colors.GREEN
        sign = "+" if include_sign else ""
    else:
        color = Colors.RED
        sign = ""

    return f"{color}{sign}${value:,.2f}{Colors.END}"


def format_percentage(value: float, include_sign: bool = True) -> str:
    """Format a value as percentage with color based on sign."""
    if value >= 0:
        color = Colors.GREEN
        sign = "+" if include_sign else ""
    else:
        color = Colors.RED
        sign = ""

    return f"{color}{sign}{value:.1f}%{Colors.END}"


def format_volume(value: float) -> str:
    """Format volume with abbreviations."""
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    elif value >= 1_000:
        return f"${value / 1_000:.2f}K"
    else:
        return f"${value:.2f}"


def truncate_address(address: str, length: int = 6) -> str:
    """Truncate wallet address for display."""
    if len(address) <= length * 2 + 3:
        return address
    return f"{address[:length]}...{address[-4:]}"


class Dashboard:
    """
    Interactive CLI dashboard for the Polymarket leaderboard.

    Displays trader rankings with auto-refresh capability.
    """

    def __init__(
        self,
        analytics_engine: Optional[PolymarketAnalytics] = None,
        export_engine: Optional[LeaderboardExporter] = None
    ):
        """
        Initialize dashboard.

        Args:
            analytics_engine: Analytics instance.
            export_engine: Export instance.
        """
        self.analytics = analytics_engine or analytics
        self.exporter = export_engine or exporter
        self.running = False

    def print_header(self):
        """Print dashboard header."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}           POLYMARKET TRADER LEADERBOARD{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.END}")
        print(f"{Colors.DIM}Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC{Colors.END}")

    def print_summary(self):
        """Print summary statistics."""
        try:
            summary = self.analytics.get_leaderboard_summary()
            stats = db.get_stats()

            print(f"\n{Colors.BOLD}Database Summary:{Colors.END}")
            print(f"  Traders: {Colors.YELLOW}{summary['total_traders']:,}{Colors.END}")
            print(f"  Markets: {Colors.YELLOW}{stats['total_markets']:,}{Colors.END} ({stats['active_markets']:,} active)")
            print(f"  Total Trades: {Colors.YELLOW}{summary['total_trades']:,}{Colors.END}")
            print(f"  Total Volume: {Colors.YELLOW}{format_volume(summary['total_volume'])}{Colors.END}")
            print(f"  Avg Trade Size: {Colors.YELLOW}{format_volume(summary['avg_trade_size'])}{Colors.END}")
        except Exception as e:
            print(f"{Colors.RED}Error loading summary: {e}{Colors.END}")

    def print_leaderboard(
        self,
        metric: LeaderboardMetric,
        limit: int = 20,
        title: Optional[str] = None
    ):
        """
        Print a leaderboard table.

        Args:
            metric: Ranking metric.
            limit: Number of entries to show.
            title: Custom title (uses metric name if not provided).
        """
        title = title or f"Top {limit} Traders by {metric.value.upper()}"
        print(f"\n{Colors.BOLD}{Colors.BLUE}{title}{Colors.END}")
        print(f"{Colors.DIM}{'-' * 80}{Colors.END}")

        try:
            entries = self.analytics.get_top_traders(metric=metric, limit=limit)

            if not entries:
                print(f"{Colors.YELLOW}No traders found. Run data collection first.{Colors.END}")
                return

            # Print table header
            print(f"{Colors.BOLD}{'Rank':<5} {'Address':<14} {'Total PNL':>14} {'Volume':>12} "
                  f"{'Trades':>8} {'Win%':>8} {'ROI':>10}{Colors.END}")
            print(f"{Colors.DIM}{'-' * 80}{Colors.END}")

            # Print entries
            for entry in entries:
                rank_str = f"#{entry.rank}"
                address = entry.display_address
                pnl = format_currency(entry.total_pnl)
                volume = format_volume(entry.total_volume)
                trades = str(entry.total_trades)
                win_rate = f"{entry.win_rate:.1f}%"
                roi = format_percentage(entry.roi)

                print(f"{rank_str:<5} {address:<14} {pnl:>22} {volume:>12} "
                      f"{trades:>8} {win_rate:>8} {roi:>18}")

        except Exception as e:
            print(f"{Colors.RED}Error loading leaderboard: {e}{Colors.END}")

    def print_trader_details(self, wallet_address: str):
        """
        Print detailed statistics for a specific trader.

        Args:
            wallet_address: Trader's wallet address.
        """
        print(f"\n{Colors.BOLD}{Colors.BLUE}Trader Details: {truncate_address(wallet_address)}{Colors.END}")
        print(f"{Colors.DIM}{'-' * 60}{Colors.END}")

        try:
            stats = self.analytics.get_trader_stats(wallet_address)

            print(f"\n{Colors.BOLD}Address:{Colors.END} {wallet_address}")
            print(f"\n{Colors.BOLD}Trading Activity:{Colors.END}")
            print(f"  Total Trades:        {Colors.YELLOW}{stats.total_trades:,}{Colors.END}")
            print(f"  Markets Participated: {Colors.YELLOW}{stats.markets_participated:,}{Colors.END}")
            print(f"  Average Bet Size:    {Colors.YELLOW}{format_volume(stats.avg_bet_size)}{Colors.END}")
            print(f"  Average Price:       {Colors.YELLOW}${stats.avg_price:.4f}{Colors.END}")

            print(f"\n{Colors.BOLD}Volume Breakdown:{Colors.END}")
            print(f"  Total Volume:  {Colors.YELLOW}{format_volume(stats.total_volume)}{Colors.END}")
            print(f"  Buy Volume:    {Colors.YELLOW}{format_volume(stats.buy_volume)}{Colors.END} ({stats.buy_count} trades)")
            print(f"  Sell Volume:   {Colors.YELLOW}{format_volume(stats.sell_volume)}{Colors.END} ({stats.sell_count} trades)")

            if stats.pnl:
                print(f"\n{Colors.BOLD}Profit & Loss:{Colors.END}")
                print(f"  Total PNL:      {format_currency(float(stats.pnl.total_pnl))}")
                print(f"  Realized PNL:   {format_currency(float(stats.pnl.realized_pnl))}")
                print(f"  Unrealized PNL: {format_currency(float(stats.pnl.unrealized_pnl))}")
                print(f"  Fees Paid:      {Colors.RED}-${float(stats.pnl.total_fees_paid):,.2f}{Colors.END}")

                print(f"\n{Colors.BOLD}Performance:{Colors.END}")
                print(f"  Win Rate:       {format_percentage(stats.pnl.win_rate, include_sign=False)}")
                print(f"  ROI:            {format_percentage(stats.pnl.roi)}")
                print(f"  Winning Trades: {Colors.GREEN}{stats.pnl.winning_trades}{Colors.END}")
                print(f"  Losing Trades:  {Colors.RED}{stats.pnl.losing_trades}{Colors.END}")

            print(f"\n{Colors.BOLD}Activity:{Colors.END}")
            if stats.first_trade:
                print(f"  First Trade: {Colors.DIM}{stats.first_trade.strftime('%Y-%m-%d %H:%M')}{Colors.END}")
            if stats.last_trade:
                print(f"  Last Trade:  {Colors.DIM}{stats.last_trade.strftime('%Y-%m-%d %H:%M')}{Colors.END}")

        except Exception as e:
            print(f"{Colors.RED}Error loading trader details: {e}{Colors.END}")

    def display(self, show_pnl: bool = True, show_volume: bool = True, limit: int = 20):
        """
        Display the full dashboard.

        Args:
            show_pnl: Show PNL leaderboard.
            show_volume: Show Volume leaderboard.
            limit: Number of entries per leaderboard.
        """
        clear_screen()
        self.print_header()
        self.print_summary()

        if show_pnl:
            self.print_leaderboard(LeaderboardMetric.PNL, limit, f"Top {limit} by PNL")

        if show_volume:
            self.print_leaderboard(LeaderboardMetric.VOLUME, limit, f"Top {limit} by Volume")

        print(f"\n{Colors.DIM}Press Ctrl+C to exit | 'e' to export | 't' for trader lookup{Colors.END}")

    def run_interactive(
        self,
        refresh_interval: int = 30,
        limit: int = 20
    ):
        """
        Run the dashboard in interactive mode with auto-refresh.

        Args:
            refresh_interval: Seconds between refreshes.
            limit: Number of entries per leaderboard.
        """
        self.running = True
        print(f"{Colors.CYAN}Starting dashboard with {refresh_interval}s refresh...{Colors.END}")

        try:
            while self.running:
                self.display(limit=limit)

                # Wait for refresh or user input
                print(f"\n{Colors.DIM}Refreshing in {refresh_interval} seconds...{Colors.END}")

                start_time = time.time()
                while time.time() - start_time < refresh_interval:
                    time.sleep(1)

        except KeyboardInterrupt:
            self.running = False
            print(f"\n{Colors.YELLOW}Dashboard stopped.{Colors.END}")

    def export_current(self, output_dir: str = "exports"):
        """
        Export current leaderboard data.

        Args:
            output_dir: Output directory for exports.
        """
        print(f"\n{Colors.CYAN}Exporting leaderboards...{Colors.END}")
        try:
            files = self.exporter.export_combined_leaderboards(output_dir)
            print(f"{Colors.GREEN}Exported to:{Colors.END}")
            for name, path in files.items():
                print(f"  {name}: {path}")
        except Exception as e:
            print(f"{Colors.RED}Export failed: {e}{Colors.END}")


def main():
    """Main entry point for the dashboard."""
    parser = argparse.ArgumentParser(
        description="Polymarket Trader Leaderboard Dashboard"
    )
    parser.add_argument(
        "--refresh", "-r",
        type=int,
        default=30,
        help="Refresh interval in seconds (default: 30)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=20,
        help="Number of traders to show (default: 20)"
    )
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Display once without auto-refresh"
    )
    parser.add_argument(
        "--pnl-only",
        action="store_true",
        help="Show only PNL leaderboard"
    )
    parser.add_argument(
        "--volume-only",
        action="store_true",
        help="Show only Volume leaderboard"
    )
    parser.add_argument(
        "--trader", "-t",
        type=str,
        help="Show details for specific wallet address"
    )
    parser.add_argument(
        "--export", "-e",
        type=str,
        nargs="?",
        const="exports",
        help="Export leaderboards to directory (default: exports)"
    )
    parser.add_argument(
        "--metric", "-m",
        type=str,
        choices=["pnl", "volume", "trade_count", "win_rate", "roi"],
        default="pnl",
        help="Sort leaderboard by metric (default: pnl)"
    )

    args = parser.parse_args()

    dashboard = Dashboard()

    # Handle export
    if args.export:
        dashboard.export_current(args.export)
        return

    # Handle trader lookup
    if args.trader:
        dashboard.print_header()
        dashboard.print_trader_details(args.trader)
        return

    # Handle single metric display
    if args.pnl_only or args.volume_only:
        dashboard.print_header()
        dashboard.print_summary()
        if args.pnl_only:
            dashboard.print_leaderboard(LeaderboardMetric.PNL, args.limit)
        if args.volume_only:
            dashboard.print_leaderboard(LeaderboardMetric.VOLUME, args.limit)
        return

    # Handle custom metric
    if args.metric != "pnl":
        metric = LeaderboardMetric(args.metric)
        dashboard.print_header()
        dashboard.print_summary()
        dashboard.print_leaderboard(metric, args.limit)
        return

    # Run dashboard
    if args.no_refresh:
        dashboard.display(limit=args.limit)
    else:
        dashboard.run_interactive(
            refresh_interval=args.refresh,
            limit=args.limit
        )


if __name__ == "__main__":
    main()
