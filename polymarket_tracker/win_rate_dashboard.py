"""
Win Rate Dashboard for Polymarket Tracker.

This module provides an interactive CLI dashboard for win rate analysis:
- Top traders by win rate with minimum bet requirements
- Streak leaderboards
- Category and size breakdowns
- Statistical significance indicators
- Auto-refresh capability

Usage:
    python -m polymarket_tracker.win_rate_dashboard
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Optional

from .win_rate import (
    WinRateAnalyzer,
    TraderWinRateAnalysis,
    BetSizeCategory,
    win_rate_analyzer,
)
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
    MAGENTA = "\033[35m"
    WHITE = "\033[97m"


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


def format_percentage(value: float, show_color: bool = True) -> str:
    """Format percentage with optional color based on value."""
    if not show_color:
        return f"{value:.1f}%"

    if value >= 60:
        color = Colors.GREEN
    elif value >= 50:
        color = Colors.YELLOW
    else:
        color = Colors.RED

    return f"{color}{value:.1f}%{Colors.END}"


def format_streak(streak: int, streak_type: str) -> str:
    """Format streak with color."""
    if streak_type == "winning":
        return f"{Colors.GREEN}W{streak}{Colors.END}"
    elif streak_type == "losing":
        return f"{Colors.RED}L{streak}{Colors.END}"
    else:
        return f"{Colors.DIM}-{Colors.END}"


def format_confidence_interval(lower: float, upper: float) -> str:
    """Format confidence interval."""
    return f"{Colors.DIM}[{lower:.1f}% - {upper:.1f}%]{Colors.END}"


def format_significance(has_sufficient_data: bool) -> str:
    """Format statistical significance indicator."""
    if has_sufficient_data:
        return f"{Colors.GREEN}*{Colors.END}"
    else:
        return f"{Colors.YELLOW}~{Colors.END}"


class WinRateDashboard:
    """
    Interactive CLI dashboard for win rate analysis.

    Displays:
    - Win rate leaderboard with minimum resolved bets filter
    - Streak information
    - Category breakdowns
    - Statistical significance indicators
    """

    def __init__(self, analyzer: Optional[WinRateAnalyzer] = None):
        """
        Initialize dashboard.

        Args:
            analyzer: WinRateAnalyzer instance.
        """
        self.analyzer = analyzer or win_rate_analyzer
        self.running = False

    def print_header(self):
        """Print dashboard header."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 90}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}              POLYMARKET WIN RATE LEADERBOARD{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 90}{Colors.END}")
        print(f"{Colors.DIM}Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC{Colors.END}")
        print(f"{Colors.DIM}Legend: {Colors.GREEN}*{Colors.END}{Colors.DIM} = Statistically significant (10+ bets) | "
              f"{Colors.YELLOW}~{Colors.END}{Colors.DIM} = Insufficient data{Colors.END}")

    def print_summary(self):
        """Print summary statistics."""
        try:
            stats = db.get_stats()
            print(f"\n{Colors.BOLD}Database Summary:{Colors.END}")
            print(f"  Traders: {Colors.YELLOW}{stats['total_traders']:,}{Colors.END}")
            print(f"  Markets: {Colors.YELLOW}{stats['total_markets']:,}{Colors.END} ({stats['active_markets']:,} active)")
            print(f"  Total Bets: {Colors.YELLOW}{stats['total_bets']:,}{Colors.END}")
        except Exception as e:
            print(f"{Colors.RED}Error loading summary: {e}{Colors.END}")

    def print_win_rate_leaderboard(
        self,
        limit: int = 20,
        min_bets: int = 20,
        sort_by: str = "win_rate",
        title: Optional[str] = None
    ):
        """
        Print win rate leaderboard.

        Args:
            limit: Number of traders to show.
            min_bets: Minimum resolved bets for inclusion.
            sort_by: Sort metric.
            title: Custom title.
        """
        title = title or f"Top {limit} by Win Rate (min {min_bets} resolved bets)"
        print(f"\n{Colors.BOLD}{Colors.BLUE}{title}{Colors.END}")
        print(f"{Colors.DIM}{'-' * 90}{Colors.END}")

        try:
            entries = self.analyzer.get_win_rate_leaderboard(
                limit=limit,
                min_resolved_bets=min_bets,
                sort_by=sort_by
            )

            if not entries:
                print(f"{Colors.YELLOW}No traders found with {min_bets}+ resolved bets.{Colors.END}")
                print(f"{Colors.DIM}Try lowering the minimum with --min-bets{Colors.END}")
                return

            # Print header
            print(f"{Colors.BOLD}{'Rank':<5} {'Sig':<4} {'Address':<14} {'Win Rate':>10} "
                  f"{'W/L':>8} {'Conf Int':>18} {'Streak':>8} {'Risk-Adj':>10}{Colors.END}")
            print(f"{Colors.DIM}{'-' * 90}{Colors.END}")

            for entry in entries:
                rank_str = f"#{entry['rank']}"
                sig = format_significance(entry['has_sufficient_data'])
                address = entry['display_address']
                win_rate = format_percentage(entry['win_rate'])
                wl = f"{entry['wins']}/{entry['losses']}"
                conf_int = format_confidence_interval(entry['confidence_lower'], entry['confidence_upper'])
                streak = format_streak(entry['current_streak'], entry['current_streak_type'])
                risk_adj = f"{entry['risk_adjusted_return']:.2f}"

                print(f"{rank_str:<5} {sig:<4} {address:<14} {win_rate:>18} "
                      f"{wl:>8} {conf_int:>26} {streak:>8} {risk_adj:>10}")

        except Exception as e:
            print(f"{Colors.RED}Error loading leaderboard: {e}{Colors.END}")
            import traceback
            traceback.print_exc()

    def print_streak_leaderboard(self, limit: int = 10):
        """Print top traders by winning streak."""
        print(f"\n{Colors.BOLD}{Colors.MAGENTA}Top {limit} Longest Winning Streaks{Colors.END}")
        print(f"{Colors.DIM}{'-' * 60}{Colors.END}")

        try:
            entries = self.analyzer.get_win_rate_leaderboard(
                limit=limit,
                min_resolved_bets=10,
                sort_by="streak"
            )

            if not entries:
                print(f"{Colors.YELLOW}No traders with sufficient data.{Colors.END}")
                return

            print(f"{Colors.BOLD}{'Rank':<5} {'Address':<14} {'Best Streak':>12} "
                  f"{'Current':>10} {'Win Rate':>10}{Colors.END}")
            print(f"{Colors.DIM}{'-' * 60}{Colors.END}")

            for i, entry in enumerate(entries, 1):
                address = entry['display_address']
                best_streak = f"{Colors.GREEN}W{entry['longest_winning_streak']}{Colors.END}"
                current = format_streak(entry['current_streak'], entry['current_streak_type'])
                win_rate = format_percentage(entry['win_rate'])

                print(f"#{i:<4} {address:<14} {best_streak:>20} {current:>18} {win_rate:>18}")

        except Exception as e:
            print(f"{Colors.RED}Error loading streak leaderboard: {e}{Colors.END}")

    def print_trader_win_rate_details(self, wallet_address: str):
        """
        Print detailed win rate analysis for a trader.

        Args:
            wallet_address: Trader's wallet address.
        """
        print(f"\n{Colors.BOLD}{Colors.BLUE}Win Rate Analysis: {wallet_address[:10]}...{Colors.END}")
        print(f"{Colors.DIM}{'=' * 70}{Colors.END}")

        try:
            analysis = self.analyzer.analyze_trader_win_rate(wallet_address)

            # Overall stats
            print(f"\n{Colors.BOLD}Overall Performance:{Colors.END}")
            print(f"  Resolved Bets:    {Colors.YELLOW}{analysis.total_resolved_bets}{Colors.END}")
            print(f"  Wins:             {Colors.GREEN}{analysis.total_wins}{Colors.END}")
            print(f"  Losses:           {Colors.RED}{analysis.total_losses}{Colors.END}")
            print(f"  Ties/Cancelled:   {Colors.DIM}{analysis.total_ties + analysis.total_cancelled}{Colors.END}")
            print(f"  Win Rate:         {format_percentage(analysis.overall_win_rate)}")

            # Significance
            if analysis.has_sufficient_data:
                print(f"  Statistical Sig:  {Colors.GREEN}Yes (sufficient data){Colors.END}")
            else:
                print(f"  Statistical Sig:  {Colors.YELLOW}No (<{analysis.min_bets_for_significance} bets){Colors.END}")

            if analysis.confidence_interval:
                ci = analysis.confidence_interval
                print(f"  95% Confidence:   [{ci.lower_bound:.1f}% - {ci.upper_bound:.1f}%]")

            # Streaks
            print(f"\n{Colors.BOLD}Streaks:{Colors.END}")
            print(f"  Current:          {format_streak(analysis.streaks.current_streak, analysis.streaks.current_streak_type)}")
            print(f"  Best Winning:     {Colors.GREEN}W{analysis.streaks.longest_winning_streak}{Colors.END}")
            print(f"  Worst Losing:     {Colors.RED}L{analysis.streaks.longest_losing_streak}{Colors.END}")

            # By Category
            if analysis.by_category:
                print(f"\n{Colors.BOLD}Win Rate by Category:{Colors.END}")
                print(f"  {'Category':<20} {'Win Rate':>10} {'W/L':>10} {'Volume':>12}")
                print(f"  {'-' * 55}")
                for cat, data in sorted(analysis.by_category.items(), key=lambda x: x[1].win_rate, reverse=True):
                    if data.wins + data.losses > 0:
                        wr = format_percentage(data.win_rate, show_color=False)
                        wl = f"{data.wins}/{data.losses}"
                        vol = f"${data.total_volume:,.0f}"
                        print(f"  {cat:<20} {wr:>10} {wl:>10} {vol:>12}")

            # By Bet Size
            print(f"\n{Colors.BOLD}Win Rate by Bet Size:{Colors.END}")
            print(f"  {'Size':<12} {'Win Rate':>10} {'W/L':>10} {'Avg Bet':>12} {'PNL':>12}")
            print(f"  {'-' * 60}")
            for size_cat in [BetSizeCategory.SMALL, BetSizeCategory.MEDIUM, BetSizeCategory.LARGE]:
                if size_cat in analysis.by_bet_size:
                    data = analysis.by_bet_size[size_cat]
                    if data.wins + data.losses > 0:
                        size_name = size_cat.value.capitalize()
                        wr = format_percentage(data.win_rate, show_color=False)
                        wl = f"{data.wins}/{data.losses}"
                        avg = f"${data.avg_bet_size:,.0f}"
                        pnl = f"${data.total_pnl:,.0f}" if data.total_pnl >= 0 else f"-${abs(data.total_pnl):,.0f}"
                        print(f"  {size_name:<12} {wr:>10} {wl:>10} {avg:>12} {pnl:>12}")

            # Kelly Criterion
            if analysis.kelly_analysis:
                kelly = analysis.kelly_analysis
                print(f"\n{Colors.BOLD}Kelly Criterion Analysis:{Colors.END}")
                print(f"  Optimal Bet Size: {kelly.optimal_fraction:.1f}% of bankroll")
                print(f"  Your Avg Size:    {kelly.actual_avg_fraction:.1f}% of volume")
                print(f"  Kelly Multiple:   {kelly.kelly_multiple:.2f}x")
                if kelly.is_overbetting:
                    print(f"  Status:           {Colors.RED}Overbetting{Colors.END}")
                else:
                    print(f"  Status:           {Colors.GREEN}Conservative/Optimal{Colors.END}")
                print(f"  Recommendation:   {Colors.DIM}{kelly.recommendation}{Colors.END}")

            # Risk Metrics
            print(f"\n{Colors.BOLD}Risk Metrics:{Colors.END}")
            if analysis.sharpe_ratio is not None:
                sharpe_color = Colors.GREEN if analysis.sharpe_ratio > 1 else (Colors.YELLOW if analysis.sharpe_ratio > 0 else Colors.RED)
                print(f"  Sharpe Ratio:     {sharpe_color}{analysis.sharpe_ratio:.3f}{Colors.END}")
            print(f"  Max Drawdown:     {Colors.RED}{analysis.max_drawdown:.1f}%{Colors.END}")
            print(f"  Risk-Adj Return:  {analysis.risk_adjusted_return:.3f}")

        except Exception as e:
            print(f"{Colors.RED}Error loading analysis: {e}{Colors.END}")
            import traceback
            traceback.print_exc()

    def display(
        self,
        limit: int = 20,
        min_bets: int = 20,
        sort_by: str = "win_rate",
        show_streaks: bool = True
    ):
        """
        Display the full dashboard.

        Args:
            limit: Number of traders per leaderboard.
            min_bets: Minimum resolved bets.
            sort_by: Sort metric.
            show_streaks: Whether to show streak leaderboard.
        """
        clear_screen()
        self.print_header()
        self.print_summary()
        self.print_win_rate_leaderboard(limit=limit, min_bets=min_bets, sort_by=sort_by)

        if show_streaks:
            self.print_streak_leaderboard(limit=10)

        print(f"\n{Colors.DIM}Press Ctrl+C to exit{Colors.END}")

    def run_interactive(self, refresh_interval: int = 30, limit: int = 20, min_bets: int = 20):
        """
        Run dashboard with auto-refresh.

        Args:
            refresh_interval: Seconds between refreshes.
            limit: Traders per leaderboard.
            min_bets: Minimum resolved bets.
        """
        self.running = True
        print(f"{Colors.CYAN}Starting win rate dashboard with {refresh_interval}s refresh...{Colors.END}")

        try:
            while self.running:
                self.display(limit=limit, min_bets=min_bets)
                print(f"\n{Colors.DIM}Refreshing in {refresh_interval} seconds...{Colors.END}")

                start_time = time.time()
                while time.time() - start_time < refresh_interval:
                    time.sleep(1)

        except KeyboardInterrupt:
            self.running = False
            print(f"\n{Colors.YELLOW}Dashboard stopped.{Colors.END}")


def main():
    """Main entry point for the win rate dashboard."""
    parser = argparse.ArgumentParser(
        description="Polymarket Win Rate Leaderboard Dashboard"
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
        "--min-bets", "-m",
        type=int,
        default=20,
        help="Minimum resolved bets for inclusion (default: 20)"
    )
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Display once without auto-refresh"
    )
    parser.add_argument(
        "--sort", "-s",
        type=str,
        choices=["win_rate", "risk_adjusted", "sharpe", "streak"],
        default="win_rate",
        help="Sort metric (default: win_rate)"
    )
    parser.add_argument(
        "--trader", "-t",
        type=str,
        help="Show detailed analysis for specific wallet address"
    )
    parser.add_argument(
        "--no-streaks",
        action="store_true",
        help="Hide streak leaderboard"
    )

    args = parser.parse_args()
    dashboard = WinRateDashboard()

    # Handle trader lookup
    if args.trader:
        dashboard.print_header()
        dashboard.print_trader_win_rate_details(args.trader)
        return

    # Run dashboard
    if args.no_refresh:
        dashboard.display(
            limit=args.limit,
            min_bets=args.min_bets,
            sort_by=args.sort,
            show_streaks=not args.no_streaks
        )
    else:
        dashboard.run_interactive(
            refresh_interval=args.refresh,
            limit=args.limit,
            min_bets=args.min_bets
        )


if __name__ == "__main__":
    main()
