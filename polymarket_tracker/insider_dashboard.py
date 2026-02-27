"""
Insider Detection Dashboard for Polymarket Tracker.

This module provides an interactive CLI dashboard for monitoring
suspicious betting activity:
- Real-time suspicious bets feed
- Trader risk profiles
- Market manipulation alerts
- Correlated betting visualization

Usage:
    python -m polymarket_tracker.insider_dashboard
"""

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Optional

from .insider_detection import (
    InsiderDetector,
    DetectionConfig,
    SuspiciousBet,
    WalletRiskProfile,
    AlertSeverity,
    AlertType,
    insider_detector,
)
from .alerts import AlertManager, AlertConfig
from .database import db


# ANSI color codes
class Colors:
    """ANSI color codes for terminal styling."""
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[35m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"
    DIM = "\033[2m"
    BG_RED = "\033[41m"
    BG_YELLOW = "\033[43m"
    WHITE = "\033[97m"


def supports_color() -> bool:
    """Check if terminal supports color."""
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


if not supports_color():
    for attr in dir(Colors):
        if not attr.startswith("_"):
            setattr(Colors, attr, "")


def clear_screen():
    """Clear terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def format_severity(severity: AlertSeverity) -> str:
    """Format severity with color."""
    colors = {
        AlertSeverity.LOW: Colors.BLUE,
        AlertSeverity.MEDIUM: Colors.YELLOW,
        AlertSeverity.HIGH: Colors.RED,
        AlertSeverity.CRITICAL: f"{Colors.BG_RED}{Colors.WHITE}",
    }
    color = colors.get(severity, Colors.DIM)
    return f"{color}{severity.value.upper():>8}{Colors.END}"


def format_score(score: float) -> str:
    """Format anomaly score with color."""
    if score >= 90:
        return f"{Colors.BG_RED}{Colors.WHITE}{score:5.1f}{Colors.END}"
    elif score >= 80:
        return f"{Colors.RED}{score:5.1f}{Colors.END}"
    elif score >= 60:
        return f"{Colors.YELLOW}{score:5.1f}{Colors.END}"
    elif score >= 40:
        return f"{Colors.CYAN}{score:5.1f}{Colors.END}"
    else:
        return f"{Colors.DIM}{score:5.1f}{Colors.END}"


def format_amount(amount: float) -> str:
    """Format dollar amount."""
    if amount >= 1000000:
        return f"${amount/1000000:.1f}M"
    elif amount >= 1000:
        return f"${amount/1000:.1f}K"
    else:
        return f"${amount:.0f}"


def truncate(text: str, length: int = 30) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= length:
        return text
    return text[:length-3] + "..."


class InsiderDashboard:
    """
    Interactive CLI dashboard for insider detection.

    Displays suspicious activity in real-time with auto-refresh.
    """

    def __init__(
        self,
        detector: Optional[InsiderDetector] = None,
        alert_manager: Optional[AlertManager] = None
    ):
        self.detector = detector or insider_detector
        self.alert_manager = alert_manager
        self.running = False
        self.last_refresh: Optional[datetime] = None

    def print_header(self):
        """Print dashboard header."""
        print(f"\n{Colors.BOLD}{Colors.RED}{'=' * 90}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.RED}           POLYMARKET INSIDER DETECTION DASHBOARD{Colors.END}")
        print(f"{Colors.BOLD}{Colors.RED}{'=' * 90}{Colors.END}")
        print(f"{Colors.DIM}Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC{Colors.END}")
        print(f"{Colors.DIM}DISCLAIMER: Flags require human review. Not proof of wrongdoing.{Colors.END}")

    def print_config_summary(self):
        """Print current detection configuration."""
        config = self.detector.config
        print(f"\n{Colors.BOLD}Detection Thresholds:{Colors.END}")
        print(f"  Large Bet: >${config.LARGE_BET_THRESHOLD:,.0f}")
        print(f"  Resolution Window: {config.RESOLUTION_WINDOW_HOURS}h")
        print(f"  Dormant Threshold: {config.DORMANT_DAYS_THRESHOLD} days")
        print(f"  Alert Score: >{config.ALERT_SCORE_THRESHOLD}")

    def print_suspicious_bets_feed(
        self,
        limit: int = 15,
        min_score: float = 40.0,
        hours: int = 24
    ):
        """
        Print feed of suspicious bets.

        Args:
            limit: Maximum bets to show.
            min_score: Minimum anomaly score.
            hours: Look back period.
        """
        print(f"\n{Colors.BOLD}{Colors.RED}Suspicious Bets (Last {hours}h, Score >{min_score}){Colors.END}")
        print(f"{Colors.DIM}{'-' * 90}{Colors.END}")

        since = datetime.utcnow() - timedelta(hours=hours)

        try:
            suspicious_bets = self.detector.analyze_all_bets(since=since, limit=200)
            filtered = [b for b in suspicious_bets if b.anomaly_score >= min_score][:limit]

            if not filtered:
                print(f"{Colors.GREEN}No suspicious bets detected in the last {hours} hours.{Colors.END}")
                return

            # Header
            print(f"{Colors.BOLD}{'Severity':<10} {'Score':>6} {'Wallet':<14} {'Amount':>10} "
                  f"{'Market':<25} {'Flags':<20}{Colors.END}")
            print(f"{Colors.DIM}{'-' * 90}{Colors.END}")

            for bet in filtered:
                severity = format_severity(bet.severity)
                score = format_score(bet.anomaly_score)
                wallet = f"{bet.wallet_address[:6]}...{bet.wallet_address[-4:]}"
                amount = format_amount(bet.amount * bet.price)
                market = truncate(bet.market_question or bet.market_id, 25)
                flags = ", ".join(a.value.split("_")[0][:8] for a in bet.alert_types[:2])

                print(f"{severity} {score} {wallet:<14} {amount:>10} {market:<25} {flags:<20}")

        except Exception as e:
            print(f"{Colors.RED}Error loading suspicious bets: {e}{Colors.END}")

    def print_high_risk_wallets(self, limit: int = 10, min_score: float = 60.0):
        """
        Print list of high-risk wallets.

        Args:
            limit: Maximum wallets to show.
            min_score: Minimum risk score.
        """
        print(f"\n{Colors.BOLD}{Colors.MAGENTA}High-Risk Wallets{Colors.END}")
        print(f"{Colors.DIM}{'-' * 90}{Colors.END}")

        try:
            # Get all wallets with suspicious activity
            with db.get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT DISTINCT wallet_address
                    FROM traders
                    ORDER BY total_volume DESC
                    LIMIT 100
                    """
                )
                wallets = [row["wallet_address"] for row in cursor.fetchall()]

            risk_profiles = []
            for wallet in wallets:
                try:
                    profile = self.detector.build_wallet_risk_profile(wallet)
                    if profile.overall_risk_score >= min_score:
                        risk_profiles.append(profile)
                except Exception:
                    continue

            risk_profiles.sort(key=lambda x: x.overall_risk_score, reverse=True)
            risk_profiles = risk_profiles[:limit]

            if not risk_profiles:
                print(f"{Colors.GREEN}No high-risk wallets identified.{Colors.END}")
                return

            print(f"{Colors.BOLD}{'Risk':>6} {'Wallet':<14} {'Susp/Total':>12} {'Volume':>12} "
                  f"{'Flags':<30}{Colors.END}")
            print(f"{Colors.DIM}{'-' * 90}{Colors.END}")

            for profile in risk_profiles:
                score = format_score(profile.overall_risk_score)
                wallet = f"{profile.wallet_address[:6]}...{profile.wallet_address[-4:]}"
                ratio = f"{profile.total_suspicious_bets}/{profile.total_bets_analyzed}"
                volume = format_amount(profile.total_volume)
                flags = ", ".join(profile.flags[:3]) if profile.flags else "-"

                print(f"{score} {wallet:<14} {ratio:>12} {volume:>12} {truncate(flags, 30):<30}")

        except Exception as e:
            print(f"{Colors.RED}Error loading risk profiles: {e}{Colors.END}")

    def print_correlated_betting(self, limit: int = 5):
        """Print detected correlated betting patterns."""
        print(f"\n{Colors.BOLD}{Colors.YELLOW}Correlated Betting Patterns{Colors.END}")
        print(f"{Colors.DIM}{'-' * 90}{Colors.END}")

        try:
            # Check recent markets for correlation
            with db.get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT DISTINCT market_id
                    FROM bets
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                    LIMIT 50
                    """,
                    (datetime.utcnow() - timedelta(hours=24),)
                )
                markets = [row["market_id"] for row in cursor.fetchall()]

            all_groups = []
            for market_id in markets:
                groups = self.detector.detect_correlated_betting(market_id)
                all_groups.extend(groups)

            # Sort by correlation score
            all_groups.sort(key=lambda x: x.correlation_score, reverse=True)
            all_groups = all_groups[:limit]

            if not all_groups:
                print(f"{Colors.GREEN}No correlated betting patterns detected.{Colors.END}")
                return

            print(f"{Colors.BOLD}{'Score':>6} {'Wallets':>8} {'Amount':>10} {'Window':>10} "
                  f"{'Outcome':<10} {'Market':<30}{Colors.END}")
            print(f"{Colors.DIM}{'-' * 90}{Colors.END}")

            for group in all_groups:
                score = format_score(group.correlation_score)
                wallets = str(len(group.wallets))
                amount = format_amount(group.total_amount)
                window = f"{(group.timestamp_end - group.timestamp_start).total_seconds() / 60:.0f}m"
                outcome = group.outcome[:10].upper()
                market = truncate(group.market_id, 30)

                print(f"{score} {wallets:>8} {amount:>10} {window:>10} {outcome:<10} {market:<30}")

        except Exception as e:
            print(f"{Colors.RED}Error loading correlation data: {e}{Colors.END}")

    def print_market_alerts(self, limit: int = 5):
        """Print market manipulation alerts."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}Market Manipulation Alerts{Colors.END}")
        print(f"{Colors.DIM}{'-' * 90}{Colors.END}")

        try:
            # Check recent markets for volume spikes
            with db.get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT market_id, question, volume
                    FROM markets
                    WHERE active = 1
                    ORDER BY volume DESC
                    LIMIT 20
                    """
                )
                markets = cursor.fetchall()

            alerts = []
            for market in markets:
                alert = self.detector.detect_volume_spike(market["market_id"])
                if alert:
                    alerts.append(alert)

            alerts = alerts[:limit]

            if not alerts:
                print(f"{Colors.GREEN}No market manipulation alerts.{Colors.END}")
                return

            print(f"{Colors.BOLD}{'Severity':<10} {'Type':<15} {'Spike':>8} "
                  f"{'Market':<40}{Colors.END}")
            print(f"{Colors.DIM}{'-' * 90}{Colors.END}")

            for alert in alerts:
                severity = format_severity(alert.severity)
                alert_type = alert.alert_type.replace("_", " ").title()[:15]
                spike = f"{alert.details.get('spike_multiplier', 0):.1f}x"
                market = truncate(alert.market_question or alert.market_id, 40)

                print(f"{severity} {alert_type:<15} {spike:>8} {market:<40}")

        except Exception as e:
            print(f"{Colors.RED}Error loading market alerts: {e}{Colors.END}")

    def print_alert_type_stats(self):
        """Print breakdown of alert types."""
        print(f"\n{Colors.BOLD}Alert Type Distribution (Last 24h){Colors.END}")
        print(f"{Colors.DIM}{'-' * 50}{Colors.END}")

        try:
            since = datetime.utcnow() - timedelta(hours=24)
            suspicious_bets = self.detector.analyze_all_bets(since=since, limit=500)

            type_counts: dict[str, int] = {}
            for bet in suspicious_bets:
                for alert_type in bet.alert_types:
                    type_counts[alert_type.value] = type_counts.get(alert_type.value, 0) + 1

            if not type_counts:
                print(f"{Colors.DIM}No alerts in the last 24 hours.{Colors.END}")
                return

            # Sort by count
            sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)

            for alert_type, count in sorted_types:
                bar_length = min(30, count)
                bar = "" * bar_length
                label = alert_type.replace("_", " ").title()
                print(f"  {label:<30} {count:>4} {Colors.RED}{bar}{Colors.END}")

        except Exception as e:
            print(f"{Colors.RED}Error loading stats: {e}{Colors.END}")

    def print_wallet_details(self, wallet_address: str):
        """
        Print detailed analysis for a specific wallet.

        Args:
            wallet_address: Wallet to analyze.
        """
        print(f"\n{Colors.BOLD}{Colors.MAGENTA}Wallet Risk Analysis{Colors.END}")
        print(f"{Colors.DIM}{'=' * 70}{Colors.END}")
        print(f"\n{Colors.BOLD}Address:{Colors.END} {wallet_address}")

        try:
            profile = self.detector.build_wallet_risk_profile(wallet_address)

            # Risk score
            score_color = Colors.RED if profile.overall_risk_score >= 80 else (
                Colors.YELLOW if profile.overall_risk_score >= 60 else Colors.GREEN
            )
            print(f"\n{Colors.BOLD}Overall Risk Score:{Colors.END} "
                  f"{score_color}{profile.overall_risk_score:.1f}/100{Colors.END}")

            # Activity
            print(f"\n{Colors.BOLD}Activity:{Colors.END}")
            print(f"  Total Bets Analyzed: {profile.total_bets_analyzed}")
            print(f"  Suspicious Bets:     {profile.total_suspicious_bets}")
            suspicion_rate = (profile.total_suspicious_bets / max(1, profile.total_bets_analyzed)) * 100
            print(f"  Suspicion Rate:      {suspicion_rate:.1f}%")
            print(f"  Total Volume:        {format_amount(profile.total_volume)}")
            print(f"  Avg Bet Size:        {format_amount(profile.avg_bet_size)}")

            if profile.first_seen:
                print(f"  First Seen:          {profile.first_seen.strftime('%Y-%m-%d')}")
            if profile.last_seen:
                print(f"  Last Seen:           {profile.last_seen.strftime('%Y-%m-%d')}")

            # Alert breakdown
            if profile.alert_counts:
                print(f"\n{Colors.BOLD}Alert Breakdown:{Colors.END}")
                for alert_type, count in sorted(profile.alert_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {alert_type.replace('_', ' ').title():<30} {count:>4}")

            # Flags
            if profile.flags:
                print(f"\n{Colors.BOLD}Behavioral Flags:{Colors.END}")
                for flag in profile.flags:
                    print(f"  {Colors.YELLOW}- {flag.replace('_', ' ').title()}{Colors.END}")

            # Niche market performance
            if profile.win_rate_niche > 0:
                print(f"\n{Colors.BOLD}Niche Market Performance:{Colors.END}")
                wr_color = Colors.RED if profile.win_rate_niche > 75 else Colors.YELLOW
                print(f"  Win Rate on Low-Volume Markets: {wr_color}{profile.win_rate_niche:.1f}%{Colors.END}")

            # Associated wallets (Sybil)
            if profile.associated_wallets:
                print(f"\n{Colors.BOLD}Potential Sybil Connections:{Colors.END}")
                for assoc in profile.associated_wallets[:5]:
                    print(f"  {Colors.DIM}{assoc[:10]}...{assoc[-6:]}{Colors.END}")

            # Recent suspicious bets
            print(f"\n{Colors.BOLD}Recent Suspicious Bets:{Colors.END}")
            with db.get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT bet_id FROM bets
                    WHERE wallet_address = ?
                    ORDER BY timestamp DESC
                    LIMIT 10
                    """,
                    (wallet_address,)
                )
                bet_ids = [row["bet_id"] for row in cursor.fetchall()]

            for bet_id in bet_ids[:5]:
                bet = self.detector.calculate_anomaly_score(bet_id)
                if bet.anomaly_score > 20:
                    score = format_score(bet.anomaly_score)
                    print(f"  {score} - {truncate(bet.market_question or bet.market_id, 40)}")

        except Exception as e:
            print(f"{Colors.RED}Error analyzing wallet: {e}{Colors.END}")

    def display(
        self,
        min_score: float = 40.0,
        hours: int = 24,
        show_config: bool = False
    ):
        """
        Display full dashboard.

        Args:
            min_score: Minimum score for display.
            hours: Look back period.
            show_config: Whether to show configuration.
        """
        clear_screen()
        self.print_header()

        if show_config:
            self.print_config_summary()

        self.print_suspicious_bets_feed(limit=12, min_score=min_score, hours=hours)
        self.print_high_risk_wallets(limit=8, min_score=60)
        self.print_correlated_betting(limit=3)
        self.print_market_alerts(limit=3)
        self.print_alert_type_stats()

        print(f"\n{Colors.DIM}Press Ctrl+C to exit | Thresholds adjustable via --min-score{Colors.END}")

    def run_interactive(
        self,
        refresh_interval: int = 30,
        min_score: float = 40.0,
        hours: int = 24
    ):
        """
        Run dashboard with auto-refresh.

        Args:
            refresh_interval: Seconds between refreshes.
            min_score: Minimum anomaly score.
            hours: Look back period.
        """
        self.running = True
        print(f"{Colors.CYAN}Starting insider detection dashboard ({refresh_interval}s refresh)...{Colors.END}")

        try:
            while self.running:
                self.display(min_score=min_score, hours=hours)
                self.last_refresh = datetime.utcnow()

                print(f"\n{Colors.DIM}Refreshing in {refresh_interval} seconds...{Colors.END}")

                start = time.time()
                while time.time() - start < refresh_interval:
                    time.sleep(1)

        except KeyboardInterrupt:
            self.running = False
            print(f"\n{Colors.YELLOW}Dashboard stopped.{Colors.END}")


def main():
    """Main entry point for insider detection dashboard."""
    parser = argparse.ArgumentParser(
        description="Polymarket Insider Detection Dashboard"
    )
    parser.add_argument(
        "--refresh", "-r",
        type=int,
        default=30,
        help="Refresh interval in seconds (default: 30)"
    )
    parser.add_argument(
        "--min-score", "-s",
        type=float,
        default=40.0,
        help="Minimum anomaly score to display (default: 40)"
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Hours to look back (default: 24)"
    )
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Display once without auto-refresh"
    )
    parser.add_argument(
        "--wallet", "-w",
        type=str,
        help="Analyze specific wallet address"
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Show detection configuration"
    )
    parser.add_argument(
        "--analyze-bet",
        type=str,
        help="Analyze specific bet by ID"
    )

    args = parser.parse_args()
    dashboard = InsiderDashboard()

    # Analyze specific bet
    if args.analyze_bet:
        print(f"\n{Colors.BOLD}Analyzing Bet: {args.analyze_bet}{Colors.END}")
        bet = insider_detector.calculate_anomaly_score(args.analyze_bet)
        print(f"\nAnomaly Score: {format_score(bet.anomaly_score)}")
        print(f"Severity: {format_severity(bet.severity)}")
        print(f"Wallet: {bet.wallet_address}")
        print(f"Amount: {format_amount(bet.amount * bet.price)}")
        print(f"Market: {bet.market_question}")
        print(f"\nFlags: {', '.join(a.value for a in bet.alert_types)}")
        print(f"\nScore Breakdown:")
        for factor, score in bet.score_breakdown.items():
            print(f"  {factor}: +{score}")
        return

    # Analyze specific wallet
    if args.wallet:
        dashboard.print_wallet_details(args.wallet)
        return

    # Run dashboard
    if args.no_refresh:
        dashboard.display(
            min_score=args.min_score,
            hours=args.hours,
            show_config=args.show_config
        )
    else:
        dashboard.run_interactive(
            refresh_interval=args.refresh,
            min_score=args.min_score,
            hours=args.hours
        )


if __name__ == "__main__":
    main()
