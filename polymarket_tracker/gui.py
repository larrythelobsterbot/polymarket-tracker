"""
Polymarket Tracker GUI Application.

A desktop GUI built with Tkinter for monitoring Polymarket trading data,
viewing leaderboards, and detecting suspicious trading patterns.

Usage:
    python -m polymarket_tracker.gui
"""

import asyncio
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime, timedelta
from typing import Optional, Callable
import queue
import logging

from .database import db
from .config import settings
from .analytics import PolymarketAnalytics, LeaderboardMetric
from .win_rate import WinRateAnalyzer
from .insider_detection import InsiderDetector, AlertSeverity
from .collector import DataCollector

logger = logging.getLogger(__name__)


class AsyncRunner:
    """Runs async functions in a background thread."""

    def __init__(self):
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self._started = False

    def start(self):
        """Start the async event loop in a background thread."""
        if self._started:
            return

        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()

        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()
        self._started = True

    def run(self, coro, callback: Optional[Callable] = None):
        """Run a coroutine and optionally call a callback with the result."""
        if not self._started:
            self.start()

        def wrapper():
            try:
                result = asyncio.run_coroutine_threadsafe(coro, self.loop).result()
                if callback:
                    callback(result, None)
            except Exception as e:
                if callback:
                    callback(None, e)

        threading.Thread(target=wrapper, daemon=True).start()

    def stop(self):
        """Stop the async event loop."""
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)


class StatusBar(ttk.Frame):
    """Status bar at the bottom of the application."""

    def __init__(self, parent):
        super().__init__(parent)

        self.status_label = ttk.Label(self, text="Ready", anchor="w")
        self.status_label.pack(side="left", fill="x", expand=True, padx=5)

        self.db_label = ttk.Label(self, text="", anchor="e")
        self.db_label.pack(side="right", padx=5)

        self.update_db_stats()

    def set_status(self, message: str):
        """Set the status message."""
        self.status_label.config(text=message)

    def update_db_stats(self):
        """Update database statistics display."""
        try:
            with db.get_connection() as conn:
                traders = conn.execute("SELECT COUNT(*) FROM traders").fetchone()[0]
                markets = conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0]
                bets = conn.execute("SELECT COUNT(*) FROM bets").fetchone()[0]

            self.db_label.config(text=f"📊 {traders:,} traders | {markets:,} markets | {bets:,} bets")
        except Exception as e:
            self.db_label.config(text="Database: Error")


class DashboardTab(ttk.Frame):
    """Main dashboard with leaderboards and statistics."""

    def __init__(self, parent, status_bar: StatusBar):
        super().__init__(parent)
        self.status_bar = status_bar
        self.analytics = PolymarketAnalytics()

        self._create_widgets()
        self.refresh_data()

    def _create_widgets(self):
        """Create dashboard widgets."""
        # Top controls
        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=10, pady=5)

        ttk.Label(controls, text="Sort by:").pack(side="left", padx=5)

        self.metric_var = tk.StringVar(value="pnl")
        metrics = [
            ("PNL", "pnl"),
            ("Volume", "volume"),
            ("Win Rate", "win_rate"),
            ("ROI", "roi"),
            ("Trade Count", "trade_count"),
        ]
        for text, value in metrics:
            ttk.Radiobutton(
                controls, text=text, variable=self.metric_var,
                value=value, command=self.refresh_data
            ).pack(side="left", padx=5)

        ttk.Button(controls, text="🔄 Refresh", command=self.refresh_data).pack(side="right", padx=5)

        # Limit control
        ttk.Label(controls, text="Top:").pack(side="right", padx=5)
        self.limit_var = tk.StringVar(value="20")
        limit_combo = ttk.Combobox(controls, textvariable=self.limit_var, width=5,
                                    values=["10", "20", "50", "100"])
        limit_combo.pack(side="right", padx=5)
        limit_combo.bind("<<ComboboxSelected>>", lambda e: self.refresh_data())

        # Main content - Paned window for leaderboard and details
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=10, pady=5)

        # Leaderboard frame
        leaderboard_frame = ttk.LabelFrame(paned, text="Leaderboard")
        paned.add(leaderboard_frame, weight=2)

        # Leaderboard treeview
        columns = ("rank", "wallet", "pnl", "volume", "win_rate", "trades", "roi")
        self.tree = ttk.Treeview(leaderboard_frame, columns=columns, show="headings", height=20)

        self.tree.heading("rank", text="#")
        self.tree.heading("wallet", text="Wallet")
        self.tree.heading("pnl", text="Unreal. PNL")
        self.tree.heading("volume", text="Volume")
        self.tree.heading("win_rate", text="Win %")
        self.tree.heading("trades", text="Trades")
        self.tree.heading("roi", text="ROI %")

        self.tree.column("rank", width=40, anchor="center")
        self.tree.column("wallet", width=150)
        self.tree.column("pnl", width=100, anchor="e")
        self.tree.column("volume", width=100, anchor="e")
        self.tree.column("win_rate", width=70, anchor="center")
        self.tree.column("trades", width=70, anchor="center")
        self.tree.column("roi", width=70, anchor="e")

        scrollbar = ttk.Scrollbar(leaderboard_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.tree.bind("<<TreeviewSelect>>", self._on_trader_select)

        # Details frame
        details_frame = ttk.LabelFrame(paned, text="Trader Details")
        paned.add(details_frame, weight=1)

        self.details_text = tk.Text(details_frame, wrap="word", width=40, height=25,
                                     font=("Consolas", 10))
        self.details_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.details_text.insert("1.0", "Select a trader to view details...")
        self.details_text.config(state="disabled")

        # Summary stats at bottom
        summary_frame = ttk.LabelFrame(self, text="Summary Statistics")
        summary_frame.pack(fill="x", padx=10, pady=5)

        self.summary_labels = {}
        stats = [
            ("total_pnl", "Total PNL"),
            ("total_volume", "Total Volume"),
            ("avg_win_rate", "Avg Win Rate"),
            ("active_traders", "Active Traders"),
        ]

        for i, (key, label) in enumerate(stats):
            frame = ttk.Frame(summary_frame)
            frame.pack(side="left", expand=True, padx=20, pady=5)
            ttk.Label(frame, text=label, font=("Arial", 9)).pack()
            self.summary_labels[key] = ttk.Label(frame, text="--", font=("Arial", 14, "bold"))
            self.summary_labels[key].pack()

    def refresh_data(self):
        """Refresh leaderboard data."""
        self.status_bar.set_status("Refreshing leaderboard...")

        try:
            # Get metric
            metric_map = {
                "pnl": LeaderboardMetric.PNL,
                "volume": LeaderboardMetric.VOLUME,
                "win_rate": LeaderboardMetric.WIN_RATE,
                "roi": LeaderboardMetric.ROI,
                "trade_count": LeaderboardMetric.TRADE_COUNT,
            }
            metric = metric_map.get(self.metric_var.get(), LeaderboardMetric.PNL)
            limit = int(self.limit_var.get())

            # Get leaderboard
            leaderboard = self.analytics.get_top_traders(metric=metric, limit=limit)

            # Clear and populate tree
            for item in self.tree.get_children():
                self.tree.delete(item)

            total_pnl = 0
            total_volume = 0
            total_win_rate = 0

            for entry in leaderboard:
                # Format wallet address (truncate)
                wallet = entry.wallet_address
                short_wallet = f"{wallet[:6]}...{wallet[-4:]}" if len(wallet) > 12 else wallet

                # Format numbers - show unrealized PNL since most markets aren't resolved
                # Use unrealized_pnl if available, otherwise show total_pnl
                display_pnl = entry.unrealized_pnl if entry.unrealized_pnl != 0 else entry.total_pnl
                pnl_str = f"${display_pnl:,.2f}"
                if display_pnl >= 0:
                    pnl_str = f"+{pnl_str}"

                volume_str = f"${entry.total_volume:,.0f}"
                win_str = f"{entry.win_rate:.1f}%"
                roi_str = f"{entry.roi:.1f}%"

                self.tree.insert("", "end", values=(
                    entry.rank,
                    short_wallet,
                    pnl_str,
                    volume_str,
                    win_str,
                    entry.total_trades,
                    roi_str
                ), tags=(wallet,))

                total_pnl += display_pnl
                total_volume += entry.total_volume
                total_win_rate += entry.win_rate

            # Update summary
            avg_win = total_win_rate / len(leaderboard) if leaderboard else 0
            self.summary_labels["total_pnl"].config(
                text=f"${total_pnl:,.0f}",
                foreground="green" if total_pnl >= 0 else "red"
            )
            self.summary_labels["total_volume"].config(text=f"${total_volume:,.0f}")
            self.summary_labels["avg_win_rate"].config(text=f"{avg_win:.1f}%")
            self.summary_labels["active_traders"].config(text=str(len(leaderboard)))

            self.status_bar.set_status(f"Loaded {len(leaderboard)} traders")
            self.status_bar.update_db_stats()

        except Exception as e:
            logger.exception("Error refreshing data")
            self.status_bar.set_status(f"Error: {e}")

    def _on_trader_select(self, event):
        """Handle trader selection."""
        selection = self.tree.selection()
        if not selection:
            return

        item = self.tree.item(selection[0])
        wallet = item["tags"][0] if item["tags"] else None

        if wallet:
            self._show_trader_details(wallet)

    def _show_trader_details(self, wallet: str):
        """Show detailed information for a trader."""
        self.details_text.config(state="normal")
        self.details_text.delete("1.0", "end")

        try:
            # Get trader stats
            stats = self.analytics.get_trader_stats(wallet)
            win_rate_analyzer = WinRateAnalyzer()
            win_stats = win_rate_analyzer.calculate_win_rate(wallet)

            text = f"TRADER DETAILS\n{'='*40}\n\n"
            text += f"Wallet: {wallet[:10]}...{wallet[-6:]}\n\n"

            if stats:
                text += f"📊 PERFORMANCE\n{'-'*30}\n"
                text += f"Realized PNL:  ${stats.realized_pnl:,.2f}\n"
                text += f"Total Volume:  ${stats.total_volume:,.0f}\n"
                text += f"Total Trades:  {stats.total_trades}\n"
                text += f"ROI:           {stats.roi:.2f}%\n\n"

            if win_stats:
                text += f"🎯 WIN RATE ANALYSIS\n{'-'*30}\n"
                text += f"Win Rate:      {win_stats.win_rate:.1f}%\n"
                text += f"Wins/Losses:   {win_stats.wins}/{win_stats.losses}\n"
                text += f"Confidence:    [{win_stats.confidence_lower:.1f}% - {win_stats.confidence_upper:.1f}%]\n\n"

                if win_stats.current_streak != 0:
                    streak_type = "winning" if win_stats.current_streak > 0 else "losing"
                    text += f"Current Streak: {abs(win_stats.current_streak)} {streak_type}\n"

                text += f"Best Streak:   {win_stats.longest_win_streak} wins\n"
                text += f"Worst Streak:  {win_stats.longest_loss_streak} losses\n\n"

                if win_stats.kelly_criterion:
                    text += f"📈 KELLY CRITERION\n{'-'*30}\n"
                    text += f"Optimal Fraction: {win_stats.kelly_criterion:.2%}\n"

            # Recent activity
            with db.get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT b.timestamp, m.question, b.side, b.amount, b.price
                    FROM bets b
                    JOIN markets m ON b.market_id = m.market_id
                    WHERE b.wallet_address = ?
                    ORDER BY b.timestamp DESC
                    LIMIT 5
                    """,
                    (wallet,)
                )
                recent = cursor.fetchall()

            if recent:
                text += f"\n📝 RECENT BETS\n{'-'*30}\n"
                for bet in recent:
                    ts = bet["timestamp"]
                    if isinstance(ts, str):
                        ts = datetime.fromisoformat(ts)
                    question = bet["question"][:30] + "..." if len(bet["question"]) > 30 else bet["question"]
                    text += f"• {bet['side']} ${bet['amount']:.2f} @ {bet['price']:.2f}\n"
                    text += f"  {question}\n"

            self.details_text.insert("1.0", text)

        except Exception as e:
            self.details_text.insert("1.0", f"Error loading details: {e}")

        self.details_text.config(state="disabled")


class InsiderTab(ttk.Frame):
    """Insider detection and suspicious activity monitoring."""

    def __init__(self, parent, status_bar: StatusBar):
        super().__init__(parent)
        self.status_bar = status_bar
        self.detector = InsiderDetector()

        self._create_widgets()
        self.refresh_data()

    def _create_widgets(self):
        """Create insider detection widgets."""
        # Top controls
        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=10, pady=5)

        ttk.Label(controls, text="Min Score:").pack(side="left", padx=5)
        self.score_var = tk.StringVar(value="60")
        score_spin = ttk.Spinbox(controls, from_=0, to=100, textvariable=self.score_var, width=5)
        score_spin.pack(side="left", padx=5)

        ttk.Button(controls, text="🔄 Refresh", command=self.refresh_data).pack(side="left", padx=10)
        ttk.Button(controls, text="📊 Analyze Selected", command=self._analyze_selected).pack(side="left", padx=5)

        # Main content
        paned = ttk.PanedWindow(self, orient="vertical")
        paned.pack(fill="both", expand=True, padx=10, pady=5)

        # Suspicious bets frame
        bets_frame = ttk.LabelFrame(paned, text="🚨 Suspicious Bets")
        paned.add(bets_frame, weight=2)

        columns = ("score", "severity", "wallet", "market", "amount", "alerts", "timestamp")
        self.bets_tree = ttk.Treeview(bets_frame, columns=columns, show="headings", height=12)

        self.bets_tree.heading("score", text="Score")
        self.bets_tree.heading("severity", text="Severity")
        self.bets_tree.heading("wallet", text="Wallet")
        self.bets_tree.heading("market", text="Market")
        self.bets_tree.heading("amount", text="Amount")
        self.bets_tree.heading("alerts", text="Alert Types")
        self.bets_tree.heading("timestamp", text="Time")

        self.bets_tree.column("score", width=60, anchor="center")
        self.bets_tree.column("severity", width=80, anchor="center")
        self.bets_tree.column("wallet", width=120)
        self.bets_tree.column("market", width=200)
        self.bets_tree.column("amount", width=80, anchor="e")
        self.bets_tree.column("alerts", width=150)
        self.bets_tree.column("timestamp", width=100)

        bets_scroll = ttk.Scrollbar(bets_frame, orient="vertical", command=self.bets_tree.yview)
        self.bets_tree.configure(yscrollcommand=bets_scroll.set)

        self.bets_tree.pack(side="left", fill="both", expand=True)
        bets_scroll.pack(side="right", fill="y")

        # High-risk wallets frame
        wallets_frame = ttk.LabelFrame(paned, text="⚠️ High-Risk Wallets")
        paned.add(wallets_frame, weight=1)

        columns = ("risk", "wallet", "suspicious", "total_bets", "win_rate", "volume")
        self.wallets_tree = ttk.Treeview(wallets_frame, columns=columns, show="headings", height=8)

        self.wallets_tree.heading("risk", text="Risk")
        self.wallets_tree.heading("wallet", text="Wallet")
        self.wallets_tree.heading("suspicious", text="Suspicious")
        self.wallets_tree.heading("total_bets", text="Total Bets")
        self.wallets_tree.heading("win_rate", text="Win Rate")
        self.wallets_tree.heading("volume", text="Volume")

        self.wallets_tree.column("risk", width=60, anchor="center")
        self.wallets_tree.column("wallet", width=150)
        self.wallets_tree.column("suspicious", width=80, anchor="center")
        self.wallets_tree.column("total_bets", width=80, anchor="center")
        self.wallets_tree.column("win_rate", width=80, anchor="center")
        self.wallets_tree.column("volume", width=100, anchor="e")

        wallets_scroll = ttk.Scrollbar(wallets_frame, orient="vertical", command=self.wallets_tree.yview)
        self.wallets_tree.configure(yscrollcommand=wallets_scroll.set)

        self.wallets_tree.pack(side="left", fill="both", expand=True)
        wallets_scroll.pack(side="right", fill="y")

        # Configure tag colors
        self.bets_tree.tag_configure("critical", background="#ffcccc")
        self.bets_tree.tag_configure("high", background="#ffe0cc")
        self.bets_tree.tag_configure("medium", background="#ffffcc")
        self.bets_tree.tag_configure("low", background="#e6ffe6")

    def refresh_data(self):
        """Refresh suspicious activity data."""
        self.status_bar.set_status("Scanning for suspicious activity...")

        try:
            min_score = float(self.score_var.get())

            # Clear trees
            for item in self.bets_tree.get_children():
                self.bets_tree.delete(item)
            for item in self.wallets_tree.get_children():
                self.wallets_tree.delete(item)

            # Get recent bets and score them
            with db.get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT bet_id FROM bets
                    ORDER BY timestamp DESC
                    LIMIT 500
                    """
                )
                bet_ids = [row["bet_id"] for row in cursor.fetchall()]

            suspicious_count = 0
            wallet_scores = {}

            for bet_id in bet_ids:
                try:
                    suspicious = self.detector.calculate_anomaly_score(bet_id)

                    if suspicious.anomaly_score >= min_score:
                        suspicious_count += 1

                        # Get market question
                        with db.get_connection() as conn:
                            cursor = conn.execute(
                                """
                                SELECT m.question, b.wallet_address, b.amount, b.price, b.timestamp
                                FROM bets b
                                JOIN markets m ON b.market_id = m.market_id
                                WHERE b.bet_id = ?
                                """,
                                (bet_id,)
                            )
                            row = cursor.fetchone()

                        if row:
                            wallet = row["wallet_address"]
                            short_wallet = f"{wallet[:6]}...{wallet[-4:]}"
                            question = row["question"][:40] + "..." if len(row["question"]) > 40 else row["question"]
                            amount = f"${row['amount'] * row['price']:,.2f}"

                            ts = row["timestamp"]
                            if isinstance(ts, str):
                                ts = datetime.fromisoformat(ts)
                            time_str = ts.strftime("%m/%d %H:%M") if ts else "N/A"

                            alerts = ", ".join([a.value for a in suspicious.alert_types[:2]])

                            tag = suspicious.severity.value.lower()

                            self.bets_tree.insert("", "end", values=(
                                f"{suspicious.anomaly_score:.0f}",
                                suspicious.severity.value,
                                short_wallet,
                                question,
                                amount,
                                alerts,
                                time_str
                            ), tags=(tag, wallet, bet_id))

                            # Track wallet risk
                            if wallet not in wallet_scores:
                                wallet_scores[wallet] = {"count": 0, "max_score": 0}
                            wallet_scores[wallet]["count"] += 1
                            wallet_scores[wallet]["max_score"] = max(
                                wallet_scores[wallet]["max_score"],
                                suspicious.anomaly_score
                            )

                except Exception as e:
                    logger.debug(f"Error scoring bet {bet_id}: {e}")

            # Populate high-risk wallets
            sorted_wallets = sorted(
                wallet_scores.items(),
                key=lambda x: x[1]["max_score"],
                reverse=True
            )[:20]

            for wallet, scores in sorted_wallets:
                try:
                    profile = self.detector.build_wallet_risk_profile(wallet)
                    short_wallet = f"{wallet[:6]}...{wallet[-4:]}"

                    self.wallets_tree.insert("", "end", values=(
                        f"{profile.overall_risk_score:.0f}",
                        short_wallet,
                        profile.total_suspicious_bets,
                        profile.total_bets,
                        f"{profile.win_rate:.1f}%",
                        f"${profile.total_volume:,.0f}"
                    ), tags=(wallet,))
                except Exception:
                    pass

            self.status_bar.set_status(f"Found {suspicious_count} suspicious bets")

        except Exception as e:
            logger.exception("Error refreshing insider data")
            self.status_bar.set_status(f"Error: {e}")

    def _analyze_selected(self):
        """Analyze selected bet or wallet in detail."""
        selection = self.bets_tree.selection()
        if selection:
            item = self.bets_tree.item(selection[0])
            tags = item["tags"]
            if len(tags) >= 3:
                bet_id = tags[2]
                self._show_bet_analysis(bet_id)
                return

        selection = self.wallets_tree.selection()
        if selection:
            item = self.wallets_tree.item(selection[0])
            if item["tags"]:
                wallet = item["tags"][0]
                self._show_wallet_analysis(wallet)

    def _show_bet_analysis(self, bet_id: str):
        """Show detailed bet analysis popup."""
        try:
            suspicious = self.detector.calculate_anomaly_score(bet_id)

            popup = tk.Toplevel(self)
            popup.title("Bet Analysis")
            popup.geometry("500x400")

            text = tk.Text(popup, wrap="word", font=("Consolas", 10))
            text.pack(fill="both", expand=True, padx=10, pady=10)

            analysis = f"BET ANALYSIS\n{'='*50}\n\n"
            analysis += f"Bet ID: {bet_id}\n"
            analysis += f"Anomaly Score: {suspicious.anomaly_score:.1f}/100\n"
            analysis += f"Severity: {suspicious.severity.value}\n\n"

            analysis += f"BREAKDOWN\n{'-'*40}\n"
            for key, value in suspicious.score_breakdown.items():
                analysis += f"  {key}: {value:.1f}\n"

            analysis += f"\nALERT TYPES\n{'-'*40}\n"
            for alert in suspicious.alert_types:
                analysis += f"  • {alert.value}\n"

            text.insert("1.0", analysis)
            text.config(state="disabled")

        except Exception as e:
            messagebox.showerror("Error", f"Could not analyze bet: {e}")

    def _show_wallet_analysis(self, wallet: str):
        """Show detailed wallet analysis popup."""
        try:
            profile = self.detector.build_wallet_risk_profile(wallet)

            popup = tk.Toplevel(self)
            popup.title("Wallet Risk Profile")
            popup.geometry("500x450")

            text = tk.Text(popup, wrap="word", font=("Consolas", 10))
            text.pack(fill="both", expand=True, padx=10, pady=10)

            analysis = f"WALLET RISK PROFILE\n{'='*50}\n\n"
            analysis += f"Wallet: {wallet}\n\n"

            analysis += f"RISK ASSESSMENT\n{'-'*40}\n"
            analysis += f"Overall Risk Score: {profile.overall_risk_score:.1f}/100\n"
            analysis += f"Suspicious Bets: {profile.total_suspicious_bets}/{profile.total_bets}\n"
            analysis += f"High-Risk Markets: {profile.high_risk_market_count}\n\n"

            analysis += f"TRADING STATS\n{'-'*40}\n"
            analysis += f"Total Volume: ${profile.total_volume:,.2f}\n"
            analysis += f"Win Rate: {profile.win_rate:.1f}%\n"
            analysis += f"Avg Bet Size: ${profile.avg_bet_size:,.2f}\n\n"

            analysis += f"ALERT FREQUENCY\n{'-'*40}\n"
            for alert_type, count in profile.alert_type_counts.items():
                analysis += f"  {alert_type}: {count}\n"

            text.insert("1.0", analysis)
            text.config(state="disabled")

        except Exception as e:
            messagebox.showerror("Error", f"Could not analyze wallet: {e}")


class LiveFeedTab(ttk.Frame):
    """Live feed of recent trades with market names and whale alerts."""

    def __init__(self, parent, status_bar: StatusBar):
        super().__init__(parent)
        self.status_bar = status_bar
        self.alert_threshold = 1000.0  # Default $1k for whale alerts

        self._create_widgets()
        self.refresh_feed()

    def _create_widgets(self):
        """Create live feed widgets."""
        # Top controls
        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=10, pady=5)

        ttk.Button(controls, text="🔄 Refresh", command=self.refresh_feed).pack(side="left", padx=5)

        ttk.Label(controls, text="Show:").pack(side="left", padx=(20, 5))
        self.limit_var = tk.StringVar(value="100")
        limit_combo = ttk.Combobox(controls, textvariable=self.limit_var, width=6,
                                    values=["50", "100", "250", "500"])
        limit_combo.pack(side="left", padx=5)
        limit_combo.bind("<<ComboboxSelected>>", lambda e: self.refresh_feed())

        ttk.Label(controls, text="Min Size ($):").pack(side="left", padx=(20, 5))
        self.min_size_var = tk.StringVar(value="0")
        ttk.Spinbox(controls, from_=0, to=1000000, increment=100,
                    textvariable=self.min_size_var, width=10).pack(side="left", padx=5)

        ttk.Button(controls, text="Filter", command=self.refresh_feed).pack(side="left", padx=5)

        # Whale alert threshold
        ttk.Label(controls, text="🐋 Whale Alert ($):").pack(side="right", padx=5)
        self.whale_threshold_var = tk.StringVar(value="10000")
        ttk.Spinbox(controls, from_=1000, to=1000000, increment=1000,
                    textvariable=self.whale_threshold_var, width=10).pack(side="right", padx=5)

        # Main content - split view
        paned = ttk.PanedWindow(self, orient="vertical")
        paned.pack(fill="both", expand=True, padx=10, pady=5)

        # Recent trades feed
        feed_frame = ttk.LabelFrame(paned, text="📈 Recent Trades")
        paned.add(feed_frame, weight=3)

        columns = ("time", "wallet", "side", "amount", "price", "market")
        self.feed_tree = ttk.Treeview(feed_frame, columns=columns, show="headings", height=15)

        self.feed_tree.heading("time", text="Time")
        self.feed_tree.heading("wallet", text="Wallet")
        self.feed_tree.heading("side", text="Side")
        self.feed_tree.heading("amount", text="Amount")
        self.feed_tree.heading("price", text="Price")
        self.feed_tree.heading("market", text="Market")

        self.feed_tree.column("time", width=130)
        self.feed_tree.column("wallet", width=120)
        self.feed_tree.column("side", width=50, anchor="center")
        self.feed_tree.column("amount", width=100, anchor="e")
        self.feed_tree.column("price", width=60, anchor="center")
        self.feed_tree.column("market", width=400)

        feed_scroll = ttk.Scrollbar(feed_frame, orient="vertical", command=self.feed_tree.yview)
        self.feed_tree.configure(yscrollcommand=feed_scroll.set)

        self.feed_tree.pack(side="left", fill="both", expand=True)
        feed_scroll.pack(side="right", fill="y")

        # Tag configurations for visual styling
        self.feed_tree.tag_configure("whale", background="#fff3cd", font=("Arial", 9, "bold"))
        self.feed_tree.tag_configure("buy", foreground="#28a745")
        self.feed_tree.tag_configure("sell", foreground="#dc3545")

        # Whale alerts panel
        whale_frame = ttk.LabelFrame(paned, text="🐋 Whale Alerts (Large Trades)")
        paned.add(whale_frame, weight=1)

        self.whale_text = tk.Text(whale_frame, wrap="word", height=8, font=("Consolas", 10))
        whale_scroll = ttk.Scrollbar(whale_frame, orient="vertical", command=self.whale_text.yview)
        self.whale_text.configure(yscrollcommand=whale_scroll.set)

        self.whale_text.pack(side="left", fill="both", expand=True)
        whale_scroll.pack(side="right", fill="y")

        self.whale_text.insert("1.0", "Whale alerts will appear here when large trades are detected...\n")
        self.whale_text.config(state="disabled")

    def refresh_feed(self):
        """Refresh the live feed with recent trades."""
        self.status_bar.set_status("Loading recent trades...")

        try:
            limit = int(self.limit_var.get())
            min_size = float(self.min_size_var.get())
            whale_threshold = float(self.whale_threshold_var.get())

            # Clear existing items
            for item in self.feed_tree.get_children():
                self.feed_tree.delete(item)

            # Get recent trades with market names
            with db.get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT
                        b.timestamp,
                        b.wallet_address,
                        b.side,
                        b.amount,
                        b.price,
                        b.outcome_bet,
                        m.question,
                        m.market_id
                    FROM bets b
                    LEFT JOIN markets m ON b.market_id = m.market_id
                    WHERE (b.amount * b.price) >= ?
                    ORDER BY b.timestamp DESC
                    LIMIT ?
                    """,
                    (min_size, limit)
                )
                trades = cursor.fetchall()

            whale_alerts = []
            trade_count = 0

            for trade in trades:
                ts = trade["timestamp"]
                if isinstance(ts, str):
                    try:
                        ts = datetime.fromisoformat(ts)
                    except:
                        ts = None

                time_str = ts.strftime("%Y-%m-%d %H:%M") if ts else "N/A"

                wallet = trade["wallet_address"]
                short_wallet = f"{wallet[:6]}...{wallet[-4:]}" if len(wallet) > 12 else wallet

                side = trade["side"].upper()
                amount = trade["amount"]
                price = trade["price"]
                volume = amount * price
                outcome = trade["outcome_bet"] or ""
                market_name = trade["question"] or "Unknown Market"

                # Truncate market name if too long
                if len(market_name) > 60:
                    market_name = market_name[:57] + "..."

                # Format display
                amount_str = f"${volume:,.2f}"
                price_str = f"{price:.2f}"

                # Determine display for side with outcome
                side_display = f"{side}"
                if outcome:
                    side_display = f"{side} {outcome[:3].upper()}"

                # Determine tags
                tags = [wallet]
                if side == "BUY":
                    tags.append("buy")
                else:
                    tags.append("sell")

                if volume >= whale_threshold:
                    tags.append("whale")
                    # Add to whale alerts
                    whale_alerts.append({
                        "time": time_str,
                        "wallet": short_wallet,
                        "side": side,
                        "outcome": outcome,
                        "amount": volume,
                        "market": market_name,
                        "full_wallet": wallet
                    })

                self.feed_tree.insert("", "end", values=(
                    time_str,
                    short_wallet,
                    side_display,
                    amount_str,
                    price_str,
                    market_name
                ), tags=tuple(tags))

                trade_count += 1

            # Update whale alerts text
            self.whale_text.config(state="normal")
            self.whale_text.delete("1.0", "end")

            if whale_alerts:
                self.whale_text.insert("end", f"🐋 Found {len(whale_alerts)} whale trades (>${whale_threshold:,.0f}):\n\n")
                for alert in whale_alerts[:20]:  # Show top 20 whale alerts
                    alert_text = (
                        f"⚡ {alert['time']} | {alert['wallet']} {alert['side']} ${alert['amount']:,.2f} "
                        f"on \"{alert['outcome']}\" in:\n   📊 {alert['market']}\n\n"
                    )
                    self.whale_text.insert("end", alert_text)
            else:
                self.whale_text.insert("end", f"No whale trades found above ${whale_threshold:,.0f} threshold.\n")
                self.whale_text.insert("end", "Lower the threshold or sync more data to see alerts.")

            self.whale_text.config(state="disabled")

            self.status_bar.set_status(f"Loaded {trade_count} trades, {len(whale_alerts)} whale alerts")

        except Exception as e:
            logger.exception("Error refreshing feed")
            self.status_bar.set_status(f"Error: {e}")

    def get_whale_alerts_for_notification(self, since_minutes: int = 5) -> list:
        """Get whale alerts from the last N minutes for notifications."""
        whale_threshold = float(self.whale_threshold_var.get())
        cutoff = datetime.now() - timedelta(minutes=since_minutes)

        alerts = []
        try:
            with db.get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT
                        b.timestamp,
                        b.wallet_address,
                        b.side,
                        b.amount,
                        b.price,
                        b.outcome_bet,
                        m.question
                    FROM bets b
                    LEFT JOIN markets m ON b.market_id = m.market_id
                    WHERE (b.amount * b.price) >= ?
                    AND b.timestamp >= ?
                    ORDER BY b.timestamp DESC
                    """,
                    (whale_threshold, cutoff.isoformat())
                )

                for row in cursor.fetchall():
                    volume = row["amount"] * row["price"]
                    alerts.append({
                        "wallet": row["wallet_address"],
                        "side": row["side"],
                        "outcome": row["outcome_bet"],
                        "amount": volume,
                        "market": row["question"] or "Unknown Market"
                    })
        except Exception as e:
            logger.error(f"Error getting whale alerts: {e}")

        return alerts


class CollectorTab(ttk.Frame):
    """Data collection controls and monitoring."""

    def __init__(self, parent, status_bar: StatusBar, async_runner: AsyncRunner):
        super().__init__(parent)
        self.status_bar = status_bar
        self.async_runner = async_runner
        self.is_collecting = False
        self.collector: Optional[DataCollector] = None

        self._create_widgets()

    def _create_widgets(self):
        """Create collector widgets."""
        # Controls
        controls_frame = ttk.LabelFrame(self, text="Collection Controls")
        controls_frame.pack(fill="x", padx=10, pady=10)

        # Buttons
        btn_frame = ttk.Frame(controls_frame)
        btn_frame.pack(fill="x", padx=10, pady=10)

        self.collect_btn = ttk.Button(btn_frame, text="▶️ Start Collection",
                                       command=self._start_collection)
        self.collect_btn.pack(side="left", padx=5)

        self.stop_btn = ttk.Button(btn_frame, text="⏹️ Stop",
                                    command=self._stop_collection, state="disabled")
        self.stop_btn.pack(side="left", padx=5)

        ttk.Button(btn_frame, text="🔄 Sync Markets",
                   command=self._sync_markets).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="📥 Sync Trades",
                   command=self._sync_trades).pack(side="left", padx=5)

        # Settings
        settings_frame = ttk.LabelFrame(self, text="Settings")
        settings_frame.pack(fill="x", padx=10, pady=10)

        row1 = ttk.Frame(settings_frame)
        row1.pack(fill="x", padx=10, pady=5)

        ttk.Label(row1, text="Fetch Interval (minutes):").pack(side="left")
        self.interval_var = tk.StringVar(value=str(settings.fetch_interval_minutes))
        ttk.Spinbox(row1, from_=1, to=60, textvariable=self.interval_var, width=5).pack(side="left", padx=5)

        # Filtering options
        row2 = ttk.Frame(settings_frame)
        row2.pack(fill="x", padx=10, pady=5)

        ttk.Label(row2, text="Min Trade Size ($):").pack(side="left")
        self.min_trade_var = tk.StringVar(value=str(int(settings.min_trade_size)))
        ttk.Spinbox(row2, from_=0, to=100000, increment=100, textvariable=self.min_trade_var, width=8).pack(side="left", padx=5)

        ttk.Label(row2, text="Min Market Volume ($):").pack(side="left", padx=(20, 0))
        self.min_market_var = tk.StringVar(value=str(int(settings.min_market_volume)))
        ttk.Spinbox(row2, from_=0, to=10000000, increment=10000, textvariable=self.min_market_var, width=10).pack(side="left", padx=5)

        # Preset buttons for quick filtering
        row3 = ttk.Frame(settings_frame)
        row3.pack(fill="x", padx=10, pady=5)

        ttk.Label(row3, text="Presets:").pack(side="left")
        ttk.Button(row3, text="Whales Only ($1k+)", command=lambda: self._set_filter_preset(1000, 100000)).pack(side="left", padx=5)
        ttk.Button(row3, text="High Volume ($100+)", command=lambda: self._set_filter_preset(100, 10000)).pack(side="left", padx=5)
        ttk.Button(row3, text="All Trades", command=lambda: self._set_filter_preset(0, 0)).pack(side="left", padx=5)

        # Log display
        log_frame = ttk.LabelFrame(self, text="Collection Log")
        log_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.log_text = tk.Text(log_frame, wrap="word", height=15, font=("Consolas", 9))
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)

        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

        self._log("Ready to collect data...")

    def _log(self, message: str):
        """Add message to log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.see("end")

    def _start_collection(self):
        """Start data collection."""
        if self.is_collecting:
            return

        self.is_collecting = True
        self.collect_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self._log("Starting data collection...")
        self.status_bar.set_status("Collecting data...")

        # Run collection in background
        self._run_single_collection()

    def _stop_collection(self):
        """Stop data collection."""
        self.is_collecting = False
        self.collect_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self._log("Collection stopped")
        self.status_bar.set_status("Collection stopped")

    def _run_single_collection(self):
        """Run a single collection cycle with current filter settings."""
        if not self.is_collecting:
            return

        min_trade_size = float(self.min_trade_var.get())
        min_market_volume = float(self.min_market_var.get())

        async def collect():
            collector = DataCollector()
            # Sync markets less frequently (handled internally)
            await collector.sync_markets()
            # Sync trades with filter settings
            await collector.sync_all_trades(
                min_trade_size=min_trade_size,
                min_market_volume=min_market_volume
            )
            await collector.close()

        def on_complete(result, error):
            if error:
                self._log(f"Error: {error}")
            else:
                self._log("Collection cycle complete")
                self.status_bar.update_db_stats()

            if self.is_collecting:
                interval = int(self.interval_var.get()) * 60 * 1000
                self.after(interval, self._run_single_collection)

        self.async_runner.run(collect(), on_complete)

    def _sync_markets(self):
        """Sync markets only."""
        self._log("Syncing markets...")
        self.status_bar.set_status("Syncing markets...")

        async def sync():
            collector = DataCollector()
            await collector.sync_markets()
            await collector.close()

        def on_complete(result, error):
            if error:
                self._log(f"Error syncing markets: {error}")
            else:
                self._log("Markets synced successfully")
                self.status_bar.update_db_stats()
            self.status_bar.set_status("Ready")

        self.async_runner.run(sync(), on_complete)

    def _set_filter_preset(self, min_trade: int, min_market: int):
        """Set filter values to a preset."""
        self.min_trade_var.set(str(min_trade))
        self.min_market_var.set(str(min_market))
        preset_name = "All Trades" if min_trade == 0 else f"${min_trade:,}+ trades"
        self._log(f"Filter preset applied: {preset_name}")

    def _sync_trades(self):
        """Sync trades only with current filter settings."""
        min_trade_size = float(self.min_trade_var.get())
        min_market_volume = float(self.min_market_var.get())

        self._log(f"Syncing trades (min trade: ${min_trade_size:,.0f}, min market vol: ${min_market_volume:,.0f})...")
        self.status_bar.set_status("Syncing trades...")

        async def sync():
            collector = DataCollector()
            await collector.sync_all_trades(
                min_trade_size=min_trade_size,
                min_market_volume=min_market_volume
            )
            await collector.close()

        def on_complete(result, error):
            if error:
                self._log(f"Error syncing trades: {error}")
            else:
                self._log("Trades synced successfully")
                self.status_bar.update_db_stats()
            self.status_bar.set_status("Ready")

        self.async_runner.run(sync(), on_complete)


class SettingsTab(ttk.Frame):
    """Application settings and configuration."""

    def __init__(self, parent, status_bar: StatusBar):
        super().__init__(parent)
        self.status_bar = status_bar

        self._create_widgets()

    def _create_widgets(self):
        """Create settings widgets."""
        # Database settings
        db_frame = ttk.LabelFrame(self, text="Database")
        db_frame.pack(fill="x", padx=10, pady=10)

        row = ttk.Frame(db_frame)
        row.pack(fill="x", padx=10, pady=5)

        ttk.Label(row, text="Database Path:").pack(side="left")
        self.db_path_var = tk.StringVar(value=str(settings.database_path))
        ttk.Entry(row, textvariable=self.db_path_var, width=50).pack(side="left", padx=5)
        ttk.Button(row, text="Browse...", command=self._browse_db).pack(side="left")

        # Alert settings
        alert_frame = ttk.LabelFrame(self, text="Alerts")
        alert_frame.pack(fill="x", padx=10, pady=10)

        # Discord
        discord_row = ttk.Frame(alert_frame)
        discord_row.pack(fill="x", padx=10, pady=5)

        ttk.Label(discord_row, text="Discord Webhook:").pack(side="left")
        self.discord_var = tk.StringVar()
        ttk.Entry(discord_row, textvariable=self.discord_var, width=50, show="*").pack(side="left", padx=5)

        # Email
        email_row = ttk.Frame(alert_frame)
        email_row.pack(fill="x", padx=10, pady=5)

        ttk.Label(email_row, text="Alert Email:").pack(side="left")
        self.email_var = tk.StringVar()
        ttk.Entry(email_row, textvariable=self.email_var, width=30).pack(side="left", padx=5)

        # Threshold
        thresh_row = ttk.Frame(alert_frame)
        thresh_row.pack(fill="x", padx=10, pady=5)

        ttk.Label(thresh_row, text="Alert Score Threshold:").pack(side="left")
        self.threshold_var = tk.StringVar(value="80")
        ttk.Spinbox(thresh_row, from_=0, to=100, textvariable=self.threshold_var, width=5).pack(side="left", padx=5)

        # Export settings
        export_frame = ttk.LabelFrame(self, text="Export")
        export_frame.pack(fill="x", padx=10, pady=10)

        export_row = ttk.Frame(export_frame)
        export_row.pack(fill="x", padx=10, pady=5)

        ttk.Button(export_row, text="📁 Export Leaderboard (CSV)",
                   command=self._export_csv).pack(side="left", padx=5)
        ttk.Button(export_row, text="📋 Export Leaderboard (JSON)",
                   command=self._export_json).pack(side="left", padx=5)

        # Database maintenance
        maint_frame = ttk.LabelFrame(self, text="Maintenance")
        maint_frame.pack(fill="x", padx=10, pady=10)

        maint_row = ttk.Frame(maint_frame)
        maint_row.pack(fill="x", padx=10, pady=5)

        ttk.Button(maint_row, text="🗑️ Clear Old Data",
                   command=self._clear_old_data).pack(side="left", padx=5)
        ttk.Button(maint_row, text="🔧 Vacuum Database",
                   command=self._vacuum_db).pack(side="left", padx=5)
        ttk.Button(maint_row, text="📊 Recalculate Stats",
                   command=self._recalc_stats).pack(side="left", padx=5)

    def _browse_db(self):
        """Browse for database file."""
        path = filedialog.askopenfilename(
            filetypes=[("SQLite Database", "*.db"), ("All Files", "*.*")]
        )
        if path:
            self.db_path_var.set(path)

    def _export_csv(self):
        """Export leaderboard to CSV."""
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")]
        )
        if path:
            try:
                from .export import export_leaderboard_csv
                export_leaderboard_csv(path)
                messagebox.showinfo("Export", f"Exported to {path}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")

    def _export_json(self):
        """Export leaderboard to JSON."""
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")]
        )
        if path:
            try:
                from .export import export_leaderboard_json
                export_leaderboard_json(path)
                messagebox.showinfo("Export", f"Exported to {path}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")

    def _clear_old_data(self):
        """Clear old data from database."""
        result = messagebox.askyesno(
            "Confirm",
            "This will delete bets older than 90 days. Continue?"
        )
        if result:
            try:
                cutoff = datetime.now() - timedelta(days=90)
                with db.get_connection() as conn:
                    conn.execute(
                        "DELETE FROM bets WHERE timestamp < ?",
                        (cutoff.isoformat(),)
                    )
                    conn.commit()
                messagebox.showinfo("Success", "Old data cleared")
                self.status_bar.update_db_stats()
            except Exception as e:
                messagebox.showerror("Error", f"Failed: {e}")

    def _vacuum_db(self):
        """Vacuum database to reclaim space."""
        try:
            with db.get_connection() as conn:
                conn.execute("VACUUM")
            messagebox.showinfo("Success", "Database vacuumed")
        except Exception as e:
            messagebox.showerror("Error", f"Failed: {e}")

    def _recalc_stats(self):
        """Recalculate all trader statistics."""
        self.status_bar.set_status("Recalculating stats...")
        try:
            analytics = PolymarketAnalytics()
            with db.get_connection() as conn:
                cursor = conn.execute("SELECT DISTINCT wallet_address FROM bets")
                wallets = [row["wallet_address"] for row in cursor.fetchall()]

            for wallet in wallets:
                analytics.get_trader_stats(wallet, use_cache=False)

            messagebox.showinfo("Success", f"Recalculated stats for {len(wallets)} traders")
            self.status_bar.set_status("Ready")
        except Exception as e:
            messagebox.showerror("Error", f"Failed: {e}")


class PolymarketTrackerApp(tk.Tk):
    """Main application window."""

    def __init__(self):
        super().__init__()

        self.title("Polymarket Tracker")
        self.geometry("1200x800")
        self.minsize(800, 600)

        # Set up async runner
        self.async_runner = AsyncRunner()
        self.async_runner.start()

        # Apply theme
        self.style = ttk.Style()
        self._setup_theme()

        self._create_widgets()

        # Handle close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_theme(self):
        """Set up application theme."""
        # Use clam theme for consistent look
        self.style.theme_use("clam")

        # Configure colors
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TLabel", background="#f0f0f0")
        self.style.configure("TLabelframe", background="#f0f0f0")
        self.style.configure("TLabelframe.Label", background="#f0f0f0", font=("Arial", 10, "bold"))
        self.style.configure("TNotebook.Tab", padding=[10, 5])

    def _create_widgets(self):
        """Create main application widgets."""
        # Status bar at bottom
        self.status_bar = StatusBar(self)
        self.status_bar.pack(side="bottom", fill="x")

        # Notebook for tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Create tabs
        self.dashboard_tab = DashboardTab(self.notebook, self.status_bar)
        self.notebook.add(self.dashboard_tab, text="📊 Dashboard")

        self.live_feed_tab = LiveFeedTab(self.notebook, self.status_bar)
        self.notebook.add(self.live_feed_tab, text="📈 Live Feed")

        self.insider_tab = InsiderTab(self.notebook, self.status_bar)
        self.notebook.add(self.insider_tab, text="🚨 Insider Detection")

        self.collector_tab = CollectorTab(self.notebook, self.status_bar, self.async_runner)
        self.notebook.add(self.collector_tab, text="📥 Data Collection")

        self.settings_tab = SettingsTab(self.notebook, self.status_bar)
        self.notebook.add(self.settings_tab, text="⚙️ Settings")

        # Bind tab change
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_change)

    def _on_tab_change(self, event):
        """Handle tab change."""
        tab_id = self.notebook.select()
        tab_name = self.notebook.tab(tab_id, "text")
        self.status_bar.set_status(f"Viewing: {tab_name}")

    def _on_close(self):
        """Handle application close."""
        self.async_runner.stop()
        self.destroy()


def main():
    """Main entry point for the GUI."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create and run app
    app = PolymarketTrackerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
