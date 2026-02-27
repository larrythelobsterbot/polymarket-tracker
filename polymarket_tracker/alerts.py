"""
Alert System for Polymarket Insider Detection.

This module provides alerting capabilities for suspicious activity:
- Discord webhook integration
- Email notifications
- Alert rate limiting and batching
- Alert history and deduplication
"""

import asyncio
import hashlib
import json
import logging
import smtplib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import httpx

from .config import settings
from .insider_detection import (
    SuspiciousBet,
    WalletRiskProfile,
    MarketManipulationAlert,
    CorrelatedBetGroup,
    AlertSeverity,
)

logger = logging.getLogger(__name__)


@dataclass
class AlertConfig:
    """Configuration for alert system."""

    # Discord settings
    discord_webhook_url: str = ""
    discord_enabled: bool = False

    # Email settings
    email_enabled: bool = False
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    email_from: str = ""
    email_to: list = field(default_factory=list)

    # Rate limiting
    min_alert_interval_seconds: int = 60  # Minimum time between alerts
    batch_alerts: bool = True
    batch_window_seconds: int = 300  # 5 minutes

    # Filtering
    min_score_for_alert: int = 80
    alert_on_critical_only: bool = False

    # Deduplication
    dedup_window_hours: int = 24


class AlertDeduplicator:
    """Prevents duplicate alerts for the same event."""

    def __init__(self, window_hours: int = 24):
        self.window_hours = window_hours
        self.sent_alerts: dict[str, datetime] = {}

    def _generate_key(self, alert_type: str, identifier: str) -> str:
        """Generate unique key for an alert."""
        raw = f"{alert_type}:{identifier}"
        return hashlib.md5(raw.encode()).hexdigest()

    def should_send(self, alert_type: str, identifier: str) -> bool:
        """Check if alert should be sent (not duplicate)."""
        key = self._generate_key(alert_type, identifier)
        now = datetime.utcnow()

        # Clean old entries
        cutoff = now - timedelta(hours=self.window_hours)
        self.sent_alerts = {k: v for k, v in self.sent_alerts.items() if v > cutoff}

        if key in self.sent_alerts:
            return False

        self.sent_alerts[key] = now
        return True


class DiscordAlerter:
    """Send alerts to Discord via webhook."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def _severity_color(self, severity: AlertSeverity) -> int:
        """Get Discord embed color for severity."""
        colors = {
            AlertSeverity.LOW: 0x3498DB,      # Blue
            AlertSeverity.MEDIUM: 0xF39C12,   # Orange
            AlertSeverity.HIGH: 0xE74C3C,     # Red
            AlertSeverity.CRITICAL: 0x9B59B6,  # Purple
        }
        return colors.get(severity, 0x95A5A6)

    def _severity_emoji(self, severity: AlertSeverity) -> str:
        """Get emoji for severity level."""
        emojis = {
            AlertSeverity.LOW: "",
            AlertSeverity.MEDIUM: "",
            AlertSeverity.HIGH: "",
            AlertSeverity.CRITICAL: "",
        }
        return emojis.get(severity, "")

    async def send_suspicious_bet_alert(self, bet: SuspiciousBet) -> bool:
        """
        Send alert for suspicious bet.

        Args:
            bet: Suspicious bet to alert on.

        Returns:
            True if sent successfully.
        """
        emoji = self._severity_emoji(bet.severity)
        color = self._severity_color(bet.severity)

        embed = {
            "title": f"{emoji} Suspicious Bet Detected",
            "color": color,
            "fields": [
                {
                    "name": "Anomaly Score",
                    "value": f"**{bet.anomaly_score:.1f}/100**",
                    "inline": True,
                },
                {
                    "name": "Severity",
                    "value": bet.severity.value.upper(),
                    "inline": True,
                },
                {
                    "name": "Wallet",
                    "value": f"`{bet.wallet_address[:10]}...{bet.wallet_address[-6:]}`",
                    "inline": True,
                },
                {
                    "name": "Amount",
                    "value": f"${bet.amount:,.2f}",
                    "inline": True,
                },
                {
                    "name": "Market",
                    "value": bet.market_question[:100] if bet.market_question else bet.market_id[:20],
                    "inline": False,
                },
                {
                    "name": "Bet",
                    "value": f"{bet.side} {bet.outcome_bet} @ ${bet.price:.4f}",
                    "inline": True,
                },
            ],
            "footer": {
                "text": f"Bet ID: {bet.bet_id[:16]}... | {bet.timestamp.strftime('%Y-%m-%d %H:%M UTC') if bet.timestamp else 'N/A'}",
            },
        }

        # Add alert types
        if bet.alert_types:
            embed["fields"].append({
                "name": "Flags",
                "value": ", ".join(a.value.replace("_", " ").title() for a in bet.alert_types),
                "inline": False,
            })

        # Add score breakdown
        if bet.score_breakdown:
            breakdown = "\n".join(f"- {k}: +{v}" for k, v in bet.score_breakdown.items())
            embed["fields"].append({
                "name": "Score Breakdown",
                "value": f"```\n{breakdown}\n```",
                "inline": False,
            })

        if bet.hours_to_resolution is not None and bet.hours_to_resolution > 0:
            embed["fields"].append({
                "name": "Time to Resolution",
                "value": f"{bet.hours_to_resolution:.1f} hours",
                "inline": True,
            })

        payload = {"embeds": [embed]}

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                return response.status_code == 204
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
            return False

    async def send_wallet_risk_alert(self, profile: WalletRiskProfile) -> bool:
        """Send alert for high-risk wallet."""
        severity = AlertSeverity.CRITICAL if profile.overall_risk_score >= 80 else AlertSeverity.HIGH

        embed = {
            "title": f"{self._severity_emoji(severity)} High-Risk Wallet Identified",
            "color": self._severity_color(severity),
            "fields": [
                {
                    "name": "Wallet",
                    "value": f"`{profile.wallet_address[:10]}...{profile.wallet_address[-6:]}`",
                    "inline": False,
                },
                {
                    "name": "Risk Score",
                    "value": f"**{profile.overall_risk_score:.1f}/100**",
                    "inline": True,
                },
                {
                    "name": "Suspicious Bets",
                    "value": f"{profile.total_suspicious_bets}/{profile.total_bets_analyzed}",
                    "inline": True,
                },
                {
                    "name": "Total Volume",
                    "value": f"${profile.total_volume:,.2f}",
                    "inline": True,
                },
            ],
            "footer": {
                "text": f"First seen: {profile.first_seen.strftime('%Y-%m-%d') if profile.first_seen else 'N/A'}",
            },
        }

        if profile.flags:
            embed["fields"].append({
                "name": "Flags",
                "value": ", ".join(f.replace("_", " ").title() for f in profile.flags),
                "inline": False,
            })

        if profile.associated_wallets:
            embed["fields"].append({
                "name": "Potential Sybil Connections",
                "value": str(len(profile.associated_wallets)) + " wallets",
                "inline": True,
            })

        payload = {"embeds": [embed]}

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                return response.status_code == 204
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
            return False

    async def send_correlation_alert(self, group: CorrelatedBetGroup) -> bool:
        """Send alert for correlated betting activity."""
        embed = {
            "title": " Correlated Betting Detected",
            "color": 0xE74C3C,
            "fields": [
                {
                    "name": "Market",
                    "value": group.market_id[:30],
                    "inline": False,
                },
                {
                    "name": "Outcome",
                    "value": group.outcome.upper(),
                    "inline": True,
                },
                {
                    "name": "Wallets Involved",
                    "value": str(len(group.wallets)),
                    "inline": True,
                },
                {
                    "name": "Total Amount",
                    "value": f"${group.total_amount:,.2f}",
                    "inline": True,
                },
                {
                    "name": "Time Window",
                    "value": f"{(group.timestamp_end - group.timestamp_start).total_seconds() / 60:.1f} minutes",
                    "inline": True,
                },
                {
                    "name": "Correlation Score",
                    "value": f"{group.correlation_score:.1f}/100",
                    "inline": True,
                },
            ],
            "footer": {
                "text": f"Detected at {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            },
        }

        payload = {"embeds": [embed]}

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                return response.status_code == 204
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
            return False

    async def send_market_manipulation_alert(self, alert: MarketManipulationAlert) -> bool:
        """Send alert for market manipulation."""
        embed = {
            "title": " Market Manipulation Alert",
            "color": self._severity_color(alert.severity),
            "fields": [
                {
                    "name": "Market",
                    "value": alert.market_question or alert.market_id[:30],
                    "inline": False,
                },
                {
                    "name": "Alert Type",
                    "value": alert.alert_type.replace("_", " ").title(),
                    "inline": True,
                },
                {
                    "name": "Severity",
                    "value": alert.severity.value.upper(),
                    "inline": True,
                },
            ],
            "footer": {
                "text": f"Detected at {alert.timestamp.strftime('%Y-%m-%d %H:%M UTC')}",
            },
        }

        # Add details
        for key, value in alert.details.items():
            embed["fields"].append({
                "name": key.replace("_", " ").title(),
                "value": str(value),
                "inline": True,
            })

        payload = {"embeds": [embed]}

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                return response.status_code == 204
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
            return False


class EmailAlerter:
    """Send alerts via email."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_addr: str,
        to_addrs: list[str]
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs

    def _create_bet_alert_html(self, bet: SuspiciousBet) -> str:
        """Create HTML email content for suspicious bet."""
        severity_colors = {
            AlertSeverity.LOW: "#3498DB",
            AlertSeverity.MEDIUM: "#F39C12",
            AlertSeverity.HIGH: "#E74C3C",
            AlertSeverity.CRITICAL: "#9B59B6",
        }
        color = severity_colors.get(bet.severity, "#95A5A6")

        flags_html = ", ".join(a.value.replace("_", " ").title() for a in bet.alert_types)
        breakdown_html = "<br>".join(f"- {k}: +{v}" for k, v in bet.score_breakdown.items())

        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background-color: {color}; color: white; padding: 15px; border-radius: 5px 5px 0 0;">
                <h2 style="margin: 0;">Suspicious Bet Detected</h2>
                <p style="margin: 5px 0 0 0;">Severity: {bet.severity.value.upper()}</p>
            </div>
            <div style="border: 1px solid #ddd; padding: 20px; border-radius: 0 0 5px 5px;">
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>Anomaly Score</strong></td>
                        <td style="padding: 8px; border-bottom: 1px solid #eee;">{bet.anomaly_score:.1f}/100</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>Wallet</strong></td>
                        <td style="padding: 8px; border-bottom: 1px solid #eee;"><code>{bet.wallet_address}</code></td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>Amount</strong></td>
                        <td style="padding: 8px; border-bottom: 1px solid #eee;">${bet.amount:,.2f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>Market</strong></td>
                        <td style="padding: 8px; border-bottom: 1px solid #eee;">{bet.market_question or bet.market_id}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>Bet</strong></td>
                        <td style="padding: 8px; border-bottom: 1px solid #eee;">{bet.side} {bet.outcome_bet} @ ${bet.price:.4f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>Flags</strong></td>
                        <td style="padding: 8px; border-bottom: 1px solid #eee;">{flags_html}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>Score Breakdown</strong></td>
                        <td style="padding: 8px; border-bottom: 1px solid #eee;">{breakdown_html}</td>
                    </tr>
                </table>
                <p style="color: #666; font-size: 12px; margin-top: 20px;">
                    Bet ID: {bet.bet_id}<br>
                    Timestamp: {bet.timestamp.strftime('%Y-%m-%d %H:%M UTC') if bet.timestamp else 'N/A'}
                </p>
            </div>
        </body>
        </html>
        """

    def send_suspicious_bet_alert(self, bet: SuspiciousBet) -> bool:
        """Send email alert for suspicious bet."""
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[{bet.severity.value.upper()}] Polymarket Suspicious Bet - Score {bet.anomaly_score:.0f}"
            msg["From"] = self.from_addr
            msg["To"] = ", ".join(self.to_addrs)

            html_content = self._create_bet_alert_html(bet)
            msg.attach(MIMEText(html_content, "html"))

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.from_addr, self.to_addrs, msg.as_string())

            return True
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False


class AlertManager:
    """
    Manages all alert channels and coordination.

    Handles:
    - Rate limiting
    - Batching
    - Deduplication
    - Multi-channel delivery
    """

    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or AlertConfig()
        self.deduplicator = AlertDeduplicator(self.config.dedup_window_hours)
        self.last_alert_time: Optional[datetime] = None
        self.pending_alerts: list = []

        # Initialize alerters
        self.discord: Optional[DiscordAlerter] = None
        self.email: Optional[EmailAlerter] = None

        if self.config.discord_enabled and self.config.discord_webhook_url:
            self.discord = DiscordAlerter(self.config.discord_webhook_url)

        if self.config.email_enabled and self.config.smtp_username:
            self.email = EmailAlerter(
                smtp_host=self.config.smtp_host,
                smtp_port=self.config.smtp_port,
                username=self.config.smtp_username,
                password=self.config.smtp_password,
                from_addr=self.config.email_from,
                to_addrs=self.config.email_to,
            )

    def _should_rate_limit(self) -> bool:
        """Check if we should rate limit alerts."""
        if not self.last_alert_time:
            return False

        elapsed = (datetime.utcnow() - self.last_alert_time).total_seconds()
        return elapsed < self.config.min_alert_interval_seconds

    async def send_bet_alert(self, bet: SuspiciousBet) -> bool:
        """
        Send alert for suspicious bet.

        Args:
            bet: Suspicious bet to alert on.

        Returns:
            True if alert was sent or queued.
        """
        # Check score threshold
        if bet.anomaly_score < self.config.min_score_for_alert:
            return False

        # Check severity filter
        if self.config.alert_on_critical_only and bet.severity != AlertSeverity.CRITICAL:
            return False

        # Check deduplication
        if not self.deduplicator.should_send("bet", bet.bet_id):
            logger.debug(f"Duplicate alert suppressed for bet {bet.bet_id}")
            return False

        # Check rate limiting
        if self._should_rate_limit():
            if self.config.batch_alerts:
                self.pending_alerts.append(("bet", bet))
                return True
            else:
                logger.debug("Alert rate limited")
                return False

        # Send to all channels
        sent = False

        if self.discord:
            try:
                sent = await self.discord.send_suspicious_bet_alert(bet) or sent
            except Exception as e:
                logger.error(f"Discord alert failed: {e}")

        if self.email:
            try:
                sent = self.email.send_suspicious_bet_alert(bet) or sent
            except Exception as e:
                logger.error(f"Email alert failed: {e}")

        if sent:
            self.last_alert_time = datetime.utcnow()

        return sent

    async def send_wallet_alert(self, profile: WalletRiskProfile) -> bool:
        """Send alert for high-risk wallet."""
        if profile.overall_risk_score < self.config.min_score_for_alert:
            return False

        if not self.deduplicator.should_send("wallet", profile.wallet_address):
            return False

        sent = False
        if self.discord:
            try:
                sent = await self.discord.send_wallet_risk_alert(profile)
            except Exception as e:
                logger.error(f"Discord wallet alert failed: {e}")

        return sent

    async def send_correlation_alert(self, group: CorrelatedBetGroup) -> bool:
        """Send alert for correlated betting."""
        key = f"{group.market_id}_{group.outcome}_{len(group.wallets)}"
        if not self.deduplicator.should_send("correlation", key):
            return False

        sent = False
        if self.discord:
            try:
                sent = await self.discord.send_correlation_alert(group)
            except Exception as e:
                logger.error(f"Discord correlation alert failed: {e}")

        return sent

    async def send_manipulation_alert(self, alert: MarketManipulationAlert) -> bool:
        """Send market manipulation alert."""
        if not self.deduplicator.should_send("manipulation", f"{alert.market_id}_{alert.alert_type}"):
            return False

        sent = False
        if self.discord:
            try:
                sent = await self.discord.send_market_manipulation_alert(alert)
            except Exception as e:
                logger.error(f"Discord manipulation alert failed: {e}")

        return sent

    async def flush_pending(self) -> int:
        """
        Send all pending batched alerts.

        Returns:
            Number of alerts sent.
        """
        if not self.pending_alerts:
            return 0

        sent_count = 0
        for alert_type, alert_data in self.pending_alerts:
            try:
                if alert_type == "bet":
                    if await self.send_bet_alert(alert_data):
                        sent_count += 1
                # Add other types as needed
            except Exception as e:
                logger.error(f"Failed to send batched alert: {e}")

        self.pending_alerts.clear()
        return sent_count


# Global alert manager (configured via environment)
alert_manager = AlertManager()
