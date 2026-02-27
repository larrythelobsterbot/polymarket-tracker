"""
FastAPI web dashboard for Polymarket Whale Tracker.

Provides a dark-themed web UI for viewing whale trading signals,
hot markets, and whale profiles with real-time updates via WebSocket.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..database import db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Polymarket Whale Tracker", docs_url="/docs")

BASE_DIR = Path(__file__).parent

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# ---------------------------------------------------------------------------
# Jinja2 template filters
# ---------------------------------------------------------------------------


def usd_filter(value) -> str:
    """Format a number as a compact USD string."""
    try:
        n = float(value)
    except (TypeError, ValueError):
        return "$0"
    if abs(n) >= 1_000_000:
        return f"${n / 1_000_000:.1f}M"
    if abs(n) >= 1_000:
        return f"${n / 1_000:.1f}k"
    return f"${n:,.0f}"


def address_filter(value) -> str:
    """Truncate a wallet address to '0xabcd...ef12' format."""
    if not value or len(str(value)) < 10:
        return str(value) if value else ""
    s = str(value)
    return f"{s[:6]}...{s[-4:]}"


def time_ago_filter(value) -> str:
    """Format a datetime (or ISO string) as a human-readable 'X ago' string."""
    if value is None:
        return ""
    try:
        if isinstance(value, str):
            # Handle common SQLite timestamp formats
            value = value.replace("Z", "+00:00")
            try:
                dt = datetime.fromisoformat(value)
            except ValueError:
                # Try parsing without timezone
                dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        elif isinstance(value, datetime):
            dt = value
        else:
            return str(value)

        # Make comparison timezone-naive
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)

        now = datetime.utcnow()
        delta = now - dt
        total_seconds = int(delta.total_seconds())

        if total_seconds < 0:
            return "just now"
        if total_seconds < 60:
            return f"{total_seconds}s ago"

        minutes = total_seconds // 60
        if minutes < 60:
            return f"{minutes}m ago"

        hours = minutes // 60
        if hours < 24:
            return f"{hours}h ago"

        days = hours // 24
        return f"{days}d ago"
    except Exception:
        return str(value)


# Register filters
templates.env.filters["usd"] = usd_filter
templates.env.filters["address"] = address_filter
templates.env.filters["time_ago"] = time_ago_filter

# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------


class ConnectionManager:
    """Manages active WebSocket connections for real-time signal broadcasting."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict) -> None:
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


manager = ConnectionManager()

# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    min_score: float = Query(default=0, ge=0, le=100),
):
    """Live signal feed page."""
    signals = db.get_recent_signals(limit=50, min_score=min_score)
    stats = db.get_stats()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "page": "feed",
            "signals": signals,
            "stats": stats,
            "min_score": min_score,
        },
    )


@app.get("/markets", response_class=HTMLResponse)
async def markets(
    request: Request,
    hours: int = Query(default=24, ge=1, le=168),
):
    """Hot markets page."""
    hot_markets = db.get_hot_markets(hours=hours)
    return templates.TemplateResponse(
        "markets.html",
        {
            "request": request,
            "page": "markets",
            "markets": hot_markets,
            "hours": hours,
        },
    )


@app.get("/whales", response_class=HTMLResponse)
async def whales(
    request: Request,
    grade: Optional[str] = Query(default=None),
):
    """Whale directory page."""
    profiles = db.get_whale_profiles(min_grade=grade)
    return templates.TemplateResponse(
        "whales.html",
        {
            "request": request,
            "page": "whales",
            "whales": profiles,
            "grade": grade,
        },
    )


@app.get("/whales/{address}", response_class=HTMLResponse)
async def whale_detail(request: Request, address: str):
    """Individual whale profile page."""
    profile = db.get_whale_profile(address)
    bets = db.get_bets_for_wallet(address, limit=50)
    return templates.TemplateResponse(
        "whale_detail.html",
        {
            "request": request,
            "page": "whales",
            "profile": profile,
            "bets": bets,
            "address": address,
        },
    )


@app.get("/markets/{market_id}", response_class=HTMLResponse)
async def market_detail(request: Request, market_id: str):
    """Market detail page with signals and bets."""
    market = db.get_market(market_id)
    bets = db.get_bets_for_market(market_id, limit=100)
    signals = db.get_signals_for_market(market_id)
    return templates.TemplateResponse(
        "market_detail.html",
        {
            "request": request,
            "page": "markets",
            "market": market,
            "bets": bets,
            "signals": signals,
            "market_id": market_id,
        },
    )


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------


@app.get("/api/signals")
async def api_signals(
    limit: int = Query(default=20, ge=1, le=200),
    min_score: float = Query(default=0, ge=0, le=100),
):
    """JSON endpoint for HTMX polling — returns recent signals."""
    return db.get_recent_signals(limit=limit, min_score=min_score)


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------


@app.websocket("/ws/signals")
async def ws_signals(websocket: WebSocket):
    """WebSocket endpoint for real-time signal streaming."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive; wait for client messages (pings, etc.)
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
