# Whale Signal Dashboard — Design Document

**Date:** 2026-02-27
**Goal:** Detect strange large bets on Polymarket pools and surface them as actionable signals via a live web dashboard.

## Problem

The Polymarket Tracker already collects trades, calculates PNL, and scores bets for insider-trading suspicion. But the anomaly detection is forensic — it flags suspicious activity after the fact. The goal is to turn detected anomalies into **forward-looking trading signals**: when a whale or cluster of wallets makes an unusual bet, surface it in real time so the user can evaluate and potentially act on it.

## Approach

**Approach A: Signal Layer** — Add a signal scoring engine on top of the existing detection system, plus a FastAPI web dashboard with WebSocket push. Builds on the existing Python/SQLite stack with minimal new dependencies.

---

## 1. Whale Wallet Identification & Profiling

### Classification Criteria

A wallet becomes a "whale" when:
- **Cumulative volume** exceeds $50k (configurable), OR
- **Any single bet** exceeds $5k

### Whale Profile Data (new `whale_profiles` table)

| Field | Description |
|-------|-------------|
| wallet_address | Primary key, FK to traders |
| grade | A/B/C/D based on resolved-market win rate (min 20 resolved bets) |
| win_rate | Win rate on resolved markets only (decisive outcomes) |
| roi | Return on investment |
| sharpe_ratio | Risk-adjusted returns |
| category_specialization | JSON — win rates broken down by market category |
| avg_bet_timing | Average hours before resolution when bets are placed |
| activity_pattern | "steady" / "dormant_burst" / "sporadic" |
| total_volume | Cumulative USD volume |
| total_resolved_bets | Number of resolved decisive bets |
| last_updated | Timestamp |

### Data Accuracy Safeguards

- Win rates count only fully resolved markets (no open, tied, or cancelled)
- Grades require minimum 20 resolved decisive bets; otherwise "ungraded"
- Stale wallets (inactive 90+ days) get flagged, not removed
- Profiles recalculated every collection cycle

### Grading Scale

| Grade | Win Rate | Min Bets |
|-------|----------|----------|
| A | >= 65% | 20 |
| B | >= 55% | 20 |
| C | >= 45% | 20 |
| D | < 45% | 20 |
| Ungraded | Any | < 20 |

---

## 2. Signal Engine

### Signal Types

Three independent detectors feed into a composite conviction score:

#### Smart Money Signal (40% weight)
- Fires when a graded whale (B or above) takes a position
- Score weighted by: wallet grade, bet size relative to their average, historical category accuracy
- Higher weight for A-grade wallets betting in their specialty category

#### Volume Spike Signal (30% weight)
- Fires when a single bet or burst of bets exceeds 3x the market's rolling average bet size within a 1-hour window
- Rolling average computed over the last 7 days of trading activity
- Normalized by market liquidity to avoid false positives on thin markets

#### Coordinated Cluster Signal (30% weight)
- Fires when 3+ distinct wallets bet the same outcome within a 30-minute window
- Reuses existing correlated betting detection from `insider_detection.py`
- Enhanced with Sybil similarity check to flag potential wash trading (discount signal if wallets are likely same entity)

### Composite Conviction Score

```
conviction = (smart_money_score * 0.4) + (volume_spike_score * 0.3) + (cluster_score * 0.3)
```

- Range: 0-100
- Signal fires when score >= 60 (configurable threshold)
- Multiple signal types converging amplifies the score

### Storage (new `signals` table)

| Field | Description |
|-------|-------------|
| signal_id | Auto-increment PK |
| market_id | FK to markets |
| outcome | Which outcome the signal points to |
| conviction_score | Composite score 0-100 |
| smart_money_score | Component score |
| volume_spike_score | Component score |
| cluster_score | Component score |
| contributing_wallets | JSON array of wallet addresses involved |
| details | JSON — full breakdown of why signal fired |
| timestamp | When signal was generated |
| market_resolved | Boolean — filled in after resolution |
| signal_correct | Boolean — did the signaled outcome win? (for future backtesting) |

---

## 3. Web Dashboard

### Stack

- **Backend:** FastAPI + Jinja2 templates + WebSocket
- **Frontend:** HTMX for interactivity, minimal JS, no heavy framework
- **Entry point:** `python -m polymarket_tracker.web` on port 8080

### Pages

#### Live Feed (`/`)
- Real-time signal stream via WebSocket
- Each signal card shows: market question, outcome, conviction score, signal type breakdown, contributing whale wallets (linked to profiles)
- Color-coded by conviction: green (60-74), yellow (75-89), red (90+)
- Filterable by minimum score, category, signal type

#### Hot Markets (`/markets`)
- Markets ranked by recent signal activity (last 24h)
- Columns: market question, category, total signals, highest conviction, end date
- Filterable by category, minimum score, time range
- Click through to market detail

#### Whale Directory (`/whales`)
- All tracked whales with grade, win rate, volume, specialization
- Sortable by grade, volume, win rate, recent activity
- Search by wallet address
- Click through to whale profile

#### Whale Profile (`/whales/<address>`)
- Full wallet history: all bets, PNL, win rate by category
- Activity timeline
- Recent signals this wallet contributed to

#### Market Detail (`/markets/<id>`)
- All bets on this market, whale bets highlighted
- Signal history for this market
- Current prices and volume

---

## 4. Integration with Existing System

- Signal engine runs as part of the existing collection cycle (every 5 minutes)
- After new trades are collected, the signal engine evaluates them
- Whale profiles are recalculated on a slower cadence (every 30 minutes, alongside market sync)
- Web dashboard is a separate process that reads from the same SQLite database
- Existing Discord/email alerts can optionally forward signals too

## 5. Future Iteration (Not in Scope Now)

- Backtesting: Track signal accuracy over time, show historical hit rates
- Signal strength calibration based on backtested data
- Telegram bot for mobile queries
- Auto-trade integration
