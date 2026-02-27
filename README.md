# Polymarket Tracker

A data collection and storage system for Polymarket trading data. This system fetches market information, trades, and trader statistics from the Polymarket APIs and stores them in a local SQLite database.

## Features

- **Market Tracking**: Fetches and stores all active prediction markets
- **Trade Collection**: Captures trades across markets with incremental syncing
- **Trader Statistics**: Tracks wallet addresses, trading volumes, and activity
- **Position Tracking**: Stores current positions for analysis
- **Rate Limiting**: Built-in rate limiting to comply with API limits
- **Scheduled Collection**: Automatic data collection every 5 minutes (configurable)
- **Error Handling**: Retry logic and graceful error recovery
- **Leaderboard System**: Rank traders by PNL, volume, win rate, and ROI
- **PNL Calculations**: Realized and unrealized profit/loss with fee accounting
- **CLI Dashboard**: Interactive terminal dashboard with auto-refresh
- **Export Functionality**: Export leaderboards to JSON and CSV formats
- **Insider Detection**: Anomaly scoring system to detect suspicious trading patterns
- **Alert System**: Discord webhooks and email notifications for suspicious activity
- **ML Classifier**: Optional machine learning-based insider detection

## Architecture

```
polymarket_tracker/
├── __init__.py          # Package initialization
├── config.py            # Configuration management (env vars)
├── models.py            # Pydantic models for API data
├── database.py          # SQLite database operations
├── api_client.py        # Polymarket API client with rate limiting
├── collector.py         # Data collection service
├── analytics.py         # PNL calculations and leaderboard ranking
├── dashboard.py         # CLI dashboard with auto-refresh
├── export.py            # JSON and CSV export functionality
├── sample_data.py       # Sample data generator for testing
├── insider_detection.py # Anomaly detection and scoring engine
├── alerts.py            # Discord and email alert system
├── insider_dashboard.py # Suspicious activity CLI dashboard
├── ml_detector.py       # Optional ML-based insider detection
└── main.py              # CLI entry point and scheduler
```

## Database Schema

### Tables

- **traders**: Wallet addresses with volume and activity statistics
- **markets**: Prediction market metadata (question, end date, outcome)
- **bets**: Individual trade records
- **positions**: Current positions per wallet/market
- **sync_state**: Tracks last sync timestamps for incremental updates

## Installation

### Using Poetry (Recommended)

```bash
# Clone or navigate to the project directory
cd "Polymarket Tracker"

# Install dependencies with Poetry
poetry install

# Activate the virtual environment
poetry shell
```

### Using pip

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your settings:
   ```env
   # Database path (default: current directory)
   DATABASE_PATH=polymarket_data.db

   # Collection interval in minutes
   FETCH_INTERVAL_MINUTES=5

   # Logging level
   LOG_LEVEL=INFO
   ```

## Usage

### Launch the GUI Application

The easiest way to use Polymarket Tracker is through the graphical interface:

```bash
python -m polymarket_tracker.gui
```

The GUI includes:
- **Dashboard Tab**: View trader leaderboards, sort by PNL/volume/win rate, and see detailed trader stats
- **Insider Detection Tab**: Monitor suspicious betting activity with anomaly scores
- **Data Collection Tab**: Start/stop data collection, sync markets and trades
- **Settings Tab**: Configure alerts, export data, and manage the database

### Run Continuous Collection (CLI)

Start the tracker to collect data every 5 minutes:

```bash
python -m polymarket_tracker.main
```

Or with Poetry:

```bash
poetry run python -m polymarket_tracker.main
```

### Run Single Collection Cycle

Fetch data once and exit:

```bash
python -m polymarket_tracker.main --once
```

### View Database Statistics

```bash
python -m polymarket_tracker.main --stats
```

### Sync Only Markets

```bash
python -m polymarket_tracker.main --sync-markets
```

### Sync Only Trades

```bash
python -m polymarket_tracker.main --sync-trades
```

## Leaderboard Dashboard

### Run Interactive Dashboard

Launch the dashboard with auto-refresh every 30 seconds:

```bash
python -m polymarket_tracker.dashboard
```

### Dashboard Options

```bash
# Custom refresh interval (60 seconds)
python -m polymarket_tracker.dashboard --refresh 60

# Show top 10 traders
python -m polymarket_tracker.dashboard --limit 10

# Display once without auto-refresh
python -m polymarket_tracker.dashboard --no-refresh

# Show only PNL leaderboard
python -m polymarket_tracker.dashboard --pnl-only

# Show only Volume leaderboard
python -m polymarket_tracker.dashboard --volume-only

# Sort by different metrics
python -m polymarket_tracker.dashboard --metric win_rate
python -m polymarket_tracker.dashboard --metric roi
python -m polymarket_tracker.dashboard --metric trade_count
```

### Trader Lookup

Get detailed statistics for a specific trader:

```bash
python -m polymarket_tracker.dashboard --trader 0x1234567890abcdef...
```

### Export Leaderboards

Export to JSON and CSV files:

```bash
# Export to default 'exports' directory
python -m polymarket_tracker.dashboard --export

# Export to custom directory
python -m polymarket_tracker.dashboard --export ./my_exports
```

### Generate Sample Data

For testing without live data:

```bash
# Generate 50 traders and 20 markets
python -m polymarket_tracker.sample_data

# Custom amounts
python -m polymarket_tracker.sample_data --traders 100 --markets 30

# Clear existing data first
python -m polymarket_tracker.sample_data --clear
```

## PNL Calculation Logic

The leaderboard calculates profit/loss using the following logic:

### Resolved Markets
```
Winning position: (shares * $1 payout) - cost_basis - fees
Losing position:  -cost_basis (shares worth $0)
```

### Fee Structure
- Polymarket charges 2% on net winnings
- No fees on losing positions

### Unrealized PNL (Open Positions)
```
Unrealized PNL = (current_price - entry_price) * position_size
```

### Example Calculation

```
Trader buys 100 "Yes" shares at $0.50 = $50 cost
Market resolves "Yes"
Payout = 100 * $1 = $100
Gross profit = $100 - $50 = $50
Fee = $50 * 0.02 = $1
Net profit = $49
```

## Win Rate Analysis

### Win Rate Dashboard

Launch the dedicated win rate dashboard:

```bash
python -m polymarket_tracker.win_rate_dashboard
```

### Dashboard Options

```bash
# Custom minimum resolved bets (default: 20)
python -m polymarket_tracker.win_rate_dashboard --min-bets 10

# Sort by different metrics
python -m polymarket_tracker.win_rate_dashboard --sort win_rate
python -m polymarket_tracker.win_rate_dashboard --sort risk_adjusted
python -m polymarket_tracker.win_rate_dashboard --sort sharpe
python -m polymarket_tracker.win_rate_dashboard --sort streak

# Detailed trader analysis
python -m polymarket_tracker.win_rate_dashboard --trader 0x1234...
```

### Win Rate Calculation

Win rates are calculated **only on resolved markets**:

```
Win Rate = (Winning Bets / Decisive Bets) * 100
Decisive Bets = Wins + Losses (excludes ties and cancelled markets)
```

### Win Rate by Category

Breakdown by market category (politics, sports, crypto, tech, etc.):
- Each category tracked separately
- Helps identify trader strengths

### Win Rate by Bet Size

Breakdown by bet amount:
- **Small**: < $100
- **Medium**: $100 - $1,000
- **Large**: > $1,000

### Streak Detection

Tracks winning and losing streaks:
- Current active streak
- Longest historical winning streak
- Longest historical losing streak

### Statistical Significance

- **Minimum 10 bets** required for statistical significance
- **Wilson Score Interval** used for confidence intervals (better for small samples)
- **95% confidence intervals** shown for win rates

### Kelly Criterion Analysis

Optimal bet sizing analysis:
```
Optimal Fraction = (win_rate * odds - loss_rate) / odds
Kelly Multiple = Actual Bet Size / Optimal Size
```

Recommendations:
- `< 0.5x Kelly`: Conservative - could increase sizes
- `0.5x - 1.5x Kelly`: Near optimal
- `> 1.5x Kelly`: Overbetting - reduce position sizes

### Risk Metrics

- **Sharpe Ratio**: Risk-adjusted return measure
- **Max Drawdown**: Largest peak-to-trough decline
- **Risk-Adjusted Return**: Return / Max Drawdown

### Edge Cases Handled

| Scenario | Handling |
|----------|----------|
| Ties | Excluded from win rate (not decisive) |
| Cancelled Markets | Treated as refunds, excluded |
| Partial Fills | Flagged, can be filtered |
| Multiple Bets Same Market | Net position calculated |
| Sell Positions | Opposite logic applied |
| Insufficient Data | Flagged with wider confidence intervals |

## Insider Trading Detection

The insider detection system identifies suspicious betting patterns that may indicate trading on non-public information.

### Insider Dashboard

Launch the suspicious activity dashboard:

```bash
python -m polymarket_tracker.insider_dashboard
```

### Dashboard Options

```bash
# Custom refresh interval
python -m polymarket_tracker.insider_dashboard --refresh 60

# Minimum anomaly score to display
python -m polymarket_tracker.insider_dashboard --min-score 70

# Show suspicious bets only (no other panels)
python -m polymarket_tracker.insider_dashboard --bets-only

# Analyze specific wallet
python -m polymarket_tracker.insider_dashboard --wallet 0x1234...

# Analyze specific market
python -m polymarket_tracker.insider_dashboard --market market_id
```

### Detection Methods

#### Timing-Based Alerts

| Alert Type | Description | Score Weight |
|------------|-------------|--------------|
| Large Bet Near Resolution | Big bets placed within 24 hours of market close | High |
| Dormant Wallet Activation | Inactive wallet (30+ days) returns with large bet | High |
| Low Liquidity Betting | Large bets placed during low-activity hours (0-6 UTC) | Medium |

#### Pattern Recognition

| Pattern | Description |
|---------|-------------|
| Niche Market Dominance | High win rate on low-volume markets |
| First Mover Advantage | Early bets on markets that move significantly |
| Correlated Betting | Multiple wallets betting together within short timeframes |

#### Sybil Detection

Identifies potential wallet clusters:
- Similar betting patterns across wallets
- Coordinated timing of bets
- Shared market preferences

### Anomaly Scoring

Each bet receives a score from 0-100 based on multiple factors:

```
Score = Timing Score + Amount Score + History Score + Liquidity Score + Probability Score
```

Severity levels:
- **CRITICAL** (90+): Immediate attention required
- **HIGH** (80-89): Significant suspicion
- **MEDIUM** (60-79): Worth monitoring
- **LOW** (<60): Normal activity

### Alert Configuration

Set up Discord and email alerts in `.env`:

```env
# Discord webhook for alerts
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# Email alerts (SMTP)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your@email.com
SMTP_PASSWORD=your-app-password
ALERT_EMAIL_FROM=alerts@yourtracker.com
ALERT_EMAIL_TO=admin@yourtracker.com

# Alert thresholds
ALERT_SCORE_THRESHOLD=80
ALERT_COOLDOWN_MINUTES=30
```

### ML-Based Detection (Optional)

Train a machine learning classifier for enhanced detection:

```bash
# Install ML dependencies
pip install scikit-learn xgboost numpy

# Train model
python -c "from polymarket_tracker.ml_detector import train_and_save_model; train_and_save_model()"
```

The ML detector uses 26 engineered features:
- **Timing features**: Hours to resolution, time of day, day of week
- **Size features**: Bet amount, relative to market volume
- **History features**: Trader's past behavior, win rate, dormancy
- **Market features**: Liquidity, volume, category, age
- **Probability features**: Implied odds, deviation from consensus

Supported models:
- Random Forest (default)
- XGBoost

### Wallet Risk Profiles

Build comprehensive risk profiles for wallets:

```python
from polymarket_tracker.insider_detection import InsiderDetector

detector = InsiderDetector()
profile = detector.build_wallet_risk_profile("0x1234...")

print(f"Risk Score: {profile.overall_risk_score}")
print(f"Suspicious Bets: {profile.total_suspicious_bets}")
print(f"High-Risk Markets: {profile.high_risk_market_count}")
```

### Correlated Betting Detection

Detect coordinated betting activity:

```python
groups = detector.detect_correlated_betting(
    market_id="market_123",
    time_window_minutes=60,
    min_wallets=3
)

for group in groups:
    print(f"Wallets: {group.wallets}")
    print(f"Correlation Score: {group.correlation_score}")
    print(f"Total Volume: ${group.total_volume:,.2f}")
```

## API Reference

### Polymarket APIs Used

| API | Base URL | Purpose |
|-----|----------|---------|
| CLOB | https://clob.polymarket.com | Order book, prices, trades |
| Gamma | https://gamma-api.polymarket.com | Market metadata, events |
| Data | https://data-api.polymarket.com | Positions, activity |

### Rate Limits

The client respects Polymarket's rate limits:
- Trades: 200 requests/10s
- Positions: 150 requests/10s
- Markets: 300 requests/10s

## Example Queries

After collecting data, you can query the SQLite database directly:

```sql
-- Top traders by volume
SELECT wallet_address, total_volume, total_trades
FROM traders
ORDER BY total_volume DESC
LIMIT 10;

-- Active markets with highest volume
SELECT market_id, question, volume
FROM markets
WHERE active = 1 AND closed = 0
ORDER BY volume DESC
LIMIT 10;

-- Recent trades
SELECT b.*, m.question
FROM bets b
JOIN markets m ON b.market_id = m.market_id
ORDER BY b.timestamp DESC
LIMIT 20;

-- Trader activity summary
SELECT
    wallet_address,
    COUNT(*) as trade_count,
    SUM(amount * price) as total_volume,
    MIN(timestamp) as first_trade,
    MAX(timestamp) as last_trade
FROM bets
GROUP BY wallet_address
ORDER BY total_volume DESC;
```

## Development

### Project Structure

- `config.py`: Uses pydantic-settings for type-safe configuration
- `models.py`: Pydantic models matching Polymarket API responses
- `database.py`: SQLite operations with context managers
- `api_client.py`: Async HTTP client with rate limiting via token bucket
- `collector.py`: Orchestrates data collection with incremental sync
- `analytics.py`: PNL calculations, volume tracking, leaderboard generation
- `dashboard.py`: Interactive CLI dashboard with ANSI colors
- `export.py`: JSON and CSV export functionality
- `sample_data.py`: Sample data generator for testing
- `insider_detection.py`: Anomaly scoring with timing, pattern, and correlation detection
- `alerts.py`: Discord webhook and SMTP email alerting
- `insider_dashboard.py`: CLI dashboard for suspicious activity monitoring
- `ml_detector.py`: Optional ML classifier using Random Forest or XGBoost
- `main.py`: CLI with APScheduler for scheduled jobs

### Running Tests

```bash
pytest
```

### Extending the Tracker

To add new data collection features:

1. Add new API methods to `api_client.py`
2. Add database tables/operations to `database.py`
3. Add collection logic to `collector.py`

## Troubleshooting

### API Errors

- **429 Too Many Requests**: The client handles this automatically with backoff
- **Connection Errors**: Check your internet connection; the client retries 3 times

### Database Issues

- **Locked Database**: Ensure only one instance is running
- **Schema Changes**: Delete the `.db` file to recreate with new schema

### Performance

- For large datasets, consider adding indexes or using a more robust database
- Adjust `FETCH_INTERVAL_MINUTES` based on your needs

## License

MIT License

## Resources

- [Polymarket Documentation](https://docs.polymarket.com)
- [CLOB API Reference](https://docs.polymarket.com/developers/CLOB/introduction)
- [Gamma API Reference](https://docs.polymarket.com/gamma-api)
