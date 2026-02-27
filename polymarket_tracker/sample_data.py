"""
Sample data generator for testing and demonstration.

This module generates realistic sample data for the Polymarket tracker
to demonstrate the leaderboard functionality without requiring live API data.
"""

import random
import string
from datetime import datetime, timedelta
from typing import Optional

from .database import Database, db


def generate_wallet_address() -> str:
    """Generate a random Ethereum-style wallet address."""
    return "0x" + "".join(random.choices(string.hexdigits.lower(), k=40))


def generate_sample_data(
    database: Optional[Database] = None,
    num_traders: int = 50,
    num_markets: int = 20,
    trades_per_trader: tuple[int, int] = (5, 50)
) -> dict:
    """
    Generate sample data for demonstration.

    Args:
        database: Database instance (uses global if not provided).
        num_traders: Number of traders to generate.
        num_markets: Number of markets to generate.
        trades_per_trader: Min and max trades per trader.

    Returns:
        Dictionary with generation statistics.
    """
    db_instance = database or db

    stats = {
        "traders_created": 0,
        "markets_created": 0,
        "bets_created": 0,
    }

    # Sample market questions
    market_questions = [
        "Will Bitcoin reach $100,000 by end of 2024?",
        "Will the Fed cut interest rates in March?",
        "Will SpaceX Starship reach orbit?",
        "Will Taylor Swift win Album of the Year?",
        "Will there be a government shutdown?",
        "Will the S&P 500 hit new all-time high?",
        "Will OpenAI release GPT-5?",
        "Will Apple announce AR glasses?",
        "Will Tesla deliver 2M vehicles?",
        "Will unemployment stay below 4%?",
        "Will inflation drop below 3%?",
        "Will Congress pass immigration reform?",
        "Will Netflix stock double?",
        "Will there be a major cyberattack?",
        "Will a new COVID variant emerge?",
        "Will electric vehicles outsell gas cars?",
        "Will gold reach $2500/oz?",
        "Will oil prices exceed $100/barrel?",
        "Will AI replace 10% of jobs?",
        "Will there be a major earthquake?",
    ]

    # Generate markets
    markets = []
    for i in range(num_markets):
        market_id = f"market_{i:04d}_{random.randint(1000, 9999)}"
        question = market_questions[i % len(market_questions)]

        # Randomly resolve some markets
        resolved = random.random() < 0.3
        outcome = random.choice(["Yes", "No"]) if resolved else None

        # Random end date (some past, some future)
        days_offset = random.randint(-30, 90)
        end_date = datetime.utcnow() + timedelta(days=days_offset)

        # Generate outcome prices
        yes_price = random.uniform(0.1, 0.9)
        outcome_prices = f'[{yes_price:.4f}, {1 - yes_price:.4f}]'

        db_instance.upsert_market(
            market_id=market_id,
            question=question,
            description=f"Sample market #{i + 1}",
            end_date=end_date,
            active=not resolved and days_offset > 0,
            closed=resolved or days_offset < 0,
            resolved=resolved,
            outcome=outcome,
            volume=random.uniform(10000, 5000000),
            liquidity=random.uniform(5000, 500000),
            category=random.choice(["crypto", "politics", "sports", "tech", "finance"]),
            slug=f"sample-market-{i}",
            token_ids=f'["token_yes_{market_id}", "token_no_{market_id}"]',
            outcome_prices=outcome_prices,
        )
        markets.append({
            "market_id": market_id,
            "resolved": resolved,
            "outcome": outcome,
        })
        stats["markets_created"] += 1

    # Generate traders and their bets
    traders = []
    for i in range(num_traders):
        wallet = generate_wallet_address()
        traders.append(wallet)

        # Each trader makes some number of trades
        num_trades = random.randint(*trades_per_trader)
        total_volume = 0

        for j in range(num_trades):
            market = random.choice(markets)
            market_id = market["market_id"]

            # Generate trade details
            amount = random.uniform(10, 5000)
            price = random.uniform(0.1, 0.9)
            side = random.choice(["BUY", "SELL"])
            outcome_bet = random.choice(["Yes", "No"])

            # Random timestamp in last 90 days
            trade_time = datetime.utcnow() - timedelta(
                days=random.randint(0, 90),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )

            bet_id = f"bet_{wallet[:8]}_{market_id}_{j}_{random.randint(1000, 9999)}"

            inserted = db_instance.insert_bet(
                bet_id=bet_id,
                wallet_address=wallet,
                market_id=market_id,
                asset_id=f"token_{outcome_bet.lower()}_{market_id}",
                amount=amount,
                price=price,
                side=side,
                outcome_bet=outcome_bet,
                timestamp=trade_time,
                fee_rate_bps=random.uniform(0, 100),
                status="MATCHED",
                transaction_hash=f"0x{random.randint(10**63, 10**64-1):064x}",
            )

            if inserted:
                stats["bets_created"] += 1
                total_volume += amount * price

        # Update trader record
        db_instance.upsert_trader(
            wallet_address=wallet,
            volume=total_volume,
            timestamp=datetime.utcnow()
        )
        stats["traders_created"] += 1

    print(f"Generated sample data:")
    print(f"  - {stats['traders_created']} traders")
    print(f"  - {stats['markets_created']} markets")
    print(f"  - {stats['bets_created']} bets")

    return stats


def clear_sample_data(database: Optional[Database] = None) -> None:
    """
    Clear all data from the database.

    Args:
        database: Database instance (uses global if not provided).
    """
    db_instance = database or db

    with db_instance.get_connection() as conn:
        conn.execute("DELETE FROM bets")
        conn.execute("DELETE FROM positions")
        conn.execute("DELETE FROM traders")
        conn.execute("DELETE FROM markets")
        conn.execute("DELETE FROM sync_state")

    print("All sample data cleared.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate sample data for Polymarket Tracker")
    parser.add_argument("--traders", type=int, default=50, help="Number of traders")
    parser.add_argument("--markets", type=int, default=20, help="Number of markets")
    parser.add_argument("--clear", action="store_true", help="Clear existing data first")

    args = parser.parse_args()

    if args.clear:
        clear_sample_data()

    generate_sample_data(
        num_traders=args.traders,
        num_markets=args.markets
    )
