"""
Main entry point for Polymarket Tracker.

This module provides the CLI interface and scheduled job runner
for the Polymarket data collection system.

Usage:
    # Run a single collection cycle
    python -m polymarket_tracker.main --once

    # Run continuous collection with scheduling
    python -m polymarket_tracker.main

    # Show database statistics
    python -m polymarket_tracker.main --stats

    # One-time backfill of resolved markets for whale grading
    python -m polymarket_tracker.main --backfill
"""

import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from .config import settings
from .collector import DataCollector, run_collection_cycle
from .database import db
from .whale_profiler import WhaleProfiler

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("polymarket_tracker.log"),
    ],
)
logger = logging.getLogger(__name__)


class PolymarketTracker:
    """
    Main application class for the Polymarket Tracker.

    Manages the scheduled data collection jobs and graceful shutdown.
    """

    def __init__(self):
        """Initialize the tracker."""
        self.scheduler: AsyncIOScheduler = None
        self.collector: DataCollector = None
        self._shutdown_event = asyncio.Event()

    async def start(self, run_once: bool = False) -> None:
        """
        Start the tracker.

        Args:
            run_once: If True, run a single collection cycle and exit.
        """
        logger.info("=" * 60)
        logger.info("Polymarket Tracker Starting")
        logger.info(f"Database: {settings.database_path}")
        logger.info(f"Collection interval: {settings.fetch_interval_minutes} minutes")
        logger.info("=" * 60)

        self.collector = DataCollector()

        # Check API connectivity
        if await self.collector.client.health_check():
            logger.info("API connectivity: OK")
        else:
            logger.warning("API connectivity: FAILED (will retry)")

        if run_once:
            # Single collection cycle
            await self.collector.collect_data()
            await self.collector.close()
            return

        # Set up scheduler for continuous collection
        self.scheduler = AsyncIOScheduler()

        # Add the collection job
        self.scheduler.add_job(
            self._run_collection_job,
            trigger=IntervalTrigger(minutes=settings.fetch_interval_minutes),
            id="data_collection",
            name="Polymarket Data Collection",
            next_run_time=datetime.now(),  # Run immediately on start
        )

        self.scheduler.start()
        logger.info("Scheduler started")

        # Wait for shutdown signal
        await self._shutdown_event.wait()

    async def _run_collection_job(self) -> None:
        """Execute a data collection job."""
        try:
            await self.collector.collect_data()
        except Exception as e:
            logger.error(f"Collection job failed: {e}")

    async def stop(self) -> None:
        """Stop the tracker gracefully."""
        logger.info("Shutting down...")

        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            logger.info("Scheduler stopped")

        if self.collector:
            await self.collector.close()
            logger.info("Collector closed")

        self._shutdown_event.set()

    def handle_signal(self, signum, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}")
        asyncio.create_task(self.stop())


def print_stats() -> None:
    """Print database statistics."""
    stats = db.get_stats()
    print("\n" + "=" * 50)
    print("Polymarket Tracker Database Statistics")
    print("=" * 50)
    print(f"  Total Traders:    {stats['total_traders']:,}")
    print(f"  Total Markets:    {stats['total_markets']:,}")
    print(f"  Active Markets:   {stats['active_markets']:,}")
    print(f"  Total Bets:       {stats['total_bets']:,}")
    print(f"  Total Volume:     ${stats['total_volume']:,.2f}")
    print(f"  Total Positions:  {stats['total_positions']:,}")
    print("=" * 50)

    # Show top traders
    top_traders = db.get_top_traders(limit=5)
    if top_traders:
        print("\nTop 5 Traders by Volume:")
        print("-" * 50)
        for i, trader in enumerate(top_traders, 1):
            print(f"  {i}. {trader['wallet_address'][:10]}... - ${trader['total_volume']:,.2f}")

    # Show recent activity
    recent_bets = db.get_recent_bets(limit=5)
    if recent_bets:
        print("\nRecent Trades:")
        print("-" * 50)
        for bet in recent_bets:
            print(f"  {bet['timestamp']} - {bet['side']} ${bet['amount']:.2f} @ {bet['price']:.4f}")

    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Polymarket Tracker - Data collection system for Polymarket"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single collection cycle and exit"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show database statistics and exit"
    )
    parser.add_argument(
        "--sync-markets",
        action="store_true",
        help="Only sync markets (no trades)"
    )
    parser.add_argument(
        "--sync-trades",
        action="store_true",
        help="Only sync trades (no markets)"
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="One-time backfill: fetch resolved markets, their trades, and grade whales, then exit"
    )
    parser.add_argument(
        "--backfill-limit",
        type=int,
        default=200,
        help="Max resolved markets to backfill per run (default: 200)"
    )
    args = parser.parse_args()

    if args.stats:
        print_stats()
        return

    async def run():
        tracker = PolymarketTracker()

        # Set up signal handlers for graceful shutdown
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, tracker.handle_signal)

        if args.backfill:
            collector = DataCollector()
            try:
                print("\n" + "=" * 60)
                print("Polymarket Tracker — Resolved Market Backfill")
                print("=" * 60)

                # Step 1: Sync resolved markets from API
                print("\n[1/3] Syncing resolved markets from Polymarket API...")
                markets_synced = await collector.sync_markets()
                resolved_count = 0
                with collector.db.get_connection() as conn:
                    row = conn.execute("SELECT COUNT(*) as cnt FROM markets WHERE resolved = 1").fetchone()
                    resolved_count = row["cnt"] if row else 0
                print(f"  Total markets synced: {markets_synced}")
                print(f"  Resolved markets in DB: {resolved_count}")

                # Step 2: Backfill trades for resolved markets
                needing = collector.db.get_resolved_markets_needing_trades(limit=args.backfill_limit)
                print(f"\n[2/3] Backfilling trades for {len(needing)} resolved markets (limit: {args.backfill_limit})...")
                trades = await collector.sync_resolved_market_trades(max_markets=args.backfill_limit)
                print(f"  Trades backfilled: {trades}")

                # Step 3: Refresh whale profiles
                print("\n[3/3] Grading whale wallets...")
                profiler = WhaleProfiler(database=collector.db)
                whales = profiler.refresh_all_profiles()
                print(f"  Whale profiles updated: {whales}")

                # Summary
                remaining = collector.db.get_resolved_markets_needing_trades(limit=1)
                print("\n" + "=" * 60)
                if remaining:
                    print(f"Done! {len(remaining)}+ resolved markets still need trades.")
                    print(f"Run --backfill again to process the next batch.")
                else:
                    print("Done! All resolved markets have trade data.")
                print("=" * 60 + "\n")

            finally:
                await collector.close()
            return

        if args.sync_markets:
            collector = DataCollector()
            try:
                await collector.sync_markets()
            finally:
                await collector.close()
            return

        if args.sync_trades:
            collector = DataCollector()
            try:
                await collector.sync_all_trades()
            finally:
                await collector.close()
            return

        await tracker.start(run_once=args.once)

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
