"""
Data collection service for Polymarket Tracker.

This module handles the periodic collection of data from Polymarket:
- Fetches new trades and updates trader statistics
- Syncs market information
- Maintains incremental sync state to avoid re-fetching data

The collector runs as a scheduled job using APScheduler.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

from .api_client import PolymarketClient, create_client, PolymarketAPIError
from .database import Database, db
from .config import settings
from .signal_engine import SignalEngine
from .whale_profiler import WhaleProfiler

logger = logging.getLogger(__name__)


class DataCollector:
    """
    Collects and stores Polymarket trading data.

    Handles incremental syncing of markets, trades, and positions
    with state tracking to resume collection efficiently.
    """

    def __init__(
        self,
        client: Optional[PolymarketClient] = None,
        database: Optional[Database] = None
    ):
        """
        Initialize the data collector.

        Args:
            client: Polymarket API client (created if not provided).
            database: Database instance (uses global if not provided).
        """
        self.client = client or create_client()
        self.db = database or db
        self._running = False

    async def sync_markets(self) -> int:
        """
        Sync active and resolved markets from Polymarket.

        Returns:
            Number of markets synced.
        """
        logger.info("Starting market sync...")
        markets_synced = 0

        try:
            # Fetch active markets
            markets = await self.client.get_all_active_markets(max_pages=20)

            # Also fetch resolved markets (needed for whale win-rate grading)
            try:
                resolved = await self.client.get_all_resolved_markets(max_pages=5)
                markets.extend(resolved)
                logger.info(f"Fetched {len(resolved)} resolved markets for whale grading")
            except Exception as e:
                logger.warning(f"Could not fetch resolved markets: {e}")

            for market_data in markets:
                try:
                    # Extract market fields with safe defaults
                    condition_id = market_data.get("conditionId") or market_data.get("condition_id")
                    if not condition_id:
                        continue

                    question = market_data.get("question", "Unknown")
                    description = market_data.get("description")

                    # Parse end date
                    end_date = None
                    end_date_str = market_data.get("endDateIso") or market_data.get("endDate")
                    if end_date_str:
                        try:
                            end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                        except (ValueError, TypeError):
                            pass

                    # Extract token IDs
                    token_ids = None
                    clob_token_ids = market_data.get("clobTokenIds")
                    if clob_token_ids:
                        if isinstance(clob_token_ids, list):
                            token_ids = json.dumps(clob_token_ids)
                        else:
                            token_ids = clob_token_ids

                    # Extract outcome prices
                    outcome_prices = market_data.get("outcomePrices")
                    if outcome_prices and isinstance(outcome_prices, list):
                        outcome_prices = json.dumps(outcome_prices)

                    self.db.upsert_market(
                        market_id=condition_id,
                        question=question,
                        description=description,
                        end_date=end_date,
                        active=market_data.get("active", True),
                        closed=market_data.get("closed", False),
                        resolved=market_data.get("resolved", False),
                        outcome=market_data.get("outcome"),
                        volume=float(market_data.get("volume") or 0),
                        liquidity=float(market_data.get("liquidity") or 0) if market_data.get("liquidity") else None,
                        category=market_data.get("category"),
                        slug=market_data.get("slug"),
                        token_ids=token_ids,
                        outcome_prices=outcome_prices,
                    )
                    markets_synced += 1

                except Exception as e:
                    logger.error(f"Error processing market {market_data.get('conditionId')}: {e}")
                    continue

            logger.info(f"Market sync complete: {markets_synced} markets synced")
            self.db.set_sync_state("last_market_sync", datetime.utcnow().isoformat())

        except PolymarketAPIError as e:
            logger.error(f"API error during market sync: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during market sync: {e}")

        return markets_synced

    async def sync_trades_for_market(
        self,
        market_id: str,
        since_timestamp: Optional[int] = None,
        min_trade_size: float = 0.0
    ) -> int:
        """
        Sync trades for a specific market.

        Args:
            market_id: Market condition ID.
            since_timestamp: Unix timestamp to fetch trades after.
            min_trade_size: Minimum trade size in dollars to include.

        Returns:
            Number of new trades synced.
        """
        trades_synced = 0
        trades_skipped = 0

        try:
            trades = await self.client.get_market_trades(
                condition_id=market_id,
                after=since_timestamp
            )

            for trade_data in trades:
                try:
                    # Calculate trade volume first for filtering
                    size = float(trade_data.get("size", 0))
                    price = float(trade_data.get("price", 0))
                    volume = size * price

                    # Skip trades below minimum size
                    if volume < min_trade_size:
                        trades_skipped += 1
                        continue

                    # Data API uses transactionHash as unique ID
                    tx_hash = trade_data.get("transactionHash") or trade_data.get("transaction_hash")
                    trade_id = trade_data.get("id") or tx_hash
                    if not trade_id:
                        continue

                    # Data API uses proxyWallet for wallet address
                    wallet_address = (
                        trade_data.get("proxyWallet") or
                        trade_data.get("makerAddress") or
                        trade_data.get("maker_address") or
                        trade_data.get("owner") or
                        "unknown"
                    )

                    # Parse timestamp - Data API uses 'timestamp' field directly
                    timestamp = datetime.utcnow()
                    ts_value = trade_data.get("timestamp") or trade_data.get("matchTime") or trade_data.get("match_time")
                    if ts_value:
                        try:
                            ts = float(ts_value)
                            if ts > 1e12:  # Milliseconds
                                ts = ts / 1000
                            timestamp = datetime.fromtimestamp(ts)
                        except (ValueError, TypeError):
                            pass

                    # Insert bet record - Data API uses 'asset' not 'assetId'
                    inserted = self.db.insert_bet(
                        bet_id=trade_id,
                        wallet_address=wallet_address,
                        market_id=market_id,
                        asset_id=trade_data.get("asset") or trade_data.get("assetId") or trade_data.get("asset_id", ""),
                        amount=size,
                        price=price,
                        side=trade_data.get("side", "unknown"),
                        outcome_bet=trade_data.get("outcome", ""),
                        timestamp=timestamp,
                        fee_rate_bps=float(trade_data.get("feeRateBps") or trade_data.get("fee_rate_bps") or 0),
                        status=trade_data.get("status"),
                        transaction_hash=tx_hash,
                    )

                    if inserted:
                        trades_synced += 1
                        # Update trader statistics
                        self.db.upsert_trader(
                            wallet_address=wallet_address,
                            volume=volume,
                            timestamp=timestamp
                        )

                except Exception as e:
                    logger.error(f"Error processing trade {trade_data.get('id')}: {e}")
                    continue

            if trades_skipped > 0:
                logger.debug(f"Skipped {trades_skipped} trades below ${min_trade_size:.0f} minimum")

        except PolymarketAPIError as e:
            logger.error(f"API error fetching trades for market {market_id}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching trades for market {market_id}: {e}")

        return trades_synced

    async def sync_resolved_market_trades(self, max_markets: int = 50) -> int:
        """
        Backfill trades for resolved markets that have no trade data yet.

        This is essential for whale grading — we need trades on resolved
        markets to compute win rates. Only processes markets where we have
        the outcome but zero bets in the DB.

        Args:
            max_markets: Maximum resolved markets to process per cycle.

        Returns:
            Total number of trades synced for resolved markets.
        """
        resolved_markets = self.db.get_resolved_markets_needing_trades(limit=max_markets)

        if not resolved_markets:
            return 0

        logger.info(f"Backfilling trades for {len(resolved_markets)} resolved markets...")
        total_trades = 0

        for i, market in enumerate(resolved_markets):
            market_id = market["market_id"]
            try:
                trades = await self.sync_trades_for_market(market_id)
                total_trades += trades

                if trades > 0:
                    logger.debug(f"Resolved market {market_id}: {trades} trades backfilled")

                if (i + 1) % 10 == 0:
                    logger.info(f"Resolved trade backfill: {i + 1}/{len(resolved_markets)} markets, {total_trades} trades")

                await asyncio.sleep(0.3)

            except Exception as e:
                logger.error(f"Error backfilling trades for resolved market {market_id}: {e}")
                continue

        logger.info(f"Resolved trade backfill complete: {total_trades} trades from {len(resolved_markets)} markets")
        return total_trades

    async def sync_all_trades(
        self,
        max_markets: int = 100,
        min_trade_size: float = 0.0,
        min_market_volume: float = 0.0
    ) -> int:
        """
        Sync trades for all active markets.

        Args:
            max_markets: Maximum number of markets to process.
            min_trade_size: Minimum trade size in dollars to include.
            min_market_volume: Minimum market volume to process (skip low-volume markets).

        Returns:
            Total number of new trades synced.
        """
        if min_trade_size > 0:
            logger.info(f"Starting trade sync (min trade: ${min_trade_size:,.0f}, min market volume: ${min_market_volume:,.0f})...")
        else:
            logger.info("Starting trade sync for all markets...")
        total_trades = 0
        markets_skipped = 0

        # Get last sync timestamp
        last_sync_str = self.db.get_sync_state("last_trade_sync")
        since_timestamp = None
        if last_sync_str:
            try:
                last_sync = datetime.fromisoformat(last_sync_str)
                since_timestamp = int(last_sync.timestamp())
            except (ValueError, TypeError):
                pass

        # Get active markets from database, sorted by volume
        active_markets = self.db.get_active_markets(limit=max_markets)

        if not active_markets:
            logger.warning("No active markets in database. Run market sync first.")
            return 0

        for i, market in enumerate(active_markets):
            market_id = market["market_id"]
            market_volume = float(market.get("volume", 0) or 0)

            # Skip low-volume markets
            if min_market_volume > 0 and market_volume < min_market_volume:
                markets_skipped += 1
                continue

            try:
                trades = await self.sync_trades_for_market(
                    market_id,
                    since_timestamp,
                    min_trade_size=min_trade_size
                )
                total_trades += trades

                if trades > 0:
                    logger.debug(f"Market {market_id}: {trades} new trades")

                # Progress logging every 10 markets
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(active_markets)} markets, {total_trades} total trades")

                # Small delay between markets to avoid rate limiting
                await asyncio.sleep(0.2)

            except Exception as e:
                logger.error(f"Error syncing trades for market {market_id}: {e}")
                continue

        # Update sync state
        self.db.set_sync_state("last_trade_sync", datetime.utcnow().isoformat())

        if markets_skipped > 0:
            logger.info(f"Skipped {markets_skipped} markets below ${min_market_volume:,.0f} volume threshold")
        logger.info(f"Trade sync complete: {total_trades} new trades from {len(active_markets) - markets_skipped} markets")

        return total_trades

    async def collect_data(self) -> dict:
        """
        Run a complete data collection cycle.

        This is the main entry point for scheduled collection jobs.

        Returns:
            Dictionary with collection statistics.
        """
        logger.info("=" * 50)
        logger.info("Starting data collection cycle...")
        start_time = datetime.utcnow()

        stats = {
            "start_time": start_time.isoformat(),
            "markets_synced": 0,
            "trades_synced": 0,
            "errors": [],
        }

        try:
            # Step 1: Sync markets (less frequently)
            last_market_sync = self.db.get_sync_state("last_market_sync")
            should_sync_markets = True

            if last_market_sync:
                try:
                    last_sync = datetime.fromisoformat(last_market_sync)
                    # Only sync markets every 30 minutes
                    if datetime.utcnow() - last_sync < timedelta(minutes=30):
                        should_sync_markets = False
                except (ValueError, TypeError):
                    pass

            if should_sync_markets:
                stats["markets_synced"] = await self.sync_markets()

                # Backfill trades for resolved markets (needed for whale grading)
                try:
                    resolved_trades = await self.sync_resolved_market_trades(max_markets=50)
                    stats["resolved_trades_backfilled"] = resolved_trades
                except Exception as e:
                    logger.error(f"Resolved trade backfill error: {e}")
                    stats["resolved_trades_backfilled"] = 0

                # Refresh whale profiles after backfilling resolved trades
                try:
                    profiler = WhaleProfiler(database=self.db)
                    stats["whales_profiled"] = profiler.refresh_all_profiles()
                except Exception as e:
                    logger.error(f"Whale profiling error: {e}")
                    stats["whales_profiled"] = 0

            # Step 2: Sync trades for all active markets
            # Check if we had a prior trade sync (to detect first run)
            had_prior_trade_sync = self.db.get_sync_state("last_trade_sync") is not None
            stats["trades_synced"] = await self.sync_all_trades()

            # Step 3: Evaluate recent bets for signals
            # Skip on first sync — all bets are historical, not new activity
            if not had_prior_trade_sync:
                logger.info("First trade sync complete — skipping signal evaluation (all bets are historical)")
                stats["signals_fired"] = 0
            else:
                try:
                    signal_engine = SignalEngine(database=self.db)
                    signals = signal_engine.evaluate_recent_bets(
                        since_minutes=settings.fetch_interval_minutes
                    )
                    stats["signals_fired"] = len(signals)
                    if signals:
                        logger.info(f"Signal engine: {len(signals)} signals fired")
                except Exception as e:
                    logger.error(f"Signal evaluation error: {e}")
                    stats["signals_fired"] = 0

        except Exception as e:
            error_msg = f"Collection cycle error: {e}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)

        end_time = datetime.utcnow()
        stats["end_time"] = end_time.isoformat()
        stats["duration_seconds"] = (end_time - start_time).total_seconds()

        # Log database stats
        db_stats = self.db.get_stats()
        logger.info(f"Database stats: {db_stats}")
        logger.info(f"Collection cycle complete in {stats['duration_seconds']:.2f}s")
        logger.info("=" * 50)

        return stats

    async def run_continuous(self, interval_minutes: Optional[int] = None) -> None:
        """
        Run continuous data collection.

        Args:
            interval_minutes: Minutes between collection cycles.
        """
        interval = interval_minutes or settings.fetch_interval_minutes
        self._running = True

        logger.info(f"Starting continuous collection (interval: {interval} minutes)")

        while self._running:
            try:
                await self.collect_data()
            except Exception as e:
                logger.error(f"Collection error: {e}")

            # Wait for next cycle
            logger.info(f"Sleeping for {interval} minutes...")
            await asyncio.sleep(interval * 60)

    def stop(self) -> None:
        """Stop continuous collection."""
        self._running = False
        logger.info("Collection stopped")

    async def close(self) -> None:
        """Clean up resources."""
        await self.client.close()


async def run_collection_cycle() -> dict:
    """
    Convenience function to run a single collection cycle.

    Returns:
        Collection statistics.
    """
    collector = DataCollector()
    try:
        return await collector.collect_data()
    finally:
        await collector.close()
