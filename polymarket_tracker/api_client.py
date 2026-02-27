"""
Polymarket API client with rate limiting and error handling.

This module provides async HTTP clients for the Polymarket APIs:
- CLOB API: Order book, prices, trades
- Gamma API: Markets and events metadata
- Data API: User positions and activity

Rate limiting is implemented per the Polymarket documentation:
- Trades: 200 requests/10s
- Positions: 150 requests/10s
- Markets: 300 requests/10s
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Optional

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .config import settings
from .models import Market, Trade, Position, MarketTradeEvent

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter for API requests.

    Implements a sliding window rate limiter that tracks requests
    over a 10-second window to comply with Polymarket's rate limits.
    """

    def __init__(self, requests_per_10s: int):
        """
        Initialize rate limiter.

        Args:
            requests_per_10s: Maximum requests allowed per 10 seconds.
        """
        self.requests_per_10s = requests_per_10s
        self.window_size = 10.0  # seconds
        self.request_timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """
        Acquire permission to make a request.

        Blocks if rate limit would be exceeded.
        """
        async with self._lock:
            now = time.time()
            # Remove timestamps outside the window
            self.request_timestamps = [
                ts for ts in self.request_timestamps
                if now - ts < self.window_size
            ]

            if len(self.request_timestamps) >= self.requests_per_10s:
                # Calculate wait time
                oldest = self.request_timestamps[0]
                wait_time = self.window_size - (now - oldest)
                if wait_time > 0:
                    logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                    # Clean up old timestamps after waiting
                    now = time.time()
                    self.request_timestamps = [
                        ts for ts in self.request_timestamps
                        if now - ts < self.window_size
                    ]

            self.request_timestamps.append(time.time())


class PolymarketAPIError(Exception):
    """Custom exception for Polymarket API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class PolymarketClient:
    """
    Async client for Polymarket APIs.

    Provides methods to fetch markets, trades, and positions with
    automatic rate limiting and retry logic.
    """

    def __init__(self):
        """Initialize the Polymarket API client."""
        self.clob_url = settings.clob_api_url
        self.gamma_url = settings.gamma_api_url
        self.data_url = settings.data_api_url
        self.chain_id = settings.chain_id
        self.timeout = settings.request_timeout_seconds

        # Initialize rate limiters for different endpoint categories
        self.rate_limiters = {
            "trades": RateLimiter(settings.rate_limit_trades),
            "positions": RateLimiter(settings.rate_limit_positions),
            "markets": RateLimiter(settings.rate_limit_markets),
            "default": RateLimiter(1000),  # Conservative default
        }

        # HTTP client (created per-request for async safety)
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _get_rate_limiter(self, endpoint_type: str) -> RateLimiter:
        """Get the appropriate rate limiter for an endpoint type."""
        return self.rate_limiters.get(endpoint_type, self.rate_limiters["default"])

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    )
    async def _request(
        self,
        method: str,
        url: str,
        endpoint_type: str = "default",
        **kwargs
    ) -> dict:
        """
        Make an HTTP request with rate limiting and retries.

        Args:
            method: HTTP method (GET, POST, etc.).
            url: Full URL to request.
            endpoint_type: Type of endpoint for rate limiting.
            **kwargs: Additional arguments passed to httpx.

        Returns:
            Response JSON as dict.

        Raises:
            PolymarketAPIError: If the request fails.
        """
        # Apply rate limiting
        rate_limiter = self._get_rate_limiter(endpoint_type)
        await rate_limiter.acquire()

        client = await self._get_client()

        try:
            response = await client.request(method, url, **kwargs)

            if response.status_code == 429:
                # Rate limited - wait and retry
                retry_after = int(response.headers.get("Retry-After", "10"))
                logger.warning(f"Rate limited, waiting {retry_after}s")
                await asyncio.sleep(retry_after)
                raise httpx.HTTPError("Rate limited")

            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            raise PolymarketAPIError(
                f"HTTP error: {e.response.text}",
                status_code=e.response.status_code
            )
        except httpx.TimeoutException:
            logger.error(f"Request timeout: {url}")
            raise
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise PolymarketAPIError(str(e))

    # ========== CLOB API Methods ==========

    async def get_markets(
        self,
        next_cursor: Optional[str] = None
    ) -> tuple[list[dict], Optional[str]]:
        """
        Get markets from the CLOB API.

        Args:
            next_cursor: Pagination cursor.

        Returns:
            Tuple of (markets list, next_cursor).
        """
        params = {}
        if next_cursor:
            params["next_cursor"] = next_cursor

        url = f"{self.clob_url}/markets"
        response = await self._request("GET", url, "markets", params=params)

        # Response format: {"data": [...], "next_cursor": "..."}
        if isinstance(response, list):
            return response, None
        return response.get("data", response), response.get("next_cursor")

    async def get_market(self, condition_id: str) -> dict:
        """
        Get a specific market by condition ID.

        Args:
            condition_id: Market condition ID.

        Returns:
            Market data.
        """
        url = f"{self.clob_url}/markets/{condition_id}"
        return await self._request("GET", url, "markets")

    async def get_market_trades(
        self,
        condition_id: str,
        maker: Optional[str] = None,
        taker: Optional[str] = None,
        before: Optional[int] = None,
        after: Optional[int] = None,
        limit: int = 100
    ) -> list[dict]:
        """
        Get trades for a specific market using the public Data API.

        Args:
            condition_id: Market condition ID.
            maker: Filter by maker address.
            taker: Filter by taker address.
            before: Unix timestamp upper bound.
            after: Unix timestamp lower bound.
            limit: Maximum trades to return.

        Returns:
            List of trade data.
        """
        # Use the public Data API instead of CLOB (which requires auth)
        params = {"market": condition_id, "limit": limit}
        if maker:
            params["maker"] = maker
        if taker:
            params["taker"] = taker
        if before:
            params["before"] = str(before)
        if after:
            params["after"] = str(after)

        url = f"{self.data_url}/trades"
        response = await self._request("GET", url, "trades", params=params)

        if isinstance(response, list):
            return response
        return response.get("data", [])

    async def get_price(self, token_id: str, side: str = "buy") -> dict:
        """
        Get current price for a token.

        Args:
            token_id: Token ID.
            side: "buy" or "sell".

        Returns:
            Price data.
        """
        params = {"token_id": token_id, "side": side}
        url = f"{self.clob_url}/price"
        return await self._request("GET", url, "default", params=params)

    async def get_order_book(self, token_id: str) -> dict:
        """
        Get order book for a token.

        Args:
            token_id: Token ID.

        Returns:
            Order book data with bids and asks.
        """
        params = {"token_id": token_id}
        url = f"{self.clob_url}/book"
        return await self._request("GET", url, "default", params=params)

    async def get_prices_history(
        self,
        token_id: str,
        interval: str = "1d",
        fidelity: int = 60
    ) -> list[dict]:
        """
        Get historical prices for a token.

        Args:
            token_id: Token ID.
            interval: Time interval (max, 1w, 1d, 6h, 1h).
            fidelity: Data point frequency in minutes.

        Returns:
            List of historical price points.
        """
        params = {
            "market": token_id,
            "interval": interval,
            "fidelity": fidelity,
        }
        url = f"{self.clob_url}/prices-history"
        response = await self._request("GET", url, "default", params=params)

        if isinstance(response, list):
            return response
        return response.get("history", [])

    # ========== Gamma API Methods ==========

    async def get_gamma_markets(
        self,
        active: bool = True,
        closed: bool = False,
        limit: int = 100,
        offset: int = 0
    ) -> list[dict]:
        """
        Get markets from the Gamma API with metadata.

        Args:
            active: Filter for active markets.
            closed: Filter for closed markets.
            limit: Maximum results per page.
            offset: Pagination offset.

        Returns:
            List of market data with full metadata.
        """
        params = {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "limit": limit,
            "offset": offset,
        }
        url = f"{self.gamma_url}/markets"
        response = await self._request("GET", url, "markets", params=params)

        if isinstance(response, list):
            return response
        return response.get("data", response)

    async def get_gamma_events(
        self,
        active: bool = True,
        limit: int = 100,
        offset: int = 0
    ) -> list[dict]:
        """
        Get events from the Gamma API.

        Args:
            active: Filter for active events.
            limit: Maximum results per page.
            offset: Pagination offset.

        Returns:
            List of event data.
        """
        params = {
            "active": str(active).lower(),
            "limit": limit,
            "offset": offset,
        }
        url = f"{self.gamma_url}/events"
        return await self._request("GET", url, "markets", params=params)

    # ========== Data API Methods ==========

    async def get_positions(
        self,
        address: str,
        limit: int = 100,
        offset: int = 0
    ) -> list[dict]:
        """
        Get positions for a wallet address.

        Note: This endpoint requires authentication for the specific user.
        For public data, use market-level position aggregation.

        Args:
            address: Wallet address.
            limit: Maximum results.
            offset: Pagination offset.

        Returns:
            List of position data.
        """
        params = {
            "user": address,
            "limit": limit,
            "offset": offset,
        }
        url = f"{self.data_url}/positions"
        response = await self._request("GET", url, "positions", params=params)

        if isinstance(response, list):
            return response
        return response.get("data", [])

    async def get_activity(
        self,
        address: str,
        limit: int = 100
    ) -> list[dict]:
        """
        Get activity history for a wallet address.

        Args:
            address: Wallet address.
            limit: Maximum results.

        Returns:
            List of activity records.
        """
        params = {
            "user": address,
            "limit": limit,
        }
        url = f"{self.data_url}/activity"
        response = await self._request("GET", url, "trades", params=params)

        if isinstance(response, list):
            return response
        return response.get("data", [])

    # ========== Utility Methods ==========

    async def health_check(self) -> bool:
        """
        Check if the APIs are accessible.

        Returns:
            True if APIs are healthy.
        """
        try:
            # Simple request to check connectivity
            await self._request("GET", f"{self.clob_url}/time", "default")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def get_resolved_markets(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> list[dict]:
        """
        Get resolved (closed) markets from the Gamma API.

        Args:
            limit: Maximum results per page.
            offset: Pagination offset.

        Returns:
            List of resolved market data.
        """
        params = {
            "closed": "true",
            "limit": limit,
            "offset": offset,
        }
        url = f"{self.gamma_url}/markets"
        response = await self._request("GET", url, "markets", params=params)

        if isinstance(response, list):
            return response
        return response.get("data", response)

    async def get_all_resolved_markets(self, max_pages: int = 5) -> list[dict]:
        """
        Fetch resolved markets with pagination.

        Args:
            max_pages: Maximum pages to fetch.

        Returns:
            List of resolved markets.
        """
        all_markets = []
        offset = 0
        limit = 100

        for page in range(max_pages):
            markets = await self.get_resolved_markets(limit=limit, offset=offset)

            if not markets:
                break

            all_markets.extend(markets)
            logger.info(f"Fetched resolved page {page + 1}: {len(markets)} markets")

            if len(markets) < limit:
                break

            offset += limit
            await asyncio.sleep(0.5)

        return all_markets

    async def get_all_active_markets(self, max_pages: int = 10) -> list[dict]:
        """
        Fetch all active markets with pagination.

        Args:
            max_pages: Maximum pages to fetch (safety limit).

        Returns:
            List of all active markets.
        """
        all_markets = []
        offset = 0
        limit = 100

        for page in range(max_pages):
            markets = await self.get_gamma_markets(
                active=True,
                closed=False,
                limit=limit,
                offset=offset
            )

            if not markets:
                break

            all_markets.extend(markets)
            logger.info(f"Fetched page {page + 1}: {len(markets)} markets")

            if len(markets) < limit:
                break

            offset += limit
            # Small delay between pages to be respectful
            await asyncio.sleep(0.5)

        return all_markets

    async def get_recent_trades_for_market(
        self,
        condition_id: str,
        since_timestamp: Optional[int] = None
    ) -> list[dict]:
        """
        Get recent trades for a market since a timestamp.

        Args:
            condition_id: Market condition ID.
            since_timestamp: Unix timestamp to fetch trades after.

        Returns:
            List of trades.
        """
        return await self.get_market_trades(
            condition_id=condition_id,
            after=since_timestamp
        )


# Factory function for creating clients
def create_client() -> PolymarketClient:
    """Create a new Polymarket API client."""
    return PolymarketClient()
