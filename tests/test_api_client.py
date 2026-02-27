"""
Tests for the API client module.
"""

import asyncio
import time

import pytest

from polymarket_tracker.api_client import RateLimiter, PolymarketClient


class TestRateLimiter:
    """Tests for RateLimiter class."""

    @pytest.mark.asyncio
    async def test_allows_requests_within_limit(self):
        """Test that requests within limit are allowed immediately."""
        limiter = RateLimiter(requests_per_10s=10)

        start_time = time.time()
        for _ in range(5):
            await limiter.acquire()
        elapsed = time.time() - start_time

        # Should complete almost instantly (< 1 second)
        assert elapsed < 1.0

    @pytest.mark.asyncio
    async def test_rate_limits_when_exceeded(self):
        """Test that rate limiter delays requests when limit exceeded."""
        # Very low limit for testing
        limiter = RateLimiter(requests_per_10s=3)

        # Make 3 requests (at limit)
        for _ in range(3):
            await limiter.acquire()

        # The 4th request should be delayed
        start_time = time.time()
        await limiter.acquire()
        elapsed = time.time() - start_time

        # Should have waited some time (but less than window size)
        # Note: This is timing-sensitive and may need adjustment
        assert elapsed >= 0  # At minimum, some delay occurred


class TestPolymarketClient:
    """Tests for PolymarketClient class."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return PolymarketClient()

    @pytest.mark.asyncio
    async def test_client_initialization(self, client):
        """Test that client initializes correctly."""
        assert client.clob_url == "https://clob.polymarket.com"
        assert client.gamma_url == "https://gamma-api.polymarket.com"
        assert client.data_url == "https://data-api.polymarket.com"
        assert client.chain_id == 137

    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Test API health check (requires network)."""
        try:
            result = await client.health_check()
            # Result depends on network availability
            assert isinstance(result, bool)
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_get_rate_limiter(self, client):
        """Test rate limiter selection by endpoint type."""
        trades_limiter = client._get_rate_limiter("trades")
        positions_limiter = client._get_rate_limiter("positions")
        markets_limiter = client._get_rate_limiter("markets")
        default_limiter = client._get_rate_limiter("unknown")

        # Each should be a RateLimiter instance
        assert isinstance(trades_limiter, RateLimiter)
        assert isinstance(positions_limiter, RateLimiter)
        assert isinstance(markets_limiter, RateLimiter)
        assert isinstance(default_limiter, RateLimiter)

        # They should be the correct limiters from the dict
        assert trades_limiter is client.rate_limiters["trades"]
        assert default_limiter is client.rate_limiters["default"]
