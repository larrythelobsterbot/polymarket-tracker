"""
Pydantic models for Polymarket data structures.

These models represent the data returned by the Polymarket APIs
and are used for validation and serialization.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional
from pydantic import BaseModel, Field


class Token(BaseModel):
    """Represents a market outcome token."""

    token_id: str
    outcome: str
    price: Optional[float] = None
    winner: Optional[bool] = None


class Market(BaseModel):
    """Represents a prediction market from the Gamma API."""

    condition_id: str = Field(alias="conditionId")
    question_id: Optional[str] = Field(default=None, alias="questionId")
    question: str
    description: Optional[str] = None
    end_date: Optional[datetime] = Field(default=None, alias="endDateIso")
    game_start_time: Optional[datetime] = Field(default=None, alias="gameStartTime")
    active: bool = True
    closed: bool = False
    archived: bool = False
    accepting_orders: bool = Field(default=True, alias="acceptingOrders")
    enable_order_book: bool = Field(default=True, alias="enableOrderBook")
    minimum_order_size: Optional[float] = Field(default=None, alias="minimumOrderSize")
    minimum_tick_size: Optional[float] = Field(default=None, alias="minimumTickSize")
    resolved: bool = False
    outcome: Optional[str] = None
    tokens: list[Token] = []
    volume: Optional[float] = None
    volume_24hr: Optional[float] = Field(default=None, alias="volume24hr")
    liquidity: Optional[float] = None
    spread: Optional[float] = None
    outcome_prices: Optional[str] = Field(default=None, alias="outcomePrices")
    clob_token_ids: Optional[str] = Field(default=None, alias="clobTokenIds")
    slug: Optional[str] = None
    category: Optional[str] = None
    tags: list[str] = []
    created_at: Optional[datetime] = Field(default=None, alias="createdAt")
    updated_at: Optional[datetime] = Field(default=None, alias="updatedAt")

    class Config:
        populate_by_name = True


class Trade(BaseModel):
    """Represents a trade from the CLOB API."""

    id: str
    taker_order_id: Optional[str] = Field(default=None, alias="takerOrderId")
    market: str  # condition_id
    asset_id: str = Field(alias="assetId")
    side: str  # BUY or SELL
    size: str  # Size as string (convert to Decimal for calculations)
    fee_rate_bps: str = Field(alias="feeRateBps")
    price: str
    status: str
    match_time: Optional[str] = Field(default=None, alias="matchTime")
    last_update: Optional[str] = Field(default=None, alias="lastUpdate")
    outcome: str
    bucket_index: Optional[int] = Field(default=None, alias="bucketIndex")
    owner: Optional[str] = None
    maker_address: Optional[str] = Field(default=None, alias="makerAddress")
    transaction_hash: Optional[str] = Field(default=None, alias="transactionHash")
    trader_side: Optional[str] = Field(default=None, alias="traderSide")
    type: Optional[str] = None

    class Config:
        populate_by_name = True

    @property
    def size_decimal(self) -> Decimal:
        """Get size as Decimal for precise calculations."""
        return Decimal(self.size)

    @property
    def price_decimal(self) -> Decimal:
        """Get price as Decimal for precise calculations."""
        return Decimal(self.price)

    @property
    def match_timestamp(self) -> Optional[datetime]:
        """Parse match_time to datetime."""
        if self.match_time:
            try:
                # Handle Unix timestamp (seconds or milliseconds)
                ts = float(self.match_time)
                if ts > 1e12:  # Milliseconds
                    ts = ts / 1000
                return datetime.fromtimestamp(ts)
            except (ValueError, TypeError):
                pass
        return None


class Position(BaseModel):
    """Represents a user's position in a market."""

    asset_id: str = Field(alias="assetId")
    condition_id: str = Field(alias="conditionId")
    size: str
    avg_price: str = Field(alias="avgPrice")
    initial_value: str = Field(alias="initialValue")
    current_value: str = Field(alias="currentValue")
    pnl: str
    realized_pnl: str = Field(alias="realizedPnl")
    unrealized_pnl: str = Field(alias="unrealizedPnl")
    cursor_id: Optional[str] = Field(default=None, alias="cursorId")
    outcome: str
    market_slug: Optional[str] = Field(default=None, alias="marketSlug")
    title: Optional[str] = None
    end_date: Optional[str] = Field(default=None, alias="endDate")
    price: Optional[str] = None
    proxyWallet: Optional[str] = None

    class Config:
        populate_by_name = True

    @property
    def size_decimal(self) -> Decimal:
        """Get size as Decimal for precise calculations."""
        return Decimal(self.size)

    @property
    def avg_price_decimal(self) -> Decimal:
        """Get average price as Decimal."""
        return Decimal(self.avg_price)


class MarketTradeEvent(BaseModel):
    """Represents a trade event for a specific market."""

    id: str
    market: str
    asset_id: str = Field(alias="assetId")
    side: str
    size: str
    price: str
    outcome: str
    timestamp: Optional[str] = None
    transaction_hash: Optional[str] = Field(default=None, alias="transactionHash")
    maker_address: Optional[str] = Field(default=None, alias="makerAddress")
    taker_address: Optional[str] = Field(default=None, alias="takerAddress")

    class Config:
        populate_by_name = True


class PaginatedResponse(BaseModel):
    """Generic paginated response wrapper."""

    data: list
    next_cursor: Optional[str] = Field(default=None, alias="nextCursor")
    limit: int = 100

    class Config:
        populate_by_name = True
