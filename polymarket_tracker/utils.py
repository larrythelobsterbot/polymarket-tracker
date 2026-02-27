"""
Utility functions for Polymarket Tracker.

This module provides helper functions for common operations
like data formatting, validation, and conversions.
"""

import json
import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float.

    Args:
        value: Value to convert.
        default: Default value if conversion fails.

    Returns:
        Float value or default.
    """
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_decimal(value: Any, default: Decimal = Decimal("0")) -> Decimal:
    """
    Safely convert a value to Decimal.

    Args:
        value: Value to convert.
        default: Default value if conversion fails.

    Returns:
        Decimal value or default.
    """
    if value is None:
        return default
    try:
        return Decimal(str(value))
    except (ValueError, TypeError):
        return default


def parse_timestamp(
    value: Union[str, int, float, None],
    default: Optional[datetime] = None
) -> Optional[datetime]:
    """
    Parse various timestamp formats to datetime.

    Handles:
    - ISO format strings
    - Unix timestamps (seconds or milliseconds)

    Args:
        value: Timestamp value to parse.
        default: Default value if parsing fails.

    Returns:
        Parsed datetime or default.
    """
    if value is None:
        return default

    try:
        # Try ISO format first
        if isinstance(value, str):
            # Handle various ISO formats
            value = value.replace("Z", "+00:00")
            return datetime.fromisoformat(value)

        # Try Unix timestamp
        ts = float(value)
        if ts > 1e12:  # Milliseconds
            ts = ts / 1000
        return datetime.fromtimestamp(ts)

    except (ValueError, TypeError, OSError):
        return default


def truncate_address(address: str, length: int = 8) -> str:
    """
    Truncate a wallet address for display.

    Args:
        address: Full wallet address.
        length: Number of characters to show on each side.

    Returns:
        Truncated address like "0x1234...abcd".
    """
    if not address or len(address) <= length * 2 + 3:
        return address
    return f"{address[:length]}...{address[-length:]}"


def format_volume(volume: float) -> str:
    """
    Format trading volume for display.

    Args:
        volume: Volume in dollars.

    Returns:
        Formatted string like "$1.23M" or "$123.45K".
    """
    if volume >= 1_000_000:
        return f"${volume / 1_000_000:.2f}M"
    elif volume >= 1_000:
        return f"${volume / 1_000:.2f}K"
    else:
        return f"${volume:.2f}"


def json_dumps_safe(obj: Any) -> str:
    """
    Safely serialize an object to JSON.

    Handles Decimal, datetime, and other non-serializable types.

    Args:
        obj: Object to serialize.

    Returns:
        JSON string.
    """
    def default_serializer(o):
        if isinstance(o, Decimal):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    return json.dumps(obj, default=default_serializer)


def chunk_list(lst: list, chunk_size: int) -> list[list]:
    """
    Split a list into chunks.

    Args:
        lst: List to split.
        chunk_size: Maximum items per chunk.

    Returns:
        List of chunks.
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def is_valid_wallet_address(address: str) -> bool:
    """
    Validate an Ethereum/Polygon wallet address.

    Args:
        address: Address to validate.

    Returns:
        True if valid.
    """
    if not address:
        return False
    if not address.startswith("0x"):
        return False
    if len(address) != 42:
        return False
    try:
        int(address, 16)
        return True
    except ValueError:
        return False


def calculate_pnl(
    entry_price: float,
    current_price: float,
    size: float,
    side: str
) -> float:
    """
    Calculate profit/loss for a position.

    Args:
        entry_price: Average entry price.
        current_price: Current market price.
        size: Position size.
        side: "BUY" or "SELL".

    Returns:
        Profit/loss amount.
    """
    if side.upper() == "BUY":
        return (current_price - entry_price) * size
    else:
        return (entry_price - current_price) * size
