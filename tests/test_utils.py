"""
Tests for utility functions.
"""

from datetime import datetime
from decimal import Decimal

import pytest

from polymarket_tracker.utils import (
    safe_float,
    safe_decimal,
    parse_timestamp,
    truncate_address,
    format_volume,
    is_valid_wallet_address,
    calculate_pnl,
    chunk_list,
)


class TestSafeFloat:
    """Tests for safe_float function."""

    def test_valid_float(self):
        assert safe_float(3.14) == 3.14

    def test_valid_string(self):
        assert safe_float("3.14") == 3.14

    def test_valid_int(self):
        assert safe_float(42) == 42.0

    def test_none_returns_default(self):
        assert safe_float(None) == 0.0
        assert safe_float(None, 99.0) == 99.0

    def test_invalid_returns_default(self):
        assert safe_float("invalid") == 0.0
        assert safe_float("invalid", -1.0) == -1.0


class TestSafeDecimal:
    """Tests for safe_decimal function."""

    def test_valid_decimal(self):
        assert safe_decimal("3.14159") == Decimal("3.14159")

    def test_valid_int(self):
        assert safe_decimal(42) == Decimal("42")

    def test_none_returns_default(self):
        assert safe_decimal(None) == Decimal("0")

    def test_invalid_returns_default(self):
        assert safe_decimal("invalid") == Decimal("0")


class TestParseTimestamp:
    """Tests for parse_timestamp function."""

    def test_iso_format(self):
        result = parse_timestamp("2024-01-15T10:30:00")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_iso_with_z(self):
        result = parse_timestamp("2024-01-15T10:30:00Z")
        assert result is not None

    def test_unix_timestamp_seconds(self):
        # 1705315800 = 2024-01-15 10:30:00 UTC
        result = parse_timestamp(1705315800)
        assert result is not None
        assert result.year == 2024

    def test_unix_timestamp_milliseconds(self):
        result = parse_timestamp(1705315800000)
        assert result is not None
        assert result.year == 2024

    def test_none_returns_default(self):
        assert parse_timestamp(None) is None
        default = datetime(2000, 1, 1)
        assert parse_timestamp(None, default) == default


class TestTruncateAddress:
    """Tests for truncate_address function."""

    def test_standard_address(self):
        address = "0x1234567890abcdef1234567890abcdef12345678"
        result = truncate_address(address)
        assert result == "0x123456...12345678"

    def test_short_address(self):
        address = "0x1234"
        result = truncate_address(address)
        assert result == "0x1234"  # Not truncated

    def test_custom_length(self):
        address = "0x1234567890abcdef1234567890abcdef12345678"
        result = truncate_address(address, length=4)
        assert result == "0x12...5678"


class TestFormatVolume:
    """Tests for format_volume function."""

    def test_millions(self):
        assert format_volume(1500000) == "$1.50M"
        assert format_volume(1000000) == "$1.00M"

    def test_thousands(self):
        assert format_volume(1500) == "$1.50K"
        assert format_volume(1000) == "$1.00K"

    def test_small_amounts(self):
        assert format_volume(123.45) == "$123.45"
        assert format_volume(0.50) == "$0.50"


class TestIsValidWalletAddress:
    """Tests for is_valid_wallet_address function."""

    def test_valid_address(self):
        assert is_valid_wallet_address("0x1234567890abcdef1234567890abcdef12345678")
        assert is_valid_wallet_address("0xABCDEF1234567890abcdef1234567890ABCDEF12")

    def test_invalid_no_prefix(self):
        assert not is_valid_wallet_address("1234567890abcdef1234567890abcdef12345678")

    def test_invalid_wrong_length(self):
        assert not is_valid_wallet_address("0x1234")
        assert not is_valid_wallet_address("0x" + "1" * 50)

    def test_invalid_characters(self):
        assert not is_valid_wallet_address("0xGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG")

    def test_empty_and_none(self):
        assert not is_valid_wallet_address("")
        assert not is_valid_wallet_address(None)


class TestCalculatePnl:
    """Tests for calculate_pnl function."""

    def test_buy_profit(self):
        # Bought at 0.5, now at 0.8, size 100
        pnl = calculate_pnl(0.5, 0.8, 100, "BUY")
        assert pnl == 30.0

    def test_buy_loss(self):
        # Bought at 0.8, now at 0.5, size 100
        pnl = calculate_pnl(0.8, 0.5, 100, "BUY")
        assert pnl == -30.0

    def test_sell_profit(self):
        # Sold at 0.8, now at 0.5, size 100 (price went down = profit for seller)
        pnl = calculate_pnl(0.8, 0.5, 100, "SELL")
        assert pnl == 30.0

    def test_sell_loss(self):
        # Sold at 0.5, now at 0.8, size 100 (price went up = loss for seller)
        pnl = calculate_pnl(0.5, 0.8, 100, "SELL")
        assert pnl == -30.0


class TestChunkList:
    """Tests for chunk_list function."""

    def test_even_chunks(self):
        result = chunk_list([1, 2, 3, 4, 5, 6], 2)
        assert result == [[1, 2], [3, 4], [5, 6]]

    def test_uneven_chunks(self):
        result = chunk_list([1, 2, 3, 4, 5], 2)
        assert result == [[1, 2], [3, 4], [5]]

    def test_single_chunk(self):
        result = chunk_list([1, 2, 3], 10)
        assert result == [[1, 2, 3]]

    def test_empty_list(self):
        result = chunk_list([], 5)
        assert result == []
