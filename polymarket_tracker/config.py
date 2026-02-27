"""
Configuration management for Polymarket Tracker.

Uses pydantic-settings to load configuration from environment variables
with sensible defaults for development.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Database configuration
    database_path: str = "polymarket_data.db"

    # API Configuration (for authenticated CLOB endpoints)
    polymarket_api_key: str = ""
    polymarket_api_secret: str = ""
    polymarket_passphrase: str = ""

    # API Base URLs
    clob_api_url: str = "https://clob.polymarket.com"
    gamma_api_url: str = "https://gamma-api.polymarket.com"
    data_api_url: str = "https://data-api.polymarket.com"

    # Chain configuration (137 = Polygon mainnet)
    chain_id: int = 137

    # Data collection settings
    fetch_interval_minutes: int = 5
    max_retries: int = 3
    request_timeout_seconds: int = 30

    # Trade filtering settings
    min_trade_size: float = 100.0  # Minimum trade size in dollars (default: $100)
    min_market_volume: float = 10000.0  # Minimum market volume to sync (default: $10k)

    # Rate limiting (requests per 10 seconds)
    # Conservative defaults below Polymarket's limits
    rate_limit_trades: int = 180
    rate_limit_positions: int = 130
    rate_limit_markets: int = 280

    # Logging
    log_level: str = "INFO"

    @property
    def database_url(self) -> str:
        """Get the SQLite database URL."""
        return f"sqlite:///{self.database_path}"

    @property
    def has_api_credentials(self) -> bool:
        """Check if API credentials are configured."""
        return bool(self.polymarket_api_key and self.polymarket_api_secret and self.polymarket_passphrase)


# Global settings instance
settings = Settings()
