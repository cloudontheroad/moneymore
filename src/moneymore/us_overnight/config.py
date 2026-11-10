from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel, Field, model_validator


class USOvernightConfig(BaseModel):
    """Configuration deliberately incompatible with an execution account."""

    mode: str = "US_PAPER_ONLY"
    opend_host: str = "127.0.0.1"
    opend_port: int = Field(default=11111, gt=0, le=65535)
    timezone: str = "America/New_York"
    signal_cutoff: str = "15:50"
    entry_window: tuple[str, str] = ("15:55", "16:00")
    exit_window: tuple[str, str] = ("09:30", "09:35")
    top_n: int = Field(default=10, gt=0)
    universe_top_n: int = Field(default=1000, gt=0)
    universe_ranking: str = "total_market_val_desc"
    round_trip_cost_bps: float = Field(default=20.0, ge=0)
    output_root: Path = Path("data/us_overnight_paper_only")
    minimum_price: float = Field(default=5.0, gt=0)
    minimum_median_dollar_volume_20d: float = Field(default=20_000_000, gt=0)
    minimum_listing_days: int = Field(default=120, gt=0)

    @model_validator(mode="after")
    def paper_only(self) -> USOvernightConfig:
        if self.mode != "US_PAPER_ONLY":
            raise ValueError("US overnight phase one is permanently US_PAPER_ONLY")
        ZoneInfo(self.timezone)
        if self.signal_cutoff >= self.entry_window[0]:
            raise ValueError("signal cutoff must precede the entry window")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> USOvernightConfig:
        with Path(path).open(encoding="utf-8") as handle:
            return cls.model_validate(yaml.safe_load(handle))
