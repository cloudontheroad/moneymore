from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field, field_validator


class UniverseMember(BaseModel):
    symbol: str
    session_date: date
    known_at: datetime
    price: float = Field(gt=0)
    median_dollar_volume_20d: float = Field(ge=0)
    listing_days: int = Field(ge=0)
    security_type: str = "STOCK"
    is_adr: bool = False

    @field_validator("known_at")
    @classmethod
    def timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("known_at must be timezone-aware")
        return value


class RiskDecision(BaseModel):
    allowed: bool
    scale: float = Field(ge=0, le=1)
    flags: tuple[str, ...] = ()
