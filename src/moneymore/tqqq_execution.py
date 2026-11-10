"""Shared integer-share sizing and next-open fills for the US ETF sleeve."""

from dataclasses import dataclass


def share_delta(equity: float, target_weight: float, close: float, held: int) -> int:
    return int(equity * target_weight // close) - held


@dataclass(frozen=True)
class Fill:
    quantity: int
    price: float
    fee: float
    cash_delta: float


def fill_at_open(
    side: str, requested: int, opening: float, cash: float, held: int,
    *, slippage: float = 0.0005, fee_rate: float = 0.0001,
) -> Fill:
    buy = side == "BUY"
    price = opening * (1 + slippage if buy else 1 - slippage)
    available = max(0, int(cash // (price * (1 + fee_rate)))) if buy else held
    quantity = min(requested, available)
    fee = quantity * price * fee_rate
    cash_delta = -quantity * price - fee if buy else quantity * price - fee
    return Fill(quantity, price, fee, cash_delta)
