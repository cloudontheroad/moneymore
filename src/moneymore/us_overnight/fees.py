"""Public Futu HK fixed-plan fees for U.S. listed stocks, in USD."""

from decimal import ROUND_HALF_UP, Decimal

FUTU_HK_US_FIXED = "futu_hk_us_fixed"


def futu_hk_us_order_fee(shares: float, price: float, *, sell: bool = False) -> float:
    """Estimate one order filled once using the paper ledger's share quantity."""
    quantity = Decimal(str(shares))
    if not quantity:
        return 0.0
    value = quantity * Decimal(str(price))
    cent = Decimal("0.01")
    cap = value * Decimal("0.005")
    fees = [
        max(Decimal("0.99"), min(cap, quantity * Decimal("0.0049"))),
        max(Decimal("1.00"), min(cap, quantity * Decimal("0.005"))),
        quantity * Decimal("0.003"),
        quantity * Decimal("0.000003"),
    ]
    if sell:
        fees += [
            max(cent, value * Decimal("0.0000206")),
            min(Decimal("9.79"), max(cent, quantity * Decimal("0.000195"))),
        ]
    return float(sum(fee.quantize(cent, rounding=ROUND_HALF_UP) for fee in fees))


def settle_futu_position(position: dict, exit_open: float, dividend: float) -> dict:
    shares = position["shares"]
    start = position["entry_open"]
    buy_fee = futu_hk_us_order_fee(shares, start)
    sell_fee = futu_hk_us_order_fee(shares, exit_open, sell=True)
    cost = round(buy_fee + sell_fee, 2)
    pnl = shares * (exit_open + dividend - start) - cost
    return position | {
        "exit_open": exit_open, "dividend_per_share": dividend,
        "net_return": pnl / position["allocation_usd"], "pnl_usd": pnl,
        "cost_usd": cost, "buy_fee_usd": buy_fee, "sell_fee_usd": sell_fee,
    }


def futu_book_totals(positions: list[dict], capital: float) -> dict:
    pnl = sum(row["pnl_usd"] for row in positions)
    return {
        "realized": positions, "fee_model": FUTU_HK_US_FIXED,
        "cost_usd": round(sum(row["cost_usd"] for row in positions), 2),
        "capital_usd": capital, "pnl_usd": pnl, "daily_return": pnl / capital,
        "top1_net_return": positions[0]["net_return"] if positions else 0.0,
        "top10_net_return": sum(row["net_return"] for row in positions) / len(positions) if positions else 0.0,
    }
