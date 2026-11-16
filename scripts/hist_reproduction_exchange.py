"""Research-only Qlib cash guard; no modification of the installed exchange."""
import math

from qlib.backtest.decision import Order
from qlib.backtest.exchange import Exchange


def cash_safe_buy(amount, price, cash, rate, minimum_cost, lot):
    """Keep affordable fills unchanged; floor unaffordable buys to whole lots.

    Qlib adds 0.1 raw shares when rounding. At the cash boundary this can
    admit a lot whose value plus commission exceeds cash. Do not waive fees,
    clamp the ledger, change target weights, or apply the tolerance here.
    """
    if not all(math.isfinite(x) for x in (amount, price, cash, rate, minimum_cost)):
        raise ValueError("Non-finite cash guard input")
    if amount < 0 or price <= 0 or rate < 0 or minimum_cost < 0 or cash < -1e-5:
        raise ValueError("Invalid cash guard input")
    if lot is not None and (not math.isfinite(lot) or lot <= 0):
        raise ValueError("Invalid lot")

    def cost(qty):
        value = qty * price
        return max(minimum_cost, value * rate) if value > 1e-5 else 0.

    if amount * price + cost(amount) <= cash:
        return amount, amount * price, cost(amount)
    affordable = max(0., min(amount, (cash-minimum_cost)/price, cash/(1+rate)/price))
    qty = math.floor(affordable/lot)*lot if lot else math.nextafter(affordable, 0.)
    # A binary division/multiplication at an exact boundary may round upward.
    if qty * price + cost(qty) > cash:
        qty = max(0., qty-lot) if lot else math.nextafter(qty, 0.)
    value, fee = qty * price, cost(qty)
    if value + fee > cash and value > 0:
        raise ValueError("Cash guard could not produce an affordable fill")
    return qty, value, fee


class CashSafeExchange(Exchange):
    def _calc_trade_info_by_order(self, order, position, dealt_order_amount):
        order.cash_guard_overspend = 0.
        price, value, fee = super()._calc_trade_info_by_order(order, position, dealt_order_amount)
        if order.direction == Order.BUY and position is not None and value+fee > position.get_cash():
            if self.impact_cost != 0:
                raise ValueError("HIST cash guard requires the frozen zero-impact protocol")
            order.cash_guard_overspend = value+fee-position.get_cash()
            lot = self.get_amount_of_trade_unit(factor=order.factor)
            order.deal_amount, value, fee = cash_safe_buy(
                order.deal_amount, price, position.get_cash(), self.open_cost, self.min_cost, lot)
        return price, value, fee
