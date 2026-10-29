"""Replay adapter for the existing sticky platform risk gate, not a new timing formula."""

import math
from pathlib import Path

from ..costs import execution_price, transaction_fee
from ..execution.paper import PaperBroker
from ..execution.risk_state import constrained_target_weight, evaluate_account_risk_state
from ..models import Side
from .baseline_v2_daily_replay import DailyOpeningPlans


class PlatformRiskDailyPlans(DailyOpeningPlans):
    def __init__(self, *args, risk_database, account_id, **kwargs):
        super().__init__(*args, **kwargs)
        self.risk_database = Path(risk_database)
        self.account_id = account_id
        self.peak = self.config.initial_cash

    def closing(self, date, snapshot):
        self.peak = max(self.peak, float(snapshot["equity"]))
        return super().closing(date, snapshot)

    def evaluate_opening_risk(self, date, snapshot):
        # Opening snapshot contains no today's close. Native PaperBroker persists
        # escalation/recovery state; never auto-resume a previously frozen account.
        broker = PaperBroker(self.risk_database)
        equity = float(snapshot["equity"])
        self.peak = max(self.peak, equity)
        decision = evaluate_account_risk_state(
            self.config,
            equity,
            float(snapshot["cash"]),
            float(snapshot["market_value"]) / equity,
            equity / self.peak - 1,
            "BLOCKED" if self.input_hold else "PASS",
            broker.reconcile(self.account_id).matched,
        )
        state = broker.update_risk_state(
            self.account_id, decision.proposed_state, decision.reason_code, date
        )
        return state

    def opening(self, date, snapshot, quotes):
        state = self.evaluate_opening_risk(date, snapshot)
        if state["effective_state"] != "NORMAL":
            return {
                "status": "ACCOUNT_RISK_BLOCKED",
                "orders": [],
                "risk_state": state,
                "risk_contract": "EXISTING_PLATFORM_BLOCKS_ALL_NON_NORMAL_NO_AUTOMATIC_RECOVERY",
                "peak_known_before_or_at_open": self.peak,
            }
        return super().opening(date, snapshot, quotes)


class PlatformRiskReductionPlans(PlatformRiskDailyPlans):
    """V2-only correction: enforce native risk targets without disabling sells.

    SUSPENDED still freezes everything. Escalations remain sticky and require
    explicit recovery; this adapter does not invent a recovery/timing formula.
    """

    def liquidation_plan(self, date, snapshot, quotes, state):
        """Explicit V2 risk liquidation, never the stock-style optimizer's zero sleeve.

        Replay snapshots contain previous-close shares (T+1 available). Unsettled
        bonus rights are not positions and must not be sold before listing.
        """
        orders, targets = [], {}
        cash = float(snapshot["cash"])
        for position in snapshot["positions"]:
            symbol, quantity = position["symbol"], int(position["quantity"])
            price = float(quotes.loc[symbol, "open"])
            if not math.isfinite(price) or price <= 0:
                raise ValueError("INVALID_RISK_LIQUIDATION_REFERENCE_PRICE")
            targets[symbol] = quantity
            if not quantity or not bool(quotes.loc[symbol, "can_sell"]):
                continue
            executed = execution_price(price, Side.SELL, self.config.slippage_bps)
            fee = transaction_fee(quantity * executed, Side.SELL, self.config)
            orders.append(
                {
                    "symbol": symbol,
                    "side": "SELL",
                    "quantity": quantity,
                    "reference_price": price,
                    "estimated_execution_price": executed,
                    "estimated_fee": fee,
                }
            )
            targets[symbol] = 0
            cash += quantity * executed - fee
        return {
            "status": "PASS_REFERENCE_QUOTES_ONLY",
            "orders": orders,
            "target_quantities": targets,
            "estimated_cash": cash,
            "signal_date": self.context["signal_date"],
            "orders_enabled": False,
            "future_prices_used": False,
            "risk_state": state,
            "risk_contract": "V2_NATIVE_TARGET_REDUCTIONS_NO_AUTO_RECOVERY",
            "locked_rights_not_sold": snapshot.get("unsettled_stock_rights", {}),
        }

    def opening(self, date, snapshot, quotes):
        state = self.evaluate_opening_risk(date, snapshot)
        effective = state["effective_state"]
        if effective == "SUSPENDED":
            return {"status": "ACCOUNT_RISK_BLOCKED", "orders": [], "risk_state": state}
        if effective == "NORMAL":
            return DailyOpeningPlans.opening(self, date, snapshot, quotes)
        if not self.context or self.input_hold:
            return DailyOpeningPlans.opening(self, date, snapshot, quotes)
        # Integer solver must also prohibit buys: a weight ceiling alone can
        # otherwise round up a current odd-lot holding into an additional buy.
        quantities = {p["symbol"]: int(p["quantity"]) for p in snapshot["positions"]}
        locked = snapshot.get("unsettled_stock_rights", {})
        support = set(self.context["weights"]) | set(quantities) | set(locked)
        equity = float(snapshot["equity"])
        constrained = {
            s: constrained_target_weight(
                effective,
                self.context["weights"].get(s, 0),
                (quantities.get(s, 0) + locked.get(s, 0)) * float(quotes.loc[s, "open"]) / equity,
            )
            for s in support
        }
        if not any(constrained.values()):
            return self.liquidation_plan(date, snapshot, quotes, state)
        original = self.context
        self.context = {**original, "weights": constrained, "due": True, "market_wait": {}}
        safe_quotes = quotes.copy()
        safe_quotes["can_buy"] = False
        plan = DailyOpeningPlans.opening(self, date, snapshot, safe_quotes)
        if any(o["side"] == "BUY" for o in plan.get("orders", [])):
            raise ValueError("RISK_REDUCTION_PLAN_ATTEMPTED_BUY")
        plan.update(risk_state=state, risk_contract="V2_NATIVE_TARGET_REDUCTIONS_NO_AUTO_RECOVERY")
        # Never retain unconstrained targets across a risk-triggered run.
        return plan
