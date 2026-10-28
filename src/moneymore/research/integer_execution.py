"""Bounded HiGHS mixed-integer execution planning, research only."""

import numpy as np
import pandas as pd

from moneymore.costs import execution_price, transaction_fee
from moneymore.models import Side


def paper_intents_from_integer_plan(plan, strategy_id, signal_date):
    """Exact quantities for the isolated PaperBroker, not a real-broker adapter."""
    from moneymore.execution.risk import OrderIntent, RiskResult

    if (
        plan.get("status") != "PASS_REFERENCE_QUOTES_ONLY"
        or plan.get("orders_enabled") is not False
    ):
        raise ValueError("ONLY_VALIDATED_PAPER_REFERENCE_PLANS_ACCEPTED")
    return [
        RiskResult(
            True,
            OrderIntent(
                idempotency_key=f"{strategy_id}:{row['symbol']}:{signal_date}:{row['side']}:{row['quantity']}",
                strategy_id=strategy_id,
                symbol=row["symbol"],
                side=Side(row["side"]),
                quantity=row["quantity"],
                signal_date=signal_date,
                reason_code="RESEARCH_INTEGER_PLAN",
            ),
            None,
        )
        for row in plan["orders"]
    ]


def plan_integer_execution(
    targets,
    prices,
    quantities,
    available,
    cash,
    equity,
    config,
    styles,
    style_limits,
    locked_shares=None,
    turnover=0.20,
    can_buy=None,
    can_sell=None,
    time_limit=10.0,
    solver_threads=None,
):
    """Use supplied (known) quotes. Never promises execution at a future open.

    Lexicographic: minimize target tracking error INCLUDING cash, then maximize
    investment among tracking-optimal plans. Style and turnover apply to stock-sleeve
    weights. Exact minimum-commission disjunctions prevent fictitious fee burn.
    """
    import cvxpy as cp

    blocked = lambda reason: {
        "status": "BLOCKED",
        "reason": reason,
        "orders": [],
        "orders_enabled": False,
    }
    targets = pd.Series(targets, dtype=float)
    prices = pd.Series(prices, dtype=float)
    quantities = pd.Series(quantities, dtype=float)
    available = pd.Series(available, dtype=float)
    locked_shares = pd.Series(locked_shares if locked_shares is not None else {}, dtype=float)
    if (
        equity <= 0
        or cash < 0
        or not np.isfinite([cash, equity]).all()
        or not np.isfinite(targets).all()
        or (targets < 0).any()
        or targets.sum() > 1 + 1e-6
    ):
        return blocked("INVALID_ACCOUNT_OR_TARGET")
    dust = targets.loc[targets.le(1e-6)].sum()
    if dust > 1e-5:
        return blocked("NONTRIVIAL_TARGET_MASS_WOULD_BE_EXCLUDED")
    if targets.sum() <= 0:
        return blocked("CASH_ONLY_TARGET_POLICY_NOT_ENABLED")
    symbols = sorted(
        set(targets[targets > 1e-6].index)
        | set(quantities[quantities > 0].index)
        | set(locked_shares[locked_shares > 0].index)
    )
    if not symbols:
        return blocked("EMPTY_SUPPORT")
    if set(symbols) - set(styles.index):
        return blocked("MISSING_STYLE_FOR_REQUIRED_ASSET")
    z = (styles - styles.mean()) / styles.std(ddof=0).replace(0, 1)
    exposure = z.reindex(symbols).to_numpy(float)
    bounds = pd.Series(style_limits).reindex(styles.columns).to_numpy(float)
    p = prices.reindex(symbols).to_numpy(float)
    cur = quantities.reindex(symbols, fill_value=0).to_numpy(float)
    avail = available.reindex(symbols, fill_value=0).to_numpy(float)
    locked = locked_shares.reindex(symbols, fill_value=0).to_numpy(float)
    if (
        not np.isfinite(p).all()
        or (p <= 0).any()
        or not np.isfinite(exposure).all()
        or not np.isfinite(bounds).all()
        or (bounds < 0).any()
        or not np.isfinite(cur).all()
        or (cur < 0).any()
        or not np.equal(cur, np.floor(cur)).all()
        or not np.isfinite(avail).all()
        or (avail < 0).any()
        or (avail > cur).any()
        or not np.equal(avail, np.floor(avail)).all()
        or not np.isfinite(locked).all()
        or (locked < 0).any()
    ):
        return blocked("INVALID_QUOTES_QUANTITIES_OR_STYLE_BOUNDS")
    n = len(symbols)
    lot = config.lot_size
    min_buy = np.array([max(200, lot) if s.startswith(("688", "689")) else lot for s in symbols])
    min_sell = np.array(
        [max(1, min(200, q)) if s.startswith(("688", "689")) else 1 for s, q in zip(symbols, cur)]
    )
    odd = cur % lot
    upper_lots = np.floor(config.max_position_weight * equity / p / lot)
    big_shares = np.maximum(cur, (upper_lots + 1) * lot + odd)
    lots = cp.Variable(n, integer=True)
    retain_odd = cp.Variable(n, boolean=True)
    buy = cp.Variable(n, nonneg=True)
    sell = cp.Variable(n, nonneg=True)
    buy_on, sell_on = cp.Variable(n, boolean=True), cp.Variable(n, boolean=True)
    buy_high, sell_high = cp.Variable(n, boolean=True), cp.Variable(n, boolean=True)
    buy_comm, sell_comm = cp.Variable(n, nonneg=True), cp.Variable(n, nonneg=True)
    q = lot * lots + cp.multiply(odd, retain_odd)
    buy_price = p * (1 + config.slippage_bps / 10000)
    sell_price = p * (1 - config.slippage_bps / 10000)
    buy_notional, sell_notional = cp.multiply(buy_price, buy), cp.multiply(sell_price, sell)
    fees = cp.sum(
        buy_comm
        + sell_comm
        + config.transfer_fee_rate * (buy_notional + sell_notional)
        + config.stamp_duty_rate * sell_notional
    )
    cash_end = cash + cp.sum(sell_notional - buy_notional) - fees
    values = cp.multiply(p, q + locked)
    invested = cp.sum(values)
    constraints = [
        lots >= 0,
        lots <= upper_lots,
        q - cur == buy - sell,
        buy <= cp.multiply(big_shares, buy_on),
        sell <= cp.multiply(avail, sell_on),
        buy >= cp.multiply(min_buy, buy_on),
        sell >= cp.multiply(min_sell, sell_on),
        buy_on + sell_on <= 1,
        values <= config.max_position_weight * equity,
        cash_end >= 0,
        invested <= min(float(targets.sum()), config.max_gross_exposure) * equity,
        invested >= 0.01,
    ]
    # Conservative single MARKET-order caps; splitting would incur extra fees.
    order_caps = np.array([50000 if s.startswith(("688", "689")) else 1000000 for s in symbols])
    constraints += [
        buy <= cp.multiply(order_caps, buy_on),
        sell <= cp.multiply(order_caps, sell_on),
    ]
    for i in range(n):
        if odd[i] > 0:
            constraints.append(lots[i] <= upper_lots[i] * retain_odd[i])
        else:
            constraints.append(retain_odd[i] == 0)
        if can_buy is not None and not can_buy.get(symbols[i], False):
            constraints.append(buy[i] == 0)
        if can_sell is not None and not can_sell.get(symbols[i], False):
            constraints.append(sell[i] == 0)
    for comm, notional, on, high, execution_p in (
        (buy_comm, buy_notional, buy_on, buy_high, buy_price),
        (sell_comm, sell_notional, sell_on, sell_high, sell_price),
    ):
        linear = config.commission_rate * notional
        max_linear = config.commission_rate * big_shares * execution_p
        constraints += [
            comm >= linear,
            comm >= config.minimum_commission * on,
            comm <= linear + config.minimum_commission * (1 - high),
            comm <= config.minimum_commission * on + cp.multiply(max_linear, high),
        ]
    initial_value = p * (cur + locked)
    if initial_value.sum() > 0:
        w0 = initial_value / initial_value.sum()
        constraints.append(cp.norm1(values - cp.multiply(w0, invested)) <= turnover * invested)
    benchmark = z.mean().to_numpy()
    active_value = (exposure - benchmark).T @ values
    constraints += [
        active_value <= cp.multiply(bounds, invested),
        active_value >= -cp.multiply(bounds, invested),
    ]
    options = {"time_limit": time_limit, "mip_rel_gap": 0.0}
    if solver_threads is not None:
        options["threads"] = max(1, min(4, int(solver_threads)))
    primary_optimal_fallback = False
    secondary_status = None
    try:
        target = targets.reindex(symbols, fill_value=0).to_numpy()
        tracking = cp.norm1(values / equity - target) + cp.abs(
            cash_end / equity - (1 - targets.sum())
        )
        first = cp.Problem(cp.Minimize(tracking), constraints)
        first.solve(solver=cp.HIGHS, **options)
        if first.status != "optimal":
            return blocked("INTEGER_FEASIBILITY_" + first.status.upper())
        best_tracking = float(tracking.value)
        primary_values = [(v, np.array(v.value, copy=True)) for v in first.variables()]
        constraints.append(tracking <= best_tracking + 1e-8)
        second = cp.Problem(cp.Maximize(invested), constraints)
        try:
            second.solve(solver=cp.HIGHS, **options)
            secondary_status = second.status
        except cp.error.SolverError:
            secondary_status = "solver_error"
        if secondary_status != "optimal":
            # A secondary tie-break must not erase an already optimal primary
            # plan. Restore it, then run the SAME exact lot/cash/style checks.
            # This is not permission to use an infeasible or timed-out primary.
            for variable, value in primary_values:
                variable.save_value(value)
            primary_optimal_fallback = True
    except cp.error.SolverError:
        return blocked("INTEGER_SOLVER_ERROR")
    if q.value is None or not np.isfinite(q.value).all():
        return blocked("NONFINITE_INTEGER_SOLUTION")
    planned = np.rint(q.value).astype(int)
    orders, expected_cash, exact_fees = [], float(cash), 0.0
    for symbol, price, current, final, tradable in zip(symbols, p, cur.astype(int), planned, avail):
        difference = int(final - current)
        if not difference:
            continue
        side = Side.BUY if difference > 0 else Side.SELL
        count = abs(difference)
        if (
            count > order_caps[symbols.index(symbol)]
            or (side == Side.BUY and can_buy is not None and not can_buy.get(symbol, False))
            or (side == Side.SELL and can_sell is not None and not can_sell.get(symbol, False))
        ):
            return blocked("INTEGER_POSTCHECK_MARKET_OR_ORDER_CAP")
        star = symbol.startswith(("688", "689"))
        if (side == Side.BUY and (count % lot != 0 or (star and count < 200))) or (
            side == Side.SELL
            and (
                count > tradable
                or (count % lot != 0 and final != 0)
                or (star and count < 200 and final != 0)
            )
        ):
            return blocked("INTEGER_POSTCHECK_TRADING_RULE")
        executed = execution_price(price, side, config.slippage_bps)
        notional = count * executed
        fee = transaction_fee(notional, side, config)
        exact_fees += fee
        expected_cash += (-notional if side == Side.BUY else notional) - fee
        orders.append(
            {
                "symbol": symbol,
                "side": side.value,
                "quantity": count,
                "reference_price": float(price),
                "estimated_execution_price": float(executed),
                "estimated_fee": float(fee),
            }
        )
    actual_values = (planned + locked) * p
    actual_invested = float(actual_values.sum())
    if (
        actual_invested <= 0
        or not np.isfinite(
            [actual_invested, expected_cash, exact_fees, float(cash_end.value)]
        ).all()
    ):
        return blocked("NONFINITE_OR_EMPTY_EXECUTABLE_PORTFOLIO")
    active = (actual_values / actual_invested) @ (exposure - benchmark)
    actual_turnover = (
        float(np.abs(actual_values / actual_invested - initial_value / initial_value.sum()).sum())
        if initial_value.sum()
        else 0.0
    )
    if not np.isfinite(active).all() or not np.isfinite(actual_turnover):
        return blocked("NONFINITE_EXECUTABLE_STYLE_OR_TURNOVER")
    if (
        expected_cash < -0.01
        or abs(expected_cash - float(cash_end.value)) > 0.01
        or (actual_values > config.max_position_weight * equity + 0.01).any()
        or actual_invested > min(float(targets.sum()), config.max_gross_exposure) * equity + 0.01
        or actual_turnover > turnover + 1e-5
        or (np.abs(active) > bounds + 1e-5).any()
    ):
        return {
            **blocked("INTEGER_POSTCHECK_CASH_COST_WEIGHT_OR_STYLE"),
            "postcheck": {
                "actual_cash": expected_cash,
                "solver_cash": float(cash_end.value),
                "cash_difference": expected_cash - float(cash_end.value),
                "maximum_weight_excess_currency": float(
                    np.max(actual_values - config.max_position_weight * equity)
                ),
                "turnover_l1": actual_turnover,
                "maximum_style_excess": float(np.max(np.abs(active) - bounds)),
                "fee_rows": [
                    {
                        "symbol": symbols[i],
                        "delta": int(planned[i] - cur[i]),
                        "buy": float(buy.value[i]),
                        "sell": float(sell.value[i]),
                        "buy_flag": float(buy_on.value[i]),
                        "sell_flag": float(sell_on.value[i]),
                        "buy_commission": float(buy_comm.value[i]),
                        "sell_commission": float(sell_comm.value[i]),
                    }
                    for i in range(n)
                    if buy_comm.value[i] + sell_comm.value[i] > 0.01
                ],
            },
        }
    return {
        "status": "PASS_REFERENCE_QUOTES_ONLY",
        "primary_optimal_fallback": primary_optimal_fallback,
        "secondary_optimizer_status": secondary_status,
        "orders": sorted(
            orders,
            key=lambda o: (
                o["side"] != "SELL",
                -(o["quantity"] * o["estimated_execution_price"] - o["estimated_fee"])
                if o["side"] == "SELL"
                else 0,
                o["symbol"],
            ),
        ),
        "target_quantities": dict(zip(symbols, map(int, planned))),
        "estimated_cash": expected_cash,
        "estimated_fees": exact_fees,
        "estimated_gross": actual_invested / equity,
        "turnover_l1": actual_turnover,
        "active_style_exposures": dict(zip(styles.columns, map(float, active))),
        "tracking_optimal_stock_value": actual_invested,
        "tracking_l1": float(tracking.value),
        "excluded_dust_target_mass": float(dust),
        "orders_enabled": False,
        "future_prices_used": False,
        "warning": "Only valid at supplied quotes and availability; requires revalidation before execution. No real order is submitted.",
    }
