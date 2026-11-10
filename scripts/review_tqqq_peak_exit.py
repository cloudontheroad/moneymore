"""Test early exits after a recent high, preserving the weekly reentry gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from review_tqqq_reentry_payoff import elevated_loss, forecasts, forward_payoffs, inputs, only_reentry
from review_tqqq_sharpe import daily_returns, paired_interval, sharpe
from review_tqqq_swing import replay
from review_tqqq_swing_round_trips import path_metrics
from review_tqqq_turning_points import price_signals
from review_tqqq_weekly_cycle import execution_state

START = "2014-10-31"
CASE_PEAK = "2026-10-06"


def recent_high(close):
    return close.rolling(63).apply(np.argmax, raw=True).rsub(62)


def downside_accumulator(returns, volatility, new_high):
    """Page-style one-sided statistic on standardized losses."""
    value = 0.0
    result = []
    for ret, vol, high in zip(returns, volatility, new_high, strict=True):
        if high or not np.isfinite(vol) or not np.isfinite(ret) or vol <= 0:
            value = 0.0
        else:
            value = max(0.0, value - ret / vol - 0.5)
        result.append(value)
    return pd.Series(result, index=returns.index)


def signal_table(history, context, alternate):
    close = np.log(history.qqqClose)
    daily = close.diff()
    # All thresholds and normalizers are estimated before the observation.
    volatility = daily.rolling(63).std().shift()
    shock = -close.diff(5) / (volatility.shift(4) * np.sqrt(5))
    peak_age = recent_high(close)
    gap = close - close.rolling(63).max()
    cusum = downside_accumulator(daily, volatility, peak_age.eq(0))
    cusum_cutoff = cusum.shift().rolling(756, min_periods=252).quantile(0.90)
    semis = np.log(pd.DataFrame({"soxx": context.SOXX, "xsd": alternate.XSD}).reindex(history.index))
    semiconductor = -semis.sub(close, axis=0).diff(5)
    credit = -np.log(history.HYG / history.IEI).diff(5)
    volatility_term = np.log(history.VIX / history.VIX3M).diff(5)
    cross = semiconductor.assign(credit=credit, volatility=volatility_term)
    high_risk = cross.gt(cross.shift().rolling(504, min_periods=126).quantile(0.80))
    tech_weak = high_risk.soxx & high_risk.xsd
    credit_vol = high_risk.credit & high_risk.volatility
    recent = peak_age.le(21)
    shock_trigger = shock.ge(1.5)
    cusum_trigger = cusum.gt(cusum_cutoff) & daily.lt(0)
    return cross.assign(
        peak_age=peak_age, drawdown=np.expm1(gap), shock=shock, cusum=cusum, cusum_cutoff=cusum_cutoff,
        semis_confirm=tech_weak, credit_vol_confirm=credit_vol,
        shock_old_zone=shock_trigger & gap.ge(np.log(0.97)),
        shock_recent_peak=shock_trigger & recent,
        shock_tech=shock_trigger & recent & tech_weak,
        shock_credit_vol=shock_trigger & recent & credit_vol,
        cusum_recent_peak=cusum_trigger & recent,
        cusum_tech=cusum_trigger & recent & tech_weak,
        cusum_credit_vol=cusum_trigger & recent & credit_vol,
        # Exploratory: remove the price-shock requirement once the individual
        # risk channels are weak. The price must have left its recent high.
        tech_early=recent & gap.lt(0) & tech_weak,
        tech_credit_vol_early=recent & gap.lt(0) & tech_weak & credit_vol,
    )


RULES = ("shock_old_zone", "shock_recent_peak", "shock_tech", "shock_credit_vol",
         "cusum_recent_peak", "cusum_tech", "cusum_credit_vol", "tech_early", "tech_credit_vol_early",
         "staged_tech")


def overlay(reference, signals, safe, clock, rule, *, delay=0):
    confirm_rule = "shock_credit_vol" if rule == "staged_tech" else rule
    scout = signals.tech_early if rule == "staged_tech" else pd.Series(False, index=signals.index)
    executable = signals[[confirm_rule, "peak_age"]].assign(scout=scout).shift(1 + delay)
    target = []
    exits = []
    waiting_week = None
    used_anchor = None
    scout_anchor = None
    previous = 0.0
    level = 1.0
    for i, (long, (trigger, age, scout_trigger), permits, week) in enumerate(zip(
        reference, executable.to_numpy(), safe, clock, strict=True
    )):
        anchor = i - 1 - delay - int(age) if pd.notna(age) else None
        actionable = anchor is not None and pd.notna(week)
        if not long:
            waiting_week = None
            level = 1.0
        elif (actionable and trigger == True and anchor != used_anchor and previous > 0
              and (waiting_week is None or rule == "staged_tech")):
            used_anchor = anchor
            waiting_week = week
            level = 0.0
            exits.append(str(reference.index[i].date()))
        elif waiting_week is not None and week != waiting_week and permits:
            waiting_week = None
            level = 1.0
        elif (actionable and scout_trigger == True and previous == 1 and waiting_week is None
              and anchor != scout_anchor and anchor != used_anchor):
            scout_anchor = anchor
            waiting_week = week
            level = 0.5
        previous = level if long else 0.0
        target.append(previous)
    return pd.Series(target, index=reference.index), exits


def frozen_reentry(history, context, alternate):
    data, _ = inputs(history, context, alternate)
    projected = forecasts(data, forward_payoffs(history, data.index, 4))
    weekly_safe = ~elevated_loss(projected.price_loss)
    safe = execution_state(history, weekly_safe).eq(True)
    clock = execution_state(history, pd.Series(data.index, index=data.index))
    base = price_signals(history).trend
    return only_reentry(base, safe.astype(float)), safe, clock


def episodes(prices, reference, candidate, base_frame, frame):
    out = (reference > candidate).to_numpy()
    starts = np.flatnonzero(out & ~np.r_[False, out[:-1]])
    ends = np.flatnonzero(~out & np.r_[False, out[:-1]])
    active = np.log(frame.nav / base_frame.nav)
    trades = []
    for i in starts:
        next_end = ends[ends > i]
        j = next_end[0] if len(next_end) else len(prices) - 1
        net = active.iloc[j] - (active.iloc[i - 1] if i else 0)
        trades.append({"date": str(prices.index[i].date()), "end": str(prices.index[j].date()),
                       "net": float(net), "filled_at": float(prices.tqqqOpen.iloc[i])})
    np.testing.assert_allclose(sum(row["net"] for row in trades), active.iloc[-1], atol=1e-8)
    best = max(trades, key=lambda x: x["net"], default=None)
    return {
        "count": len(trades), "wins": sum(row["net"] > 0 for row in trades),
        "best": best, "cumulative_excess_without_best": float(np.expm1(active.iloc[-1] - max(0, best["net"])))
        if best else 0.0, "trades": trades,
    }


def compare(history, reference, safe, clock, signals, bps, delay):
    prices = history.loc[START:]
    base_frame, _ = replay(prices, reference.loc[START:], bps)
    base_returns = daily_returns(base_frame.nav)
    output = {}
    for name in ("hold", "frozen", *RULES):
        if name == "hold":
            target, dates = pd.Series(1.0, index=history.index), []
        elif name == "frozen":
            target, dates = reference, []
        else:
            target, dates = overlay(reference, signals, safe, clock, name, delay=delay)
        frame, fills = replay(prices, target.loc[START:], bps)
        returns = daily_returns(frame.nav)
        before = frame.loc[:"2025-12-31"]
        regular = returns[~prices.index.year.isin([2020, 2022])]
        result = {
            **path_metrics(frame, fills),
            "sharpe_delta_interval": paired_interval(returns, base_returns),
            "through_2025": path_metrics(before, before.turnover.gt(1e-9).sum()),
            "sharpe_excluding_2020_2022": float(sharpe(regular)),
            "cagr_excluding_2020_2022": float(np.expm1(np.log1p(regular).mean() * 252)),
            "cash_starts_october": [date for date in dates if date >= CASE_PEAK],
        }
        if CASE_PEAK in prices.index:
            recent = target.loc[CASE_PEAK:]
            cash = recent.index[recent.eq(0)]
            first = cash[0] if len(cash) else None
            result.update({
                "case_return": float(frame.nav.iloc[-1] / frame.nav.at[CASE_PEAK] - 1),
                "first_cash_case": str(first.date()) if first is not None else None,
                "first_reduced_case": str(recent.index[recent.lt(1)][0].date())
                if recent.lt(1).any() else None,
                "days_risk_on_after_first_cash": int(recent.loc[first:].gt(0).sum()) if first is not None else None,
            })
        if name in RULES:
            result["extra_cash"] = episodes(prices, reference.loc[START:], target.loc[START:], base_frame, frame)
        output[name] = result
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", default="2026-11-06")
    for name in ("history-cache", "context-cache", "alternate-cache"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()

    def read(path):
        return pd.read_csv(path, index_col=0, parse_dates=True).loc[:args.end]

    history, context, alternate = map(read, (args.history_cache, args.context_cache, args.alternate_cache))
    signals = signal_table(history, context, alternate)
    reference, safe, clock = frozen_reentry(history, context, alternate)
    for cutoff in ("2018-12-21", "2020-03-20", "2022-10-14", "2025-04-04", "2026-10-16"):
        p, c, a = (frame.loc[:cutoff] for frame in (history, context, alternate))
        early = signal_table(p, c, a)
        pd.testing.assert_frame_equal(early, signals.loc[:cutoff])
        if cutoff == "2026-10-16":
            r, s, w = frozen_reentry(p, c, a)
            pd.testing.assert_series_equal(r, reference.loc[:cutoff])
        else:
            r, s, w = reference.loc[:cutoff], safe.loc[:cutoff], clock.loc[:cutoff]
        for rule in RULES:
            expected, _ = overlay(reference, signals, safe, clock, rule)
            actual, _ = overlay(r, early, s, w, rule)
            pd.testing.assert_series_equal(actual, expected.loc[:cutoff])
    inputs_hash = {name: hashlib.sha256(frame.to_csv().encode()).hexdigest() for name, frame in
                   (("market", history), ("context", context), ("alternate", alternate))}
    print(json.dumps({
        "input_sha256": inputs_hash,
        "october_signals": json.loads(signals.loc["2026-10-01":].reset_index().to_json(
            orient="records", date_format="iso"
        )),
        "results": {f"{bps}bps_delay{delay}": compare(history, reference, safe, clock, signals, bps, delay)
                    for bps, delay in ((25, 0), (50, 0), (25, 1))},
    }, indent=2))


if __name__ == "__main__":
    main()
