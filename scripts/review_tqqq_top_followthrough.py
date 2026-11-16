"""Test Nasdaq concentration and weekly deterioration after a major high."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from review_tqqq_crisis_mechanisms import CASES, SECTORS, case_profiles, mechanisms
from review_tqqq_peak_exit import CASE_PEAK, START, episodes, frozen_reentry, overlay, signal_table
from review_tqqq_reentry_payoff import only_reentry
from review_tqqq_sharpe import daily_returns, paired_interval, sharpe
from review_tqqq_swing import replay
from review_tqqq_swing_round_trips import path_metrics
from review_tqqq_weekly_cycle import execution_state

RULES = ("equal_weight", "price_only", "credit_only", "breadth_only", "slow_confirmed")


def weekly_features(close, equal, sectors, credit):
    data = pd.concat({"index": close, "equal": equal, "credit": credit}, axis=1).reindex(close.index)
    # Use the last actual session. Keep its missing values rather than backfilling from earlier closes.
    ends = ~data.index.to_period("W-FRI").duplicated(keep="last")
    week = data.loc[ends]
    price = np.log(week["index"])
    ratio = np.log(week["equal"]) - price
    leadership = -ratio.diff(13)
    leadership_cut = leadership.shift().rolling(156, min_periods=78).quantile(.80)
    equal_trend = np.log(week["equal"]) - np.log(week["equal"]).rolling(10).mean()
    concentration = (
        leadership.gt(leadership_cut) & equal_trend.lt(0) & equal_trend.shift().lt(0)
        & price.ge(price.rolling(13).max() + np.log(.97))
    )

    # Preserve the anchor date even when the index makes lower local highs.
    age = price.rolling(26).apply(lambda w: np.argmax(w[::-1]), raw=True)
    weak = price.lt(price.rolling(10).mean())
    slow = age.between(2, 13) & weak & weak.shift(fill_value=False)
    spread_change = week.credit.diff(4)
    spread_cut = spread_change.shift().rolling(156, min_periods=78).quantile(.75).clip(lower=0)
    credit_weak = spread_change.gt(spread_cut) & week.credit.diff(2).gt(0)
    industry = sectors.reindex(week.index)
    gap = industry - industry.rolling(10).mean()
    participation_weak = gap.lt(0).where(gap.notna()).mean(axis=1).where(gap.count(axis=1).ge(8))
    breadth_weak = participation_weak.gt(.5) & participation_weak.shift().gt(.5)

    positions = close.index.get_indexer(week.index)
    anchor_week = pd.Series(np.arange(len(week)), index=week.index) - age
    anchor = pd.Series(positions, index=week.index) - anchor_week.map(dict(enumerate(positions)))
    return pd.DataFrame({
        "peak_age": anchor, "age_weeks": age, "leadership": leadership, "leadership_cut": leadership_cut,
        "equal_trend": equal_trend, "participation_weak": participation_weak, "credit_change": spread_change,
        "credit_cut": spread_cut, "equal_weight": concentration, "price_only": slow,
        "credit_only": slow & credit_weak, "breadth_only": slow & breadth_weak,
        "slow_confirmed": slow & credit_weak & breadth_weak,
    }, index=week.index)


def target_table(history, context, alternate, sectors, rates, week, check_equal):
    recent = mechanisms(history.qqqClose, context.SPY, sectors[list(SECTORS)], sectors.XLF,
                        sectors.IWF / sectors.IWD, rates, real_yields=True)
    signals = signal_table(history, context, alternate)
    signals["best"] = signals.shock_credit_vol | recent.narrowing_exit
    base, safe, clock = frozen_reentry(history, context, alternate)
    prior, _ = overlay(base, signals, safe, clock, "best")
    targets = {"hold": pd.Series(1., index=history.index), "base": base, "prior": prior}
    daily = week.reindex(history.index)
    for name in RULES:
        target, _ = overlay(base, daily, safe, clock, name)
        targets[name] = target
        targets[f"prior_{name}"] = np.minimum(prior, target)
    targets["prior_equal_delayed"] = np.minimum(prior, targets["equal_weight"].shift(1, fill_value=1.))
    allow_reentry = ~execution_state(history, week.slow_confirmed).eq(True)
    targets["prior_slow_reentry"] = only_reentry(prior, allow_reentry.astype(float))
    cross_check, _ = overlay(base, check_equal.reindex(history.index), safe, clock, "equal_weight")
    targets["prior_other_etf"] = np.minimum(prior, cross_check)
    return pd.DataFrame(targets)


def audit_cases(close, week):
    names = ("1998", "2000", "2007", "2011", "2015", "2018", "2020", "2021", "2022", "2024", "2025")
    daily = week[list(RULES)].reindex(close.index).eq(True)
    # Reuse the shared peak audit with the named rule columns rather than the old mechanism names.
    return case_profiles(close, daily, {name: CASES[name] for name in names}, rules=dict(zip(RULES, RULES)))


def evaluate(prices, targets, *, october=False):
    result = {}
    for cost in (25, 50):
        reference, _ = replay(prices, targets.prior, cost)
        models = {}
        for name, target in targets.items():
            frame, fills = replay(prices, target, cost)
            pre = frame.loc[:"2025-12-31"]
            before_q4 = frame.nav.at["2018-10-01"]
            q4 = frame.nav.loc["2018-10-01":"2018-12-24"]
            models[name] = {
                **path_metrics(frame, fills),
                "through_2025": path_metrics(pre, pre.turnover.gt(1e-9).sum()),
                "sharpe_delta_interval": paired_interval(daily_returns(frame.nav), daily_returns(reference.nav)),
                "sharpe_without_2015": float(sharpe(daily_returns(frame.nav)[frame.index.year != 2015])),
                "sharpe_without_2018": float(sharpe(daily_returns(frame.nav)[frame.index.year != 2018])),
                "q4_2018_return": float(q4.iloc[-1] / before_q4 - 1),
            }
            if october:
                models[name]["october_return"] = float(frame.nav.iloc[-1] / frame.nav.at[CASE_PEAK] - 1)
            if name.startswith("prior_"):
                trades = episodes(prices, targets.prior, target, reference, frame)
                models[name]["incremental_episodes"] = trades
        result[str(cost)] = models
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("history", "context", "alternate", "sectors", "equal", "equal-check", "fred", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()

    def read(path):
        return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()

    history, context, alternate = map(read, (args.history, args.context, args.alternate))
    sectors = read(args.sectors)
    equal = read(args.equal).iloc[:, 0]
    rates = pd.concat([read(args.fred / f"{name}.csv").iloc[:, 0].rename(name)
                       for name in ("BAA10Y", "DGS10", "DFII10")], axis=1).sort_index()
    credit = -np.log(history.HYG / history.IEI)
    modern = weekly_features(history.qqqClose, equal, sectors[list(SECTORS)], credit)
    for date in ("2018-12-21", "2021-11-19", "2025-04-04"):
        pd.testing.assert_frame_equal(modern.loc[:date], weekly_features(
            history.qqqClose.loc[:date], equal, sectors[list(SECTORS)], credit))
    # QQEW changed its tracked index on December 22, 2025.
    same_mandate = "2025-12-19"
    equal_check = read(args.equal_check).iloc[:, 0].loc[equal.index.min():same_mandate]
    cross_check = weekly_features(history.qqqClose, equal_check, sectors[list(SECTORS)], credit)

    ndx = read(args.fred / "NASDAQ100.csv").iloc[:, 0].dropna().loc[:history.index[-1]]
    composite = read(args.fred / "NASDAQCOM.csv").iloc[:, 0].dropna()
    # The composite includes NDX stocks. The ratio describes cross-index leadership.
    # Individual-stock advance/decline data would be required to measure breadth directly.
    historical = weekly_features(ndx, composite, pd.DataFrame(index=ndx.index),
                                 rates.BAA10Y.reindex(ndx.index).ffill(limit=3).shift())
    prices = history.loc[START:]
    targets = target_table(history, context, alternate, sectors, rates, modern, cross_check).loc[START:]
    stress_prices = prices.copy()
    flash_crash = "2015-08-24"
    stress_prices.loc[flash_crash, "tqqqOpen"] = prices.at[flash_crash, "tqqqClose"]
    report = {
        "historical_cases": audit_cases(ndx, historical),
        "modern_cases": audit_cases(history.qqqClose, modern),
        "dotcom_weekly": json.loads(historical.loc["1999-12":"2000-05"].to_json(orient="index", date_format="iso")),
        "modern_weekly": json.loads(modern.to_json(orient="index", date_format="iso")),
        "portfolios": evaluate(prices, targets.drop(columns="prior_other_etf"), october=True),
        "same_mandate": evaluate(prices.loc[:same_mandate],
                                 targets.loc[:same_mandate, ["prior", "prior_equal_weight", "prior_other_etf"]]),
        "flash_crash_close_stress": evaluate(stress_prices, targets[["prior", "prior_equal_weight"]]),
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
