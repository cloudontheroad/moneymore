"""Walk forward with risk features motivated by distinct historical crash mechanisms."""

from __future__ import annotations

import argparse
import json
import re
from io import StringIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from review_tqqq_kelly_forecast import standardize
from review_tqqq_peak_exit import CASE_PEAK, START, frozen_reentry, overlay, recent_high, signal_table
from review_tqqq_sharpe import daily_returns, sharpe
from review_tqqq_swing import replay
from review_tqqq_swing_round_trips import path_metrics

BASE = ("gap", "return20", "volatility", "shock5")
FAMILIES = ("narrowing", "funding", "repricing", "cascade")
CASES = {
    "1929": "1929-09-03", "1937": "1937-03-10", "1973": "1973-01-11",
    "1987": "1987-08-25", "1998": "1998-07-20", "2000": "2000-03-27",
    "2007": "2007-10-31", "2011": "2011-07-22", "2015": "2015-07-20",
    "2018": "2018-10-01", "2020": "2020-02-19", "2021": "2021-11-19",
    "2022": "2022-08-15", "2024": "2024-07-10", "2025": "2025-02-19",
}
SECTORS = ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY")


def french_daily(path, header, offset):
    with ZipFile(path) as archive:
        lines = archive.read(archive.namelist()[0]).decode("latin-1").splitlines()
    start = next(i for i, row in enumerate(lines) if header in row) + offset
    stop = next(i for i in range(start + 1, len(lines)) if not re.match(r"^\s*\d{8},", lines[i]))
    table = pd.read_csv(StringIO("\n".join(lines[start:stop])), index_col=0, na_values=[-99.99, -999])
    table.index = pd.to_datetime(table.index.astype(str), format="%Y%m%d")
    return table / 100


def rank(series):
    return series.rolling(756, min_periods=252).rank(pct=True)


def conjunction(*parts):
    return pd.concat(parts, axis=1).min(axis=1, skipna=False)


def mechanisms(price, market, sectors, finance, growth_value, rates, *, real_yields):
    close = np.log(price)
    daily = close.diff()
    market = np.log(market.reindex(close.index))
    log_sectors = np.log(sectors.reindex(close.index))
    rates = rates.reindex(close.index).ffill(limit=3).shift(1)
    vol = daily.rolling(63).std()
    shock = -close.diff(5) / (vol.shift(5) * np.sqrt(5))
    relative = close - market
    trend_gap = log_sectors - log_sectors.rolling(63).mean()
    weak = trend_gap.lt(0).where(trend_gap.notna()).mean(axis=1).where(trend_gap.count(axis=1).ge(8))
    sector5 = log_sectors.diff(5)
    synchronization = sector5.lt(0).where(sector5.notna()).mean(axis=1).where(sector5.count(axis=1).ge(8))
    long_yield = rates.DFII10 if real_yields else rates.DGS10
    families = pd.DataFrame({
        "narrowing": conjunction(rank(relative.diff(126)), rank(weak.diff(20))),
        "funding": conjunction(rank(rates.BAA10Y.diff(20)), rank(-(np.log(finance.reindex(close.index)) - market).diff(20))),
        "repricing": conjunction(rank(long_yield.diff(20)), rank(-np.log(growth_value.reindex(close.index)).diff(20))),
        "cascade": conjunction(rank(shock), rank(synchronization)),
    }, index=close.index)
    age = recent_high(close)
    recent = age.le(21)
    pressure = rank(shock)
    return families.assign(
        gap=close - close.rolling(63).max(), return20=close.diff(20), volatility=vol,
        shock5=shock, peak_age=age, recent=recent, participation_weak=weak,
        real_or_nominal_yield_change=long_yield.diff(20), baa_change=rates.BAA10Y.diff(20),
        no_safe_haven=conjunction(families.cascade, rank(rates.DGS10.diff(5))),
        narrowing_exit=recent & families.narrowing.ge(.80) & rank(-relative.diff(5)).ge(.65),
        funding_exit=recent & families.funding.ge(.80) & pressure.ge(.65),
        repricing_exit=recent & families.repricing.ge(.80) & pressure.ge(.65),
        cascade_exit=recent & families.cascade.ge(.85),
    )


def outcomes(close, open_=None, horizon=42):
    # Before TQQQ existed, evaluate next close to future closes of the actual index.
    execution = close if open_ is None else open_
    index = close.index
    path = pd.concat([close, execution], axis=1).min(axis=1).to_numpy()
    entry = execution.to_numpy()
    rows = [
        (index[i + horizon], path[i + 1:i + horizon + 1].min() / entry[i + 1] - 1)
        for i in range(len(index) - horizon)
    ]
    return pd.DataFrame(rows, index=index[:len(rows)], columns=["settled", "loss"]).reindex(index)


def forecast(data, outcome, threshold):
    groups = {"price": BASE, **{family: (*BASE, family) for family in FAMILIES},
              "all": (*BASE, *FAMILIES)}
    valid = data[[*BASE, *FAMILIES]].notna().all(axis=1)
    result = pd.DataFrame(index=data.index, columns=list(groups), dtype=float)
    periods = data.index.to_period("W-FRI")
    # Training uses the first actual session as the week boundary and prior weeks' last close.
    completed_weeks = ~periods.duplicated(keep="last")
    for _, days in data.groupby(periods, sort=False):
        start = days.index[0]
        train = completed_weeks & valid & data.recent & outcome.settled.lt(start)
        y = outcome.loc[train, "loss"].le(-threshold)
        available = days.index[valid.loc[days.index]]
        if len(y) < 80 or min(y.sum(), (~y).sum()) < 8 or not len(available):
            continue
        for name, features in groups.items():
            columns = list(features)
            x, query = standardize(data.loc[train, columns].to_numpy(), data.loc[available, columns].to_numpy())
            model = LogisticRegression(C=.15, max_iter=500).fit(x, y)
            result.loc[available, name] = model.predict_proba(query)[:, 1]
    return result


def classification(data, outcome, estimates, threshold, start, end):
    joined = estimates.join(outcome).loc[start:end]
    weeks = ~joined.index.to_period("W-FRI").duplicated(keep="last")
    joined = joined.loc[weeks & data.recent.reindex(joined.index)].dropna()
    y = joined.loss.le(-threshold)
    aucs = {name: float(roc_auc_score(y, joined[name])) for name in estimates} if y.nunique() == 2 else {}
    return {"weeks": len(y), "tail": int(y.sum()), "auc": aucs}


def case_profiles(price, signals, cases, *, rules=None):
    rules = rules if rules is not None else {family: f"{family}_exit" for family in FAMILIES}
    output = {}
    for name, anchor in cases.items():
        center = pd.Timestamp(anchor)
        width = pd.DateOffset(days=8)
        near = price.loc[center - width:center + width]
        if near.empty:
            continue
        peak = near.idxmax()
        loc = price.index.get_loc(peak)
        after = price.iloc[loc + 1:loc + 64]
        end = after.idxmin()
        window = signals.loc[price.index[loc - 5]:end]
        alerts = {}
        for family, column in rules.items():
            hits = window.index[window[column]]
            if len(hits):
                date = hits[0]
                alerts[family] = {"date": str(date.date()), "days_after_peak": int(price.index.get_loc(date) - loc),
                                  "loss_at_signal": float(price.at[date] / price.at[peak] - 1)}
        output[name] = {"peak": str(peak.date()), "trough": str(end.date()),
                        "drawdown": float(price.at[end] / price.at[peak] - 1), "alerts": alerts}
    return output


def portfolio(history, context, alternate, signals, predictions):
    prior = signal_table(history, context, alternate)
    flags = prior[["peak_age", "shock_credit_vol"]].join(signals[[f"{f}_exit" for f in FAMILIES]])
    flags["any_mechanism"] = flags[[f"{f}_exit" for f in FAMILIES]].any(axis=1)
    for name in ("price", "all"):
        prob = predictions[name]
        cutoff = prob.shift().rolling(504, min_periods=126).quantile(.80)
        flags[f"trained_{name}"] = signals.recent & prob.gt(cutoff)
    flags["prior_plus_trained"] = flags.shock_credit_vol | flags.trained_all
    flags["prior_plus_narrowing"] = flags.shock_credit_vol | flags.narrowing_exit
    reference, safe, clock = frozen_reentry(history, context, alternate)
    targets = {"hold": pd.Series(1.0, index=history.index), "frozen": reference}
    for name in flags.columns.drop("peak_age"):
        targets[name], _ = overlay(reference, flags, safe, clock, name)
    prices = history.loc[START:]
    result = {}
    for cost in (25, 50):
        models = {}
        for name, target in targets.items():
            frame, fills = replay(prices, target.loc[START:], cost)
            case = target.loc[CASE_PEAK:]
            cash = case.index[case.eq(0)]
            pre = frame.loc[:"2025-12-31"]
            regular = daily_returns(frame.nav)[~frame.index.year.isin([2020, 2022])]
            models[name] = {
                **path_metrics(frame, fills),
                "through_2025": path_metrics(pre, pre.turnover.gt(1e-9).sum()),
                "sharpe_without_2020_2022": float(sharpe(regular)),
                "october_return": float(frame.nav.iloc[-1] / frame.nav.at[CASE_PEAK] - 1),
                "october_cash_date": str(cash[0].date()) if len(cash) else None,
            }
        result[str(cost)] = models
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("history", "context", "alternate", "sectors", "industry", "market", "fred", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    def read(path):
        return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()

    history, context, alternate, sectors = map(read, (args.history, args.context, args.alternate, args.sectors))
    rates = pd.concat([read(args.fred / f"{name}.csv").iloc[:, 0].rename(name)
                       for name in ("BAA10Y", "DGS10", "DFII10")], axis=1).sort_index()
    ff = french_daily(args.market, "Mkt-RF", 0)
    market = (1 + ff["Mkt-RF"] + ff.RF).cumprod()
    industry_returns = french_daily(args.industry, "Average Value Weighted Returns -- Daily", 1)
    industries = (1 + industry_returns).cumprod()
    ndx = read(args.fred / "NASDAQ100.csv").iloc[:, 0].dropna().reindex(market.index).dropna()
    composite = read(args.fred / "NASDAQCOM.csv").iloc[:, 0].dropna().reindex(market.index).dropna()
    growth = industry_returns[["Hardw", "Softw", "Chips"]].mean(axis=1)
    value = industry_returns[["Banks", "Oil", "Util"]].mean(axis=1)
    hist_growth = ((1 + growth) / (1 + value)).cumprod()
    long = mechanisms(ndx, market, industries, industries.Banks, hist_growth, rates, real_yields=False)
    early = mechanisms(market, market, industries, industries.Banks, hist_growth, rates, real_yields=False)
    comp_signals = mechanisms(composite, market, industries, industries.Banks, hist_growth, rates, real_yields=False)
    modern = mechanisms(history.qqqClose, context.SPY, sectors[list(SECTORS)], sectors.XLF, sectors.IWF / sectors.IWD,
                        rates, real_yields=True)
    # Check that removing the right tail leaves all features and model values through the cut intact.
    cut = "2022-12-30"
    pd.testing.assert_frame_equal(
        modern.loc[:cut],
        mechanisms(history.qqqClose.loc[:cut], context.SPY, sectors[list(SECTORS)], sectors.XLF,
                   sectors.IWF / sectors.IWD, rates.loc[:cut], real_yields=True),
    )
    modern_outcome = outcomes(history.tqqqClose, history.tqqqOpen)
    trained = forecast(modern, modern_outcome, .25)
    pd.testing.assert_frame_equal(trained.loc[:cut], forecast(
        modern.loc[:cut], outcomes(history.tqqqClose.loc[:cut], history.tqqqOpen.loc[:cut]), .25))
    old_outcome = outcomes(ndx)
    old_trained = forecast(long, old_outcome, .12)
    report = {
        "market_cases_before_ndx": {
            **case_profiles(market, early, {name: CASES[name] for name in ("1929", "1937")}),
            **case_profiles(composite, comp_signals, {"1973": CASES["1973"]}),
        },
        "ndx_cases": case_profiles(ndx, long, dict(list(CASES.items())[3:])),
        "ndx_walk_forward": {
            label: classification(long, old_outcome, old_trained, .12, start, end)
            for label, start, end in (("before_2010", "1986", "2009"), ("2010_2025", "2010", "2025"))
        },
        "tqqq_walk_forward": {
            label: classification(modern, modern_outcome, trained, .25, start, end)
            for label, start, end in (("2014_2019", START, "2019"), ("2020_2025", "2020", "2025"),
                                      ("2026", "2026", "2026"))
        },
        "portfolios": portfolio(history, context, alternate, modern, trained),
        "october_signals": json.loads(modern.join(trained.add_prefix("p_")).loc[CASE_PEAK:]
                                      .reset_index().to_json(orient="records", date_format="iso")),
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("ndx_walk_forward", "tqqq_walk_forward")}, indent=2))


if __name__ == "__main__":
    main()
