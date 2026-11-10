"""Screen fixed TQQQ overnight and daytime exposure policies on their own ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from review_tqqq_execution import bootstrap
from review_tqqq_external import daily_prices, summary
from moneymore.leveraged_etf import YahooDailyClient

WINDOWS = (("2013–2019", "2013", "2019"), ("2020–2022", "2020", "2022"), ("2023+", "2023", None))
COSTS = (0, 6, 10, 25)


def context(data):
    price = data.QQQ
    slow = price.rolling(200).mean()
    return pd.DataFrame({
        "bear": price.lt(slow),
        "stress": data.VIX.gt(data.VIX3M),
        "rebound": price.gt(price.shift().rolling(5).max()) & price.lt(slow),
    })


def tape(data):
    flags = context(data)
    # The close auction for an overnight hold must use the earlier completed close.
    # An opening order for the intraday hold can use yesterday's completed close.
    overnight = flags.shift(2, fill_value=False)
    daytime = flags.shift(1, fill_value=False)
    returns = {
        "overnight": np.log(data.TQQQ_open / data.TQQQ.shift()),
        "daytime": np.log(data.TQQQ / data.TQQQ_open),
    }
    halves = []
    for period, known in (("overnight", overnight), ("daytime", daytime)):
        halves.append(known.assign(period=period, step=period == "daytime", log_return=returns[period]))
    return pd.concat(halves).sort_values(["trade_date", "step"]).drop(columns="step")


def exposure_paths(frame):
    overnight = frame.period.eq("overnight")
    daytime = ~overnight
    yield "hold", pd.Series(True, index=frame.index)
    yield "overnight_only", overnight
    yield "daytime_only", daytime
    for name, condition in (
        ("bear", frame.bear), ("stress", frame.stress), ("confirmed", frame.bear & frame.stress)
    ):
        yield f"skip_overnight_{name}", ~(overnight & condition)
        yield f"skip_daytime_{name}", ~(daytime & condition)
    # The diagnostic suggested this direction. Keep it separately labeled as exploratory.
    yield "skip_overnight_bear_rebound_exploratory", ~(overnight & frame.rebound)


def replay(frame, keep, bps):
    rate = bps / 10_000
    prior = keep.shift(fill_value=False)
    buy, sell = keep & ~prior, prior & ~keep
    fees = sell * np.log1p(-rate) - buy * np.log1p(rate)
    missed = -frame.log_return.where(~keep, 0)
    candidate = frame.log_return + missed + fees
    daily = candidate.groupby(level=0).sum()
    net = missed + fees
    net.iloc[0] += np.log1p(rate)  # benchmark buys once at the first open
    np.testing.assert_allclose(net.groupby(level=0).sum(), daily - benchmark(frame, bps), atol=1e-12)
    return net, daily, buy | sell, missed


def benchmark(frame, bps):
    result = frame.log_return.groupby(level=0).sum()
    result.iloc[0] -= np.log1p(bps / 10_000)
    return result


def summarize_candidate(frame, keep, bps, *, include_ledger=False):
    net, daily, fills, missed = replay(frame, keep, bps)
    daily_net = net.groupby(level=0).sum()
    interval = bootstrap([daily_net.to_numpy()])
    windows = {}
    for label, first, last in WINDOWS:
        values = daily_net.loc[first:last]
        if not values.empty:
            windows[label] = float(np.expm1(values.sum()))
    years = daily_net.groupby(daily_net.index.year).sum()
    # Group contiguous omitted intervals and attach the re-entry cost to the prior interval.
    excluded = ~keep
    group = excluded.ne(excluded.shift(fill_value=False)).cumsum().astype("Int64")
    closed = keep & excluded.shift(fill_value=False)
    episode_id = group.where(excluded, group.shift().where(closed))
    episode_net = net.groupby(episode_id).sum()
    np.testing.assert_allclose(episode_net.sum(), net.sum(), atol=1e-12)
    biggest = max(0.0, float(episode_net.max())) if len(episode_net) else 0.0
    result = {
        **summary([1, *np.exp(daily.cumsum())], fills.sum()),
        "relative_return": float(np.expm1(net.sum())),
        "without_best_episode": float(np.expm1(net.sum() - biggest)),
        "relative_by_window": windows,
        "years_won": int((years > 0).sum()),
        "years": len(years),
        "annual_log_excess_interval": interval,
        "omissions": len(episode_net),
        "positive_omissions": int((episode_net > 0).sum()),
        "saved_down_log": float(missed.clip(lower=0).sum()),
        "missed_up_log": float(missed.clip(upper=0).sum()),
        "extra_cost_log": float(net.sum() - missed.sum()),
    }
    if include_ledger:
        ledger = pd.DataFrame({
            "episode": episode_id, "omitted": excluded, "saved": missed.clip(lower=0),
            "missed": missed.clip(upper=0), "net": net,
        })
        result["ledger"] = []
        for _, rows in ledger.groupby("episode"):
            omitted = rows.loc[rows.omitted]
            saved, foregone, active = rows[["saved", "missed", "net"]].sum()
            result["ledger"].append({
                "first_skipped_session": str(omitted.index[0].date()),
                "last_skipped_session": str(omitted.index[-1].date()),
                "reentry_trade_date": str(rows.index[-1].date()) if not rows.omitted.iloc[-1] else None,
                "saved_down_log": float(saved), "missed_up_log": float(foregone),
                "extra_cost_log": float(active - saved - foregone), "net_log": float(active),
            })
    return result


def diagnostic(frame):
    out = {}
    for label, first, last in (("all", "2013", None), *WINDOWS):
        window = frame.loc[first:last]
        out[label] = {}
        for name, rows in (
            ("all", window), ("bull", window.loc[~window.bear]),
            ("bear", window.loc[window.bear]), ("confirmed_stress", window.loc[window.bear & window.stress]),
            ("bear_rebound", window.loc[window.rebound]),
        ):
            out[label][name] = {
                period: {"count": len(group), "annual_log": float(group.log_return.sum() * 252 / window.index.nunique()),
                         "mean_log_bps": float(group.log_return.mean() * 10_000)}
                for period, group in rows.groupby("period") if len(group)
            }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", default="2026-11-06")
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=4) as pool:
        tqqq = pool.submit(daily_prices, "TQQQ", args.end)
        qqq = pool.submit(daily_prices, "QQQ", args.end)
        vix = pool.submit(YahooDailyClient().history, "^VIX", "2010-02-11", args.end)
        vix3m = pool.submit(YahooDailyClient().history, "^VIX3M", "2010-02-11", args.end)
        prices = tqqq.result().rename(columns={"Close": "TQQQ", "Open": "TQQQ_open"})[["TQQQ", "TQQQ_open"]]
        prices["QQQ"] = qqq.result().Close
        for name, future in (("VIX", vix), ("VIX3M", vix3m)):
            source = future.result().set_index("trade_date").close
            source.index = pd.to_datetime(source.index)
            prices[name] = source
    prices.index.name = "trade_date"
    if prices.loc["2013":].isna().any().any():
        raise ValueError(f"Missing aligned prices: {prices.loc['2013':].isna().sum().to_dict()}")
    segments = tape(prices)
    for cutoff in ("2016-06-24", "2020-03-16", "2022-01-24", "2025-04-04"):
        pd.testing.assert_frame_equal(tape(prices.loc[:cutoff]), segments.loc[:cutoff])
    study = segments.loc["2013":].iloc[1:]  # all accounts start at the first open
    expected = np.log(prices.TQQQ / prices.TQQQ.shift()).loc["2013":].copy()
    expected.iloc[0] = study.log_return.iloc[0]
    np.testing.assert_allclose(study.log_return.groupby(level=0).sum(), expected, atol=1e-12)
    tests = {
        bps: {
            name: summarize_candidate(study, keep, bps, include_ledger=bps == 25 and name.endswith("_exploratory"))
            for name, keep in exposure_paths(study)
        }
        for bps in COSTS
    }
    # A screen, not a proof of alpha. Repeatedly studied history never counts as a fresh holdout.
    passed = [
        name for name, result in tests[25].items()
        if name != "hold" and result["annual_log_excess_interval"][0] > 0
        and result["without_best_episode"] > 0 and min(result["relative_by_window"].values()) > 0
    ]
    print(json.dumps({
        "start": str(study.index[0].date()), "end": str(study.index[-1].date()),
        "input_sha256": hashlib.sha256(prices.to_csv().encode()).hexdigest(),
        "gross_return_diagnostics": diagnostic(study), "candidates": tests, "passed": passed,
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
