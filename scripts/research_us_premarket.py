"""Compare frozen daily rankings with one pre-open gap feature on identical dates."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import exchange_calendars as xcals

from replay_us_open2open import LABEL, ROOT, SPEC, daily_context, fit_ranker, market, metrics, training_panel
from sync_us_yahoo_adjusted_history import yahoo_symbol

ZONE = "America/New_York"
DATA = ROOT / "data/us_overnight_paper_only/daily_yahoo_adjusted"
EVENTS = ROOT / "data/us_overnight_event_warehouse"
CACHE = market.REPORT_DIR / "premarket_research_quotes.parquet"
WEIGHT = 0.20
LOOKBACK = 54
CALENDAR = xcals.get_calendar("XNYS")


def premarket_bars(payload):
    starts = pd.to_datetime(payload["timestamp"], unit="s", utc=True).tz_convert(ZONE)
    bars = pd.DataFrame({"price": np.asarray(payload["indicators"]["quote"][0]["close"], dtype=float)}, index=starts)
    ends = starts + pd.Timedelta(minutes=5)
    minutes = ends.hour * 60 + ends.minute
    valid = ((minutes >= 8 * 60 + 50) & (minutes <= 9 * 60 + 20)
             & (ends.second == 0) & (ends.minute % 5 == 0)
             & (ends <= pd.Timestamp.now(tz="UTC")))
    valid &= np.isfinite(bars.price.to_numpy()) & (bars.price.to_numpy() > 0)
    bars = bars.loc[valid].copy()
    bars["available_at"] = ends[valid]
    bars["session"] = bars.index.tz_localize(None).normalize()
    return bars.sort_index().drop_duplicates("session", keep="last")


def fetch(symbol):
    ticker = urllib.parse.quote(yahoo_symbol(symbol), safe="")
    url = (f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
           "?range=60d&interval=5m&includePrePost=true&events=splits")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 MoneyMore/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            payload = json.load(response)["chart"]["result"][0]
        if payload["meta"]["dataGranularity"] != "5m":
            raise ValueError("unexpected interval")
        quotes = premarket_bars(payload)
        splits = {pd.Timestamp(item["date"], unit="s", tz="UTC").tz_convert(ZONE).date()
                  for item in payload.get("events", {}).get("splits", {}).values()}
        quotes = quotes.loc[~quotes.session.dt.date.isin(splits)]
        bars = pd.read_parquet(DATA / f"{symbol.replace('.', '_')}.parquet",
                               columns=["timestamp", "close"]).sort_values("timestamp")
        bars["session"] = pd.to_datetime(bars.timestamp, utc=True).dt.tz_convert(ZONE).dt.tz_localize(None).dt.normalize()
        quotes = quotes[quotes.session.isin(CALENDAR.sessions)].copy()
        quotes["previous_close"] = quotes.session.map(CALENDAR.previous_session).map(bars.set_index("session").close)
        quotes["gap"] = (quotes.price / quotes.previous_close - 1).replace([np.inf, -np.inf], np.nan)
        quotes["symbol"] = symbol
        return quotes, None
    except (OSError, KeyError, ValueError, TypeError, IndexError) as exc:
        return None, f"{symbol}: {exc}"


def event_audit(end):
    result = {}
    for name in ("news", "earnings", "economic", "macro", "ratings"):
        path = EVENTS / name / "canonical.parquet"
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        observed = pd.to_datetime(frame.get("first_observed_at_utc"), utc=True, errors="coerce")
        result[name] = {"rows": len(frame),
                        "observed_by_last_test_cutoff": int((observed < end).sum())}
    return result


def run():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    panel = training_panel()
    daily, _ = daily_context(panel)
    days = []
    last_anchor = None
    for i in range(len(daily) - LOOKBACK, len(daily)):
        date = daily.index[i]
        anchor = daily.index[i - i % spec["stock_retrain"]]
        if anchor != last_anchor:
            model = fit_ranker(panel, daily, anchor, spec)
            last_anchor = anchor
        day = panel.loc[panel.timestamp == date, ["symbol", LABEL, *spec["stock_features"]]].copy()
        day["score"] = model.predict(day[spec["stock_features"]])
        day = day.nlargest(50, "score")
        day["session"] = pd.Timestamp(date.tz_convert(ZONE).date())
        days.append(day[["symbol", "session", "score", LABEL]])
    universe = sorted(set(pd.concat(days).symbol))
    print(f"Fetching {len(universe)} symbols from rolling Top50", flush=True)
    with ThreadPoolExecutor(max_workers=12) as pool:
        fetched = list(pool.map(fetch, universe))
    frames = [frame for frame, error in fetched if frame is not None]
    errors = [error for _, error in fetched if error]
    if not frames:
        raise RuntimeError(f"No extended-session quotes: {errors[:3]}")
    quotes = pd.concat(frames, ignore_index=True)
    quotes.to_parquet(CACHE, index=False)
    rows = []
    for day in days:
        session = day.session.iloc[0]
        aligned = day.merge(quotes[["session", "symbol", "gap"]], on=["session", "symbol"],
                            how="left", validate="one_to_one")
        available = aligned.gap.notna().sum()
        baseline = aligned.nlargest(10, "score")
        daily_rank = aligned.score.rank(pct=True)
        row = {"session": str(session.date()), "quotes": int(available),
               "baseline_top1": float(baseline[LABEL].iloc[0]),
               "baseline_top10": float(baseline[LABEL].mean())}
        for name, direction in (("continuation", 1), ("reversal", -1)):
            gap_rank = (direction * aligned.gap).rank(pct=True).fillna(0.5)
            aligned["blended"] = (1 - WEIGHT) * daily_rank + WEIGHT * gap_rank
            selected = aligned.nlargest(10, "blended") if available >= 35 else baseline
            row[f"{name}_top1"] = float(selected[LABEL].iloc[0])
            row[f"{name}_top10"] = float(selected[LABEL].mean())
            row[f"{name}_top10_replaced"] = len(set(selected.symbol) - set(baseline.symbol))
        rows.append(row)
    frame = pd.DataFrame(rows)
    covered = frame.quotes >= 35
    report = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "design": "Top50 daily rank; 20% rank weight on latest completed 5m gap through 09:20 NY; at least 35 valid quotes or fall back",
        "label": "same open-to-next-open net 20bp",
        "membership": spec["historical_universe"],
        "symbols_requested": len(universe), "download_errors": errors,
        "sessions": len(frame), "sessions_with_overlay": int(covered.sum()),
        "results": {k: metrics(frame[k]) for k in frame if k.endswith(("_top1", "_top10"))},
        "paired_diagnostics": {
            k: {"first_half_delta_bps": float(delta.iloc[:len(delta)//2].mean() * 1e4),
                "last_half_delta_bps": float(delta.iloc[len(delta)//2:].mean() * 1e4),
                "delta_without_best3_bps": float(delta.sort_values().iloc[:-3].mean() * 1e4)}
            for k in frame if k.endswith(("_top1", "_top10")) and not k.startswith("baseline")
            for delta in [frame[k] - frame[f"baseline_{k.split('_')[-1]}"]]
        },
        "observed_event_coverage": event_audit(
            pd.Timestamp(frame.session.iloc[-1], tz=ZONE) + pd.Timedelta(hours=9, minutes=20)
        ),
        "daily": rows,
    }
    path = market.REPORT_DIR / "premarket_research_v1.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("daily", "download_errors")}, indent=2))


if __name__ == "__main__":
    run()
