"""Capture new information and optionally book a separate pre-open comparison."""
from concurrent.futures import ThreadPoolExecutor
import json
import socket
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from research_us_premarket import premarket_bars
from run_us_open2open import write_record
from sync_us_yahoo_adjusted_history import yahoo_symbol


def get_yahoo(url):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 MoneyMore/1.0"})
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)


def capture_stock(candidate, session, prior_close_at, cutoff):
    ticker = yahoo_symbol(candidate["symbol"])
    encoded = urllib.parse.quote(ticker, safe="")
    row = {"symbol": candidate["symbol"], "gap": None, "news": []}
    try:
        chart = get_yahoo(f"https://query2.finance.yahoo.com/v8/finance/chart/{encoded}"
                         "?range=2d&interval=5m&includePrePost=true&events=splits")["chart"]["result"][0]
        if chart["meta"]["dataGranularity"] != "5m":
            raise ValueError("Unexpected chart interval")
        starts = pd.to_datetime(chart["timestamp"], unit="s", utc=True)
        prices = np.asarray(chart["indicators"]["quote"][0]["close"], dtype=float)
        ends = starts + pd.Timedelta(minutes=5)
        prior_date = prior_close_at.tz_convert("America/New_York").date()
        after_hours_end = pd.Timestamp(prior_date, tz="America/New_York") + pd.Timedelta(hours=20)
        valid_post = np.flatnonzero(
            (starts >= prior_close_at) & (ends <= after_hours_end)
            & (starts.second == 0) & (starts.minute % 5 == 0)
            & np.isfinite(prices) & (prices > 0)
        )
        if len(valid_post):
            last = valid_post[-1]
            row.update(after_hours_price=float(prices[last]), after_hours_bar_end=ends[last].isoformat())
        quote = premarket_bars(chart)
        quote = quote[(quote.session == session) & (quote.available_at <= cutoff)]
        split_dates = {pd.Timestamp(event["date"], unit="s", tz="UTC").tz_convert("America/New_York").date()
                       for event in chart.get("events", {}).get("splits", {}).values()}
        reference = candidate.get("prior_close")
        if not quote.empty and reference and reference > 0 and session.date() not in split_dates:
            latest = quote.iloc[-1]
            row.update(gap=float(latest.price / reference - 1), price=float(latest.price),
                       bar_end=latest.available_at.isoformat(),
                       quote_observed_at=pd.Timestamp.now(tz="UTC").isoformat())
    except (OSError, KeyError, TypeError, ValueError, IndexError):
        row["quote_error"] = True
    try:
        items = get_yahoo(f"https://query2.finance.yahoo.com/v1/finance/search?q={encoded}"
                          "&quotesCount=0&newsCount=8").get("news", [])
        observed = pd.Timestamp.now(tz="UTC")
        for item in items:
            published = pd.to_datetime(item.get("providerPublishTime"), unit="s", utc=True, errors="coerce")
            if prior_close_at < published <= cutoff and ticker in item.get("relatedTickers", []):
                row["news"].append({
                    "title": item.get("title"), "url": item.get("link"),
                    "published_at": published.isoformat(), "observed_at": observed.isoformat(),
                })
    except (OSError, KeyError, TypeError, ValueError, IndexError):
        row["news_error"] = True
    return row


def capture_events(config, session, prior_close_at, cutoff, symbols):
    try:
        with socket.create_connection((config.opend_host, config.opend_port), timeout=1):
            pass
    except OSError:
        return {"status": "OPEND_UNAVAILABLE", "earnings": [], "economic": []}
    from futu import Market, OpenQuoteContext

    context = OpenQuoteContext(host=config.opend_host, port=config.opend_port)
    result = {"status": "OK"}
    try:
        start = str(prior_close_at.tz_convert(config.timezone).date())
        end = str(session.date())
        sources = {
            "earnings": (lambda: context.get_earnings_calendar(
                Market.US, begin_date=start, end_date=end), "earnings_timestamp",
                ["security", "eps_actual", "eps_predict", "revenue_actual", "revenue_predict"]),
            "economic": (lambda: context.get_economic_calendar(
                start, end, market_list=[Market.US], count=100), "timestamp",
                ["title", "actual", "consensus", "previous"]),
        }
        for name, (get, field, columns) in sources.items():
            try:
                code, frame, *_ = get()
                if code != 0:
                    raise ValueError("Provider returned an error")
                published = pd.to_datetime(frame[field], unit="s", utc=True, errors="coerce")
                subset = frame.loc[(published > prior_close_at) & (published <= cutoff)].copy()
                if name == "earnings":
                    subset = subset[subset.security.isin(symbols)]
                subset["published_at"] = published
                subset["observed_at"] = pd.Timestamp.now(tz="UTC")
                result[name] = json.loads(subset.reindex(columns=columns + ["published_at", "observed_at"])
                                          .to_json(orient="records", date_format="iso"))
            except (KeyError, TypeError, ValueError):
                result[name] = []
                result["status"] = "PARTIAL"
    finally:
        context.close()
    return result


def rerank(pool, quotes, count):
    scores = pd.Series({row["symbol"]: row["score"] for row in pool}, dtype=float)
    gaps = pd.Series({row["symbol"]: row["gap"] for row in quotes}, dtype=float).reindex(scores.index)
    if gaps.notna().sum() < 35:
        return []
    blended = 0.8 * scores.rank(pct=True) + 0.2 * gaps.rank(pct=True).fillna(0.5)
    by_symbol = {row["symbol"]: row for row in pool}
    return [by_symbol[symbol] | {"rank": rank, "review_score": float(blended[symbol])}
            for rank, symbol in enumerate(blended.nlargest(count).index, 1)]


def review(output, config, calendar, *, clock=pd.Timestamp.now):
    now = clock(tz="UTC")
    session = pd.Timestamp(now.tz_convert(config.timezone).date())
    if not calendar.is_session(session):
        return {"review_status": "NOT_A_SESSION"}
    opening = calendar.session_open(session)
    information_cutoff = opening - pd.Timedelta(minutes=10)
    decision_cutoff = opening - pd.Timedelta(minutes=5)
    if not information_cutoff <= now < decision_cutoff:
        return {"review_status": "OUTSIDE_REVIEW_WINDOW"}
    root = output / "open2open_v1"
    filename = f"{session.date()}.json"
    source = root / "plans" / filename
    target = root / "premarket" / "plans" / filename
    if target.exists():
        return json.loads(target.read_text(encoding="utf-8"))
    if not source.exists():
        return {"review_status": "BASELINE_PENDING"}
    baseline = json.loads(source.read_text(encoding="utf-8"))
    pool = baseline.get("review_pool", [])
    if (baseline.get("status") != "READY" or len(pool) < 35
            or len(baseline.get("candidates", [])) != config.top_n):
        return {"review_status": "BASELINE_PENDING"}
    lower = pd.Timestamp(baseline["prior_close_at"])
    with ThreadPoolExecutor(max_workers=13) as executor:
        events = executor.submit(capture_events, config, session, lower, information_cutoff,
                                 {row["symbol"] for row in pool})
        quotes = list(executor.map(
            lambda candidate: capture_stock(candidate, session, lower, information_cutoff), pool
        ))
        try:
            event_records = events.result()
        except (OSError, RuntimeError, ValueError, ImportError):
            event_records = {"status": "COLLECTION_FAILED", "earnings": [], "economic": []}
    finished = clock(tz="UTC")
    selected = rerank(pool, quotes, len(baseline["candidates"])) if finished < decision_cutoff else []
    original = {item["symbol"] for item in baseline["candidates"]}
    updated = {item["symbol"] for item in selected}
    status = "CHANGED" if selected and original != updated else (
        "UNCHANGED" if selected else "LATE" if finished >= decision_cutoff else "INSUFFICIENT_QUOTES"
    )
    result = {key: value for key, value in baseline.items() if key not in ("review_pool", "candidates")}
    result.update(
        status="READY" if status == "CHANGED" else "REVIEW_ONLY",
        review_status=status, generated_at=finished.isoformat(),
        baseline_generated_at=baseline["generated_at"], information_cutoff=information_cutoff.isoformat(),
        added=sorted(updated - original) if selected else [],
        removed=sorted(original - updated) if selected else [],
        candidates=selected, observations=quotes, events=event_records,
        quote_count=sum(np.isfinite(row["gap"]) for row in quotes if row["gap"] is not None),
    )
    result["quote_count"] = int(result["quote_count"])
    write_record(target, result, once=True)
    return json.loads(target.read_text(encoding="utf-8"))
