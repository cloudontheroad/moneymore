"""Generate pre-open paper plans and settle them with dated daily opening bars."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import warnings

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from build_us_overnight_factor_panel import MARKET, SECTORS
from replay_us_open2open import ROOT, SPEC, daily_context, fit_market, fit_ranker, market, settled_days, training_panel
from train_us_overnight_ranker import _cross_sectionalize, _winsorize_by_date
from sync_us_yahoo_adjusted_history import download
from moneymore.us_overnight.config import USOvernightConfig
from moneymore.us_overnight.features import FEATURE_COLUMNS, daily_stock_features

BOOK = ROOT / "configs/us_open2open_paper_book.json"

def book_config():
    return json.loads(BOOK.read_text(encoding="utf-8"))


def write_record(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(record, handle, ensure_ascii=False, allow_nan=False, indent=2)
    os.replace(handle.name, path)


def bars_at(path, through):
    if not path.is_file():
        return pd.DataFrame()
    bars = pd.read_parquet(path).sort_values("timestamp")
    bars["session"] = pd.to_datetime(bars.timestamp, utc=True).dt.tz_convert(
        "America/New_York"
    ).dt.tz_localize(None).dt.normalize()
    return bars.loc[bars.session <= through]


def prior_universe(output, cutoff, *, max_age_days=7):
    snapshots = []
    for path in (output / "universe").glob("top1000-*.parquet"):
        captured = pd.to_datetime(path.stem.removeprefix("top1000-"),
                                  format="%Y-%m-%dT%H%M%SZ", utc=True)
        if captured < cutoff and (
            max_age_days is None or cutoff - captured <= pd.Timedelta(days=max_age_days)
        ):
            snapshots.append((captured, path))
    if not snapshots:
        raise ValueError("缺少盘前可用的近期市值股票池")
    return max(snapshots)[1]


def stock_inputs(output, universe, previous, config):
    rows, eligible = [], set()
    for item in universe.itertuples(index=False):
        bars = bars_at(output / "daily_yahoo_adjusted" / f"{item.code.replace('.', '_')}.parquet", previous)
        if bars.empty or bars.session.iloc[-1] != previous:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            features = daily_stock_features(bars)
        row = features.iloc[-1][FEATURE_COLUMNS].to_dict()
        rows.append(row | {"symbol": item.code, "name": item.name, "timestamp": previous})
        if (len(bars) >= config.minimum_listing_days
                and bars.close.iloc[-1] >= config.minimum_price and bars.volume.iloc[-1] > 0
                and (bars.close * bars.volume).tail(20).median() >= config.minimum_median_dollar_volume_20d):
            eligible.add(item.code)
    if len(rows) < 0.95 * len(universe):
        raise ValueError(f"前一交易日个股日线覆盖不足 {len(rows)}/{len(universe)}")
    transformed = _cross_sectionalize(_winsorize_by_date(pd.DataFrame(rows)))
    return transformed[transformed.symbol.isin(eligible)].sort_values("symbol")


def market_inputs(previous, session, fields, cal):
    context = ROOT / "data/us_overnight_corrected/features/context"
    index = cal.sessions_window(previous, -21)
    prices = {}
    for symbol in [*MARKET, *SECTORS]:
        bars = bars_at(context / f"{symbol.replace('^', 'IDX_')}.parquet", previous)
        if bars.empty or bars.session.iloc[-1] != previous:
            raise ValueError(f"缺少前一交易日市场日线 {symbol}")
        prices[symbol] = bars.set_index("session").close.reindex(index)
    prices = pd.DataFrame(prices)
    if prices.iloc[-6:].isna().any().any():
        raise ValueError("市场日线上下文不完整")
    returns = prices.pct_change(fill_method=None)
    ret5 = prices.pct_change(5, fill_method=None)
    values = {}
    for ticker in MARKET:
        key = ticker.removeprefix("^")
        values[f"market_ret1_{key}"] = returns[ticker].iloc[-1]
        values[f"market_ret5_{key}"] = ret5[ticker].iloc[-1]
        values[f"market_vol20_{key}"] = returns[ticker].tail(20).std()
    values.update(
        breadth_sector_positive_1d=(returns[SECTORS].iloc[-1] > 0).mean(),
        breadth_sector_positive_5d=(ret5[SECTORS].iloc[-1] > 0).mean(),
        calendar_dow_sin=np.sin(2 * np.pi * session.dayofweek / 5),
        calendar_dow_cos=np.cos(2 * np.pi * session.dayofweek / 5),
        calendar_month_end=float(session.is_month_end),
        calendar_month_start=float(session.is_month_start),
        calendar_quarter_end=float(session.is_quarter_end),
        calendar_opex_week=float(15 <= session.day <= 21),
    )
    return pd.DataFrame([values])[fields]


def score(output, config, cal):
    now = pd.Timestamp.now(tz="UTC")
    local_day = now.tz_convert(config.timezone).date()
    session = cal.date_to_session(local_day, direction="next")
    if now >= cal.session_open(session):
        session = cal.next_session(session)
    previous = cal.previous_session(session)
    cutoff = cal.session_open(session) - pd.Timedelta(minutes=5)
    path = output / "open2open_v1" / "plans" / f"{session.date()}.json"
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    book = book_config()
    recording = book["track_every_ranked_session"] and session >= pd.Timestamp(book["start_session"])
    record = {
        "strategy_id": spec["id"], "spec_sha256": hashlib.sha256(SPEC.read_bytes()).hexdigest(),
        "trade_session": str(session.date()), "exit_session": str(cal.next_session(session).date()),
        "feature_session": str(previous.date()), "cutoff": cutoff.isoformat(),
        "generated_at": now.isoformat(), "real_orders_enabled": False,
        "cost_bps": spec["cost_bps"], "candidates": [],
    }
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior["status"] in {"READY", "CASH"}:
            return prior
    try:
        if not cal.session_close(previous) < now < cutoff:
            raise ValueError("尚未进入可使用完整前一日行情的盘前窗口")
        try:
            snapshot = prior_universe(output, now)
            record["universe_status"] = "RECENT_SNAPSHOT"
        except ValueError:
            if not recording:
                raise
            snapshot = prior_universe(output, now, max_age_days=None)
            record["universe_status"] = "LAST_KNOWN_SNAPSHOT"
        panel = training_panel()
        daily, fields = daily_context(panel)
        if not recording and len(cal.sessions_in_range(daily.index[-1].date(), previous)) > spec["stock_retrain"] + 1:
            raise ValueError("训练面板超过一个重训周期未更新")
        context = market_inputs(previous, session, fields, cal)
        stocks = stock_inputs(output, pd.read_parquet(snapshot), previous, config)
        if len(stocks) < config.top_n:
            raise ValueError("可评分标的不足")
        # Freeze the training anchors for this forward paper book.
        fit_session = pd.Timestamp(book["start_session"]) if recording else session
        market_sessions = cal.sessions_in_range(daily.index[0].date(), fit_session)
        pos = len(market_sessions) - 1
        def anchor(period):
            return market_sessions[pos - pos % period].tz_localize(config.timezone).tz_convert("UTC")
        stock_anchor = anchor(spec["stock_retrain"])
        market_anchor = anchor(spec["market_retrain"])
        gate = fit_market(daily, fields, market_anchor, spec)
        forecast = float(gate.predict(context)[0])
        if not np.isfinite(forecast):
            raise ValueError("市场门槛预测不可用")
        with market.PANEL.open("rb") as stream:
            panel_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        with snapshot.open("rb") as stream:
            universe_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        record.update(market_forecast_bps=forecast, universe_snapshot=snapshot.name,
                      universe_sha256=universe_hash, training_panel_sha256=panel_hash,
                      stock_anchor=stock_anchor.isoformat(), market_anchor=market_anchor.isoformat(),
                      market_training_last_label_end=settled_days(
                          daily, market_anchor, spec["market_train_sessions"]
                      ).label_end.max().isoformat())
        record["status"] = "CASH"
        record["market_gate_passed"] = forecast > spec["market_threshold_bps"]
        if recording or record["market_gate_passed"]:
            model = fit_ranker(panel, daily, stock_anchor, spec)
            record["stock_training_last_label_end"] = settled_days(
                daily, stock_anchor, spec["stock_train_sessions"]
            ).label_end.max().isoformat()
            stocks["score"] = model.predict(stocks[spec["stock_features"]])
            records = stocks.nlargest(config.top_n, "score")[["symbol", "name", "score"]]
            record["candidates"] = json.loads(records.to_json(orient="records"))
            for rank, candidate in enumerate(record["candidates"], 1):
                candidate["rank"] = rank
            record["status"] = "READY"
        record["generated_at"] = pd.Timestamp.now(tz="UTC").isoformat()
        if pd.Timestamp(record["generated_at"]) >= cutoff:
            raise ValueError("评分完成时已超过盘前截止时间")
    except (ValueError, FileNotFoundError) as exc:
        record.update(status="DATA_PENDING", reason=str(exc), candidates=[])
    write_record(path, record)
    return record


def settle(output, cal, *, refresh=False, official=False):
    directory = output / "open2open_v1"
    count = 0
    book = book_config()
    now = pd.Timestamp.now(tz="UTC")
    for path in sorted((directory / "plans").glob("*.json")):
        if official and path.stem < book["start_session"]:
            continue
        target = directory / "settled" / path.name
        plan = json.loads(path.read_text(encoding="utf-8"))
        if target.exists() or plan["status"] not in {"READY", "CASH"}:
            continue
        entry, exit_ = map(pd.Timestamp, (plan["trade_session"], plan["exit_session"]))
        if str(entry.date()) != path.stem or cal.next_session(entry) != exit_:
            raise ValueError(f"Invalid open-to-open session pair in {path.name}")
        if now <= cal.session_open(entry):
            continue
        entry_path = directory / "entries" / path.name
        exiting = now > cal.session_open(exit_)
        if not exiting and entry_path.exists():
            continue
        positions, realized = [], []
        for candidate in plan["candidates"]:
            symbol = candidate["symbol"]
            if refresh:
                try:
                    bars = download(symbol, str(entry.date()), str((exit_ if exiting else entry).date()), timeout=8)
                except (OSError, KeyError, TypeError, IndexError, ValueError):
                    break
                bars["session"] = pd.to_datetime(bars.timestamp, utc=True).dt.tz_convert(
                    "America/New_York"
                ).dt.tz_localize(None).dt.normalize()
            else:
                bars = bars_at(output / "daily_yahoo_adjusted" / f"{symbol.replace('.', '_')}.parquet", exit_)
            if bars.empty or entry not in set(bars.session):
                break
            bars = bars.set_index("session")
            start = float(bars.loc[entry, "open"])
            if not np.isfinite(start) or start <= 0:
                break
            allocation = book["allocation_usd"]
            position = candidate | {
                "entry_open": start, "allocation_usd": allocation, "shares": allocation / start,
            }
            positions.append(position)
            if not exiting or exit_ not in bars.index:
                continue
            end = float(bars.loc[exit_, "open"])
            dividend = bars.loc[exit_, "cash_dividend"] if "cash_dividend" in bars else 0.0
            if not np.isfinite([start, end, dividend]).all() or min(start, end) <= 0:
                break
            net = (end + dividend) / start - 1 - plan["cost_bps"] / 1e4
            realized.append(position | {
                "exit_open": end, "dividend_per_share": float(dividend), "net_return": net,
                "pnl_usd": allocation * net, "cost_usd": allocation * plan["cost_bps"] / 1e4,
            })
        if len(positions) == len(plan["candidates"]) and not entry_path.exists():
            write_record(entry_path, {"positions": positions, "recorded_at": now.isoformat()})
        if not exiting or len(realized) != len(plan["candidates"]):
            continue
        pnl = sum(row["pnl_usd"] for row in realized)
        write_record(target, plan | {
            "settled_at": pd.Timestamp.now(tz="UTC").isoformat(), "realized": realized,
            "capital_usd": book["capital_usd"], "pnl_usd": pnl,
            "daily_return": pnl / book["capital_usd"],
            "top1_net_return": realized[0]["net_return"] if realized else 0.0,
            "top10_net_return": float(np.mean([row["net_return"] for row in realized])) if realized else 0.0,
        })
        count += 1
    return {"newly_settled": count, "real_orders_enabled": False}


def main(action=None):
    if action is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("action", choices=["score", "settle"])
        action = parser.parse_args().action
    config = USOvernightConfig.from_yaml(ROOT / "configs/us_overnight_paper_only.yaml")
    output = ROOT / config.output_root
    cal = xcals.get_calendar("XNYS")
    result = score(output, config, cal) if action == "score" else settle(output, cal, refresh=True, official=True)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
