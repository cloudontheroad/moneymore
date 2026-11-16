"""Bounded, resumable official inputs; ONLY writes this isolated research cache."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from prepare_four_strategy_2026 import save

from moneymore.data.tushare_provider import TushareProvider
from moneymore.research.ashare_mainline import sha256

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "state/ashare-three-year-20261113"
CONFIG = ROOT / "configs/ashare_three_year_v1.json"


class Fetcher:
    def __init__(self):
        self.config = json.loads(CONFIG.read_bytes())
        self.output = BASE / "data-v1"
        self.output.mkdir(parents=True, exist_ok=True)
        load_dotenv(ROOT / ".env")
        self.client = TushareProvider()._client
        # Tushare supports a bounded request timeout; never log the client or token.
        if hasattr(self.client, "_DataApi__timeout"):
            self.client._DataApi__timeout = 25
        self.calls = 0
        self.consecutive_failures = 0
        self.journal = self.output / "requests.jsonl"
        self.previous_calls = (
            len(self.journal.read_text().splitlines()) if self.journal.exists() else 0
        )

    def get(self, endpoint, tag, *, nonempty=False, **params):
        path = self.output / "official" / f"{endpoint}-{tag}.parquet"
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            return pd.read_parquet(path)
        for attempt in range(self.config["fetch"]["max_attempts"]):
            if self.previous_calls + self.calls >= self.config["fetch"]["max_requests"]:
                raise RuntimeError("BOUNDED_REQUEST_BUDGET_EXHAUSTED")
            self.calls += 1
            try:
                frame = getattr(self.client, endpoint)(**params)
                if frame is None or (nonempty and frame.empty):
                    raise ValueError("EMPTY_REQUIRED_ENDPOINT")
                tmp = path.with_suffix(".tmp")
                frame.to_parquet(tmp, index=False)
                tmp.replace(path)
                self.consecutive_failures = 0
                status = "OK"
            except Exception as exc:  # noqa: BLE001 - sanitize provider exceptions; bounded retry
                body = str(exc).lower()
                status = (
                    "PERMISSION"
                    if any(s in body for s in ("权限", "积分", "permission"))
                    else "RATE_LIMIT"
                    if any(s in body for s in ("频次", "每分钟", "rate limit"))
                    else type(exc).__name__
                )
                self.consecutive_failures += 1
            finally:
                with self.journal.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "endpoint": endpoint,
                                "tag": tag,
                                "attempt": attempt + 1,
                                "status": status,
                                "utc": datetime.now(UTC).isoformat(),
                            }
                        )
                        + "\n"
                    )
                time.sleep(self.config["fetch"]["interval_seconds"])
            if status == "OK":
                return frame
            if status == "PERMISSION" or self.consecutive_failures >= 3:
                raise RuntimeError("FETCH_CIRCUIT_OPEN:" + endpoint + ":" + status)
            if attempt + 1 < self.config["fetch"]["max_attempts"]:
                time.sleep(5)
        raise RuntimeError("FETCH_FAILED_ERROR_BODY_SUPPRESSED:" + endpoint + ":" + tag)

    def progress(self, phase, **values):
        row = {
            "phase": phase,
            "pid": os.getpid(),
            "utc": datetime.now(UTC).isoformat(),
            "new_requests": self.calls,
            **values,
        }
        save(self.output / "progress.json", row)
        print(json.dumps(row), flush=True)


def available_cached(endpoint, day):
    roots = [
        ROOT / "state/four-strategy-2026-jan-sep-v1/official",
        ROOT / "state/ashare-mainline-20261112/n4-data-v1/official",
    ]
    candidates = [r / f"{endpoint}-{day}.parquet" for r in roots]
    if endpoint == "daily_basic":
        candidates += [ROOT / "state/baseline-v2-validation/daily-inputs" / f"{day}.parquet"]
    for path in candidates:
        if not path.exists():
            continue
        proof_path = path.with_suffix(".json")
        if path.parent.name == "daily-inputs" and (
            not proof_path.exists()
            or json.loads(proof_path.read_bytes()).get("sha256") != sha256(path)
        ):
            continue
        frame = pd.read_parquet(path)
        if endpoint in {"daily_basic", "stk_limit"} and len(frame) < 3000:
            continue
        param = "record_date" if endpoint == "dividend" else "trade_date"
        if not frame.empty and (param not in frame or not frame[param].astype(str).eq(day).all()):
            continue
        return path, frame
    return None, None


def prepare(mode):
    f = Fetcher()
    cfg = f.config
    cal = f.get(
        "trade_cal",
        "20210101-20261030",
        nonempty=True,
        exchange="SSE",
        start_date=cfg["raw_start"],
        end_date=cfg["evaluation_end"],
    )
    days = sorted(cal.loc[cal.is_open.eq(1), "cal_date"].astype(str))
    save(
        f.output / "calendar.json",
        {
            "days": days,
            "evaluation": [
                d for d in days if cfg["evaluation_start"] <= d <= cfg["evaluation_end"]
            ],
        },
    )
    if mode == "market":
        basic_pieces = []
        provenance = []
        needed = [d for d in days if d >= cfg["training_start"]]
        for i, day in enumerate(needed):
            dest = f.output / "official" / f"daily_basic-{day}.parquet"
            if dest.exists():
                b = pd.read_parquet(dest)
            else:
                source, b = available_cached("daily_basic", day)
                if source:
                    dest.parent.mkdir(exist_ok=True)
                    dest.write_bytes(source.read_bytes())
                else:
                    b = f.get("daily_basic", day, nonempty=True, trade_date=day)
            if (
                len(b) < 3000
                or not b.trade_date.astype(str).eq(day).all()
                or b.ts_code.duplicated().any()
            ):
                raise ValueError("FULL_CROSS_SECTION_NOT_VERIFIED:" + day)
            b = b.loc[b.ts_code.str.endswith((".SH", ".SZ"))].copy()
            basic_pieces.append(b)
            provenance.append({"date": day, "rows": len(b), "sha256": sha256(dest)})
            if i % 20 == 0:
                f.progress("MARKET_BASIC", completed=i + 1, total=len(needed), day=day)
        basic = pd.concat(basic_pieces, ignore_index=True)
        basic.to_parquet(f.output / "daily_basic.parquet", index=False)
        # Contemporary total capitalisation, no filter on today's surviving stock list.
        eligible = basic.loc[basic.total_mv.gt(0)].sort_values(
            ["trade_date", "total_mv", "ts_code"], ascending=[True, False, True]
        )
        members = (
            eligible.groupby("trade_date", sort=True)
            .head(cfg["universe_size"])[["trade_date", "ts_code"]]
            .rename(columns={"trade_date": "date", "ts_code": "symbol"})
        )
        if not members.groupby("date").size().eq(cfg["universe_size"]).all():
            raise ValueError("INCOMPLETE_HISTORICAL_TOP1000")
        members.to_parquet(f.output / "membership.parquet", index=False)
        symbols = sorted(members.symbol.unique())
        filters = [
            ("trade_date", ">=", cfg["raw_start"]),
            ("trade_date", "<=", cfg["evaluation_end"]),
            ("ts_code", "in", symbols),
        ]
        bars = pd.read_parquet(
            ROOT / "data/processed/daily.parquet",
            columns=["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"],
            filters=filters,
        )
        adj = pd.read_parquet(ROOT / "data/processed/adj_factor.parquet", filters=filters)
        bars.trade_date = bars.trade_date.astype(str)
        adj.trade_date = adj.trade_date.astype(str)
        # Missing whole dates must be fetched, never made zero-return sessions.
        count = bars.groupby("trade_date").ts_code.nunique()
        for day in days:
            if count.get(day, 0) < 500:
                b = f.get("daily", day, nonempty=True, trade_date=day)
                a = f.get("adj_factor", day, nonempty=True, trade_date=day)
                bars = pd.concat(
                    [bars, b.loc[b.ts_code.isin(symbols), bars.columns]], ignore_index=True
                )
                adj = pd.concat(
                    [adj, a.loc[a.ts_code.isin(symbols), adj.columns]], ignore_index=True
                )
        bars = bars.drop_duplicates(["trade_date", "ts_code"], keep="last").merge(
            adj.drop_duplicates(["trade_date", "ts_code"], keep="last"),
            on=["trade_date", "ts_code"],
            how="left",
            validate="one_to_one",
        )
        if bars.loc[bars.close.gt(0), "adj_factor"].isna().any():
            raise ValueError("ADJUSTMENT_MISSING_FOR_ACTUAL_PRICE")
        bars.to_parquet(f.output / "bars.parquet", index=False)
        for name in ["baseline_v2_industry_membership", "instruments"]:
            pd.read_parquet(ROOT / f"data/processed/{name}.parquet").to_parquet(
                f.output / f"{name}.parquet", index=False
            )
        covered = members.merge(
            bars[["trade_date", "ts_code", "close"]],
            left_on=["date", "symbol"],
            right_on=["trade_date", "ts_code"],
            how="left",
            validate="one_to_one",
        )
        coverage = covered.groupby("date").close.count()
        coverage.rename("observed_prices").to_frame().to_parquet(
            f.output / "daily-coverage.parquet"
        )
        save(
            f.output / "market-results.json",
            {
                "status": "COMPLETE_PENDING_HALT_AUDIT",
                "membership_days": members.date.nunique(),
                "symbols": len(symbols),
                "price_rows": len(bars),
                "min_daily_member_prices": int(coverage.min()),
                "basic_provenance": provenance,
                "hashes": {
                    name: sha256(f.output / name)
                    for name in ["daily_basic.parquet", "membership.parquet", "bars.parquet"]
                },
                "production_modified": False,
            },
        )
        f.progress("MARKET_COMPLETE", symbols=len(symbols), member_days=members.date.nunique())
    elif mode == "core_financial":
        from moneymore.data.fundamental_research import research_financial_frame
        from moneymore.data.store import ParquetStore

        members = pd.read_parquet(f.output / "membership.parquet")
        core = research_financial_frame(ParquetStore(ROOT / "data"))
        missing = sorted(set(members.symbol) - set(core.ts_code))
        extra = []
        for i, symbol in enumerate(missing):
            frame = f.get(
                "fina_indicator",
                symbol + "-core",
                ts_code=symbol,
                start_date="20200101",
                end_date=cfg["evaluation_end"],
                fields="ts_code,ann_date,end_date,update_flag,roe,q_sales_yoy,q_roe,netdebt,fcff",
            )
            frame["growth_source"] = "TUSHARE_FINA_INDICATOR"
            frame["conflict__roe"] = frame["conflict__q_sales_yoy"] = False
            extra.append(frame)
            if i % 10 == 0:
                f.progress("CORE_FINANCIAL", completed=i + 1, total=len(missing))
        core = pd.concat([core, *extra], ignore_index=True)
        core.to_parquet(f.output / "core-financials.parquet", index=False)
        save(
            f.output / "core-financial-results.json",
            {
                "status": "COMPLETE",
                "supplemented_symbols": missing,
                "sha256": sha256(f.output / "core-financials.parquet"),
                "production_modified": False,
            },
        )
        f.progress("CORE_FINANCIAL_COMPLETE", supplemented_symbols=len(missing))
    elif mode == "financial":
        from moneymore.data.fundamental_research import research_financial_frame
        from moneymore.data.store import ParquetStore

        members = pd.read_parquet(f.output / "membership.parquet")
        symbols = sorted(members.symbol.unique())
        old = ROOT / "state/ashare-simple-rules-20261113/data-v1"
        fields = {
            "cashflow": "ts_code,ann_date,f_ann_date,end_date,report_type,update_flag,n_cashflow_act",
            "income": "ts_code,ann_date,f_ann_date,end_date,report_type,update_flag,operate_profit,n_income_attr_p",
            "fina_indicator": "ts_code,ann_date,end_date,update_flag,roe,q_sales_yoy,q_roe,netdebt,fcff",
        }
        frames = {e: [] for e in fields}
        sources = []
        for i, symbol in enumerate(symbols):
            for endpoint, columns in fields.items():
                cached = old / f"{endpoint}-{symbol}.parquet"
                if cached.exists():
                    frame = pd.read_parquet(cached)
                    sources.append(
                        {"file": str(cached.relative_to(ROOT)), "sha256": sha256(cached)}
                    )
                    # Indicator dates are reporting-period bounds; the existing cache
                    # already includes Q3 announcements. Statements filter announcement dates.
                    if endpoint != "fina_indicator":
                        newer = f.get(
                            endpoint,
                            symbol + "-202610",
                            ts_code=symbol,
                            start_date="20261001",
                            end_date=cfg["evaluation_end"],
                            fields=columns,
                        )
                        frame = pd.concat([frame, newer], ignore_index=True).drop_duplicates()
                else:
                    frame = f.get(
                        endpoint,
                        symbol,
                        ts_code=symbol,
                        start_date="20200101",
                        end_date=cfg["evaluation_end"],
                        fields=columns,
                    )
                frames[endpoint].append(frame)
            if i % 20 == 0:
                f.progress("FINANCIAL", completed=i + 1, total=len(symbols), symbol=symbol)
        for endpoint, parts in frames.items():
            pd.concat(parts, ignore_index=True).drop_duplicates().to_parquet(
                f.output / f"{endpoint}.parquet", index=False
            )
        core_path = f.output / "core-financials.parquet"
        if core_path.exists():
            core = pd.read_parquet(core_path)
        else:
            core = research_financial_frame(ParquetStore(ROOT / "data"))
        fresh = pd.concat(frames["fina_indicator"], ignore_index=True)
        # Preserve the approved historical conflict flags and growth derivations.
        for field in ["roe", "q_sales_yoy"]:
            if field not in fresh:
                fresh[field] = float("nan")
        known = pd.MultiIndex.from_frame(core[["ts_code", "ann_date", "end_date"]])
        new_rows = fresh.loc[
            ~pd.MultiIndex.from_frame(fresh[["ts_code", "ann_date", "end_date"]]).isin(known)
        ].copy()
        new_rows["growth_source"] = "TUSHARE_FINA_INDICATOR"
        new_rows["conflict__roe"] = new_rows["conflict__q_sales_yoy"] = False
        # Do not change the already frozen core input when the augmentation finishes.
        if not core_path.exists():
            core = pd.concat([core, new_rows], ignore_index=True)
            core.to_parquet(core_path, index=False)
        save(
            f.output / "financial-results.json",
            {
                "status": "COMPLETE_PENDING_COVERAGE",
                "symbols": len(symbols),
                "cached_sources": sources,
                "hashes": {
                    e: sha256(f.output / f"{e}.parquet") for e in [*fields, "core-financials"]
                },
                "original_vintage_certified": False,
            },
        )
        f.progress("FINANCIAL_COMPLETE", symbols=len(symbols))
    elif mode == "execution":
        first_signal = days[days.index(next(d for d in days if d >= cfg["evaluation_start"])) - 1]
        needed = [d for d in days if d >= first_signal]
        # Halt evidence is also needed by historical risk windows.
        halt_days = [d for d in days if d >= "20230501"]
        jobs = [
            (d, e)
            for d in halt_days
            for e in (["stk_limit", "suspend_d", "dividend"] if d in needed else ["suspend_d"])
        ]
        rows = []
        for i, (day, endpoint) in enumerate(jobs):
            dest = f.output / "official" / f"{endpoint}-{day}.parquet"
            if not dest.exists():
                source, frame = available_cached(endpoint, day)
                if source:
                    dest.write_bytes(source.read_bytes())
            frame = f.get(
                endpoint,
                day,
                nonempty=endpoint == "stk_limit",
                **{("record_date" if endpoint == "dividend" else "trade_date"): day},
            )
            if endpoint == "stk_limit" and len(frame) < 3000:
                raise ValueError("LIMIT_CROSS_SECTION_TRUNCATED")
            rows.append({"file": dest.name, "rows": len(frame), "sha256": sha256(dest)})
            if i % 30 == 0:
                f.progress("EXECUTION", completed=i + 1, total=len(jobs), day=day)
        save(f.output / "execution-results.json", {"status": "COMPLETE", "files": rows})
        f.progress("EXECUTION_COMPLETE", files=len(rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode", choices=["market", "execution", "financial", "core_financial"], required=True
    )
    a = p.parse_args()
    try:
        prepare(a.mode)
    except Exception as exc:
        save(
            BASE / "data-v1/failure.json",
            {
                "status": "FAILED",
                "mode": a.mode,
                "error_type": type(exc).__name__,
                "message": str(exc)
                if isinstance(exc, (ValueError, RuntimeError))
                else "details_suppressed",
            },
        )
        raise
