"""Read-only daily-bar validation between Yahoo adjusted data and Futu OpenD."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from futu import RET_OK, AuType, KLType, OpenQuoteContext

ROOT = Path(__file__).resolve().parents[1]
YAHOO = ROOT / "data/us_overnight_paper_only/daily_yahoo_adjusted"
OUT = ROOT / "state/us-overnight-source-validation"
SYMBOLS = ["US.AAPL", "US.MSFT", "US.NVDA", "US.CMG", "US.AVGO", "US.SMCI",
           "US.KO", "US.XOM", "US.GME", "US.QXO"]


def futu_history(context: OpenQuoteContext, symbol: str) -> pd.DataFrame:
    parts = []
    page = None
    while True:
        status, data, page = context.request_history_kline(
            symbol, start="2023-09-01", end="2026-11-06", ktype=KLType.K_DAY,
            autype=AuType.QFQ, max_count=1000, page_req_key=page,
        )
        if status != RET_OK:
            raise RuntimeError(f"{symbol}: {data}")
        parts.append(data)
        if page is None:
            break
    frame = pd.concat(parts, ignore_index=True)
    frame["date"] = pd.to_datetime(frame["time_key"]).dt.date.astype(str)
    return frame


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    context = OpenQuoteContext(host="127.0.0.1", port=11111)
    rows = []
    details = []
    try:
        for symbol in SYMBOLS:
            futu = futu_history(context, symbol)
            yahoo_path = YAHOO / f"{symbol.replace('.', '_')}.parquet"
            yahoo = pd.read_parquet(yahoo_path)
            yahoo["date"] = pd.to_datetime(yahoo["timestamp"], utc=True).dt.tz_convert(
                "America/New_York"
            ).dt.date.astype(str)
            merged = futu[["date", "open", "high", "low", "close", "volume"]].merge(
                yahoo[["date", "open", "high", "low", "close", "volume", "cash_dividend"]],
                on="date", suffixes=("_futu", "_yahoo"), how="inner",
            )
            for column in ("open", "high", "low", "close"):
                merged[f"{column}_rel_diff"] = (
                    merged[f"{column}_yahoo"] / merged[f"{column}_futu"] - 1
                )
            merged["volume_rel_diff"] = (
                merged["volume_yahoo"] / merged["volume_futu"].replace(0, np.nan) - 1
            )
            merged["futu_next_open_return"] = (
                merged["open_futu"].shift(-1) / merged["close_futu"] - 1
            )
            merged["yahoo_next_open_total_return"] = (
                (merged["open_yahoo"].shift(-1) + merged["cash_dividend"].shift(-1))
                / merged["close_yahoo"] - 1
            )
            merged["label_abs_diff"] = (
                merged["futu_next_open_return"] - merged["yahoo_next_open_total_return"]
            ).abs()
            price_diff = merged[[f"{column}_rel_diff" for column in ("open", "high", "low", "close")]].abs()
            rows.append({
                "symbol": symbol, "futu_rows": len(futu), "yahoo_rows": len(yahoo),
                "overlap_rows": len(merged),
                "missing_in_yahoo": len(set(futu["date"]) - set(yahoo["date"])),
                "missing_in_futu": len(set(yahoo["date"]) - set(futu["date"])),
                "median_price_abs_rel_diff": float(price_diff.stack().median()),
                "p99_price_abs_rel_diff": float(price_diff.stack().quantile(0.99)),
                "max_price_abs_rel_diff": float(price_diff.stack().max()),
                "median_volume_abs_rel_diff": float(merged["volume_rel_diff"].abs().median()),
                "price_rows_over_1pct": int((price_diff.max(axis=1) > 0.01).sum()),
                "median_label_abs_diff_bps": float(merged["label_abs_diff"].median() * 10_000),
                "p99_label_abs_diff_bps": float(merged["label_abs_diff"].quantile(0.99) * 10_000),
                "max_label_abs_diff_bps": float(merged["label_abs_diff"].max() * 10_000),
                "label_rows_over_10bps": int((merged["label_abs_diff"] > 0.001).sum()),
            })
            anomalies = merged.loc[price_diff.max(axis=1) > 0.01].copy()
            anomalies["symbol"] = symbol
            details.append(anomalies)
    finally:
        context.close()
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    if details:
        pd.concat(details, ignore_index=True).to_csv(OUT / "price_anomalies_over_1pct.csv", index=False)
    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(), "adjustment": "Futu QFQ vs Yahoo split-adjusted",
        "symbols": SYMBOLS, "summary": rows,
    }
    (OUT / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
