"""Validate the full cached Futu raw daily universe against Yahoo adjusted bars."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FUTU = ROOT / "data/us_overnight_paper_only/daily"
YAHOO = ROOT / "data/us_overnight_paper_only/daily_yahoo_adjusted"
OUT = ROOT / "state/us-overnight-full-source-validation"


def dated(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    result = frame.copy()
    if source == "futu":
        stamp = pd.to_datetime(result["timestamp"], utc=True)
    else:
        stamp = pd.to_datetime(result["timestamp"], utc=True).dt.tz_convert("America/New_York")
    result["date"] = stamp.dt.date.astype(str)
    return result


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    anomaly_rows = []
    yahoo_paths = {path.stem: path for path in YAHOO.glob("US_*.parquet")}
    for futu_path in sorted(FUTU.glob("US_*.parquet")):
        yahoo_path = yahoo_paths.get(futu_path.stem)
        if yahoo_path is None:
            continue
        futu = dated(pd.read_parquet(futu_path), "futu")
        yahoo = dated(pd.read_parquet(yahoo_path), "yahoo")
        symbol = str(yahoo.iloc[0]["symbol"])
        merged = futu[["date", "open", "close", "volume"]].merge(
            yahoo[["date", "open", "close", "volume", "cash_dividend"]],
            on="date", suffixes=("_futu", "_yahoo"), how="inner",
        ).sort_values("date")
        ratio = merged["close_futu"] / merged["close_yahoo"]
        ratio_change = ratio.pct_change(fill_method=None).abs()
        corporate_boundary = (ratio_change > 0.05) | (ratio_change.shift(-1) > 0.05)
        futu_total = (
            (merged["open_futu"].shift(-1) + merged["cash_dividend"].shift(-1))
            / merged["close_futu"] - 1
        )
        yahoo_total = (
            (merged["open_yahoo"].shift(-1) + merged["cash_dividend"].shift(-1))
            / merged["close_yahoo"] - 1
        )
        label_diff = (futu_total - yahoo_total).abs()
        stable = ~corporate_boundary & label_diff.notna()
        stable_diff = label_diff.loc[stable]
        mismatch = stable & (label_diff > 0.001)
        extreme = yahoo_total.abs() > 0.5
        if mismatch.any() or extreme.any():
            detail = merged.loc[mismatch | extreme, [
                "date", "open_futu", "close_futu", "open_yahoo", "close_yahoo", "cash_dividend"
            ]].copy()
            detail["symbol"] = symbol
            detail["label_diff_bps"] = label_diff.loc[detail.index] * 10_000
            detail["yahoo_total_return"] = yahoo_total.loc[detail.index]
            detail["corporate_boundary"] = corporate_boundary.loc[detail.index]
            anomaly_rows.append(detail)
        rows.append({
            "symbol": symbol,
            "overlap_rows": len(merged),
            "missing_in_yahoo": len(set(futu["date"]) - set(yahoo["date"])),
            "missing_in_futu": len(set(yahoo["date"]) - set(futu["date"])),
            "corporate_boundaries": int(corporate_boundary.sum()),
            "stable_label_median_abs_diff_bps": float(stable_diff.median() * 10_000),
            "stable_label_p99_abs_diff_bps": float(stable_diff.quantile(0.99) * 10_000),
            "stable_label_max_abs_diff_bps": float(stable_diff.max() * 10_000),
            "stable_label_rows_over_10bps": int(mismatch.sum()),
            "extreme_yahoo_label_rows": int(extreme.sum()),
            "median_volume_abs_rel_diff": float(
                (merged["volume_yahoo"] / merged["volume_futu"].replace(0, np.nan) - 1).abs().median()
            ),
        })
    summary = pd.DataFrame(rows)
    data_mismatch = summary.loc[
        (summary["stable_label_rows_over_10bps"] >= 3)
        | (summary["stable_label_p99_abs_diff_bps"] > 10)
    ].sort_values(["stable_label_rows_over_10bps", "stable_label_p99_abs_diff_bps"], ascending=False)
    extreme = summary.loc[summary["extreme_yahoo_label_rows"] > 0].sort_values(
        "extreme_yahoo_label_rows", ascending=False
    )
    summary.to_csv(OUT / "summary.csv", index=False)
    data_mismatch.to_csv(OUT / "data_mismatch_quarantine.csv", index=False)
    extreme.to_csv(OUT / "extreme_move_review.csv", index=False)
    if anomaly_rows:
        pd.concat(anomaly_rows, ignore_index=True).to_csv(OUT / "anomalies.csv", index=False)
    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "symbols_compared": len(summary),
        "data_mismatch_quarantine": data_mismatch["symbol"].tolist(),
        "extreme_move_review": extreme["symbol"].tolist(),
        "median_symbol_p99_label_diff_bps": float(summary["stable_label_p99_abs_diff_bps"].median()),
        "p99_symbol_p99_label_diff_bps": float(summary["stable_label_p99_abs_diff_bps"].quantile(0.99)),
    }
    (OUT / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
