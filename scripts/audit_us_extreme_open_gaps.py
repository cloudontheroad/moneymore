"""Cross-source audit of investable open-to-next-open gaps."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "data/us_overnight_corrected/reports"
FUTU = ROOT / "data/us_overnight_paper_only/daily"
PREDICTIONS = REPORTS / "investable_corrected_open2open_ensemble_search_winner.parquet"
THIRD_PARTY_CONFIRMATIONS = {
    ("US.APP", "2026-11-04"): {
        "next_open": 349.83,
        "price_source": "https://finance.yahoo.com/quote/APP/history/",
        "event_source": "https://investors.applovin.com/files/content_files/3Q26-Earnings-Press-Release.pdf",
        "event": "Q3 2026 earnings released after the 2026-11-04 close",
    }
}


def futu_observation(symbol: str, signal_date) -> dict[str, object]:
    path = FUTU / f"{symbol.replace('.', '_')}.parquet"
    if not path.exists():
        return {"secondary_status": "NO_FUTU_FILE"}
    bars = pd.read_parquet(path).sort_values("timestamp").copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars["session_date"] = bars["timestamp"].dt.date
    positions = bars.index[bars["session_date"] == signal_date].tolist()
    if not positions:
        return {"secondary_status": "NO_SIGNAL_DATE"}
    location = bars.index.get_loc(positions[0])
    if isinstance(location, slice) or location + 1 >= len(bars):
        return {"secondary_status": "NO_NEXT_SESSION"}
    signal = bars.iloc[location]
    following = bars.iloc[location + 1]
    signal_open = float(signal["open"])
    next_open = float(following["open"])
    if not np.isfinite(signal_open) or not np.isfinite(next_open) or signal_open <= 0:
        return {"secondary_status": "INVALID_FUTU_PRICE"}
    return {
        "secondary_status": "AVAILABLE",
        "futu_open": signal_open,
        "futu_next_open": next_open,
        "futu_open_to_next_open": next_open / signal_open - 1,
        "futu_signal_volume": float(signal["volume"]),
        "futu_next_volume": float(following["volume"]),
        "futu_next_timestamp": str(following["timestamp"]),
    }


def main() -> int:
    gaps = pd.read_csv(REPORTS / "open_gap_abs30pct_audit.csv")
    gaps = gaps.loc[gaps["investable"].astype(str).str.lower().eq("true")].copy()
    gaps["timestamp"] = pd.to_datetime(gaps["timestamp"], utc=True)
    observations = []
    for row in gaps.itertuples(index=False):
        observation = futu_observation(row.symbol, row.timestamp.date())
        observations.append(observation)
    secondary = pd.DataFrame(observations)
    audited = pd.concat([gaps.reset_index(drop=True), secondary], axis=1)
    audited["gap_difference"] = (
        audited["open_to_next_open"] - audited["futu_open_to_next_open"]
    ).abs()
    audited["volume_relative_difference"] = (
        (audited["signal_day_volume"] - audited["futu_signal_volume"]).abs()
        / audited[["signal_day_volume", "futu_signal_volume"]].max(axis=1).replace(0, np.nan)
    )
    audited["audit_class"] = "NO_SECONDARY_CONFIRMATION"
    available = audited["secondary_status"].eq("AVAILABLE")
    audited.loc[available & (audited["gap_difference"] <= 0.005), "audit_class"] = (
        "CROSS_SOURCE_CONFIRMED"
    )
    audited.loc[available & (audited["gap_difference"] > 0.005), "audit_class"] = (
        "SOURCE_MISMATCH"
    )
    audited["third_party_price_source"] = ""
    audited["event_source"] = ""
    audited["event_explanation"] = ""
    for index, row in audited.iterrows():
        key = (str(row["symbol"]), str(row["timestamp"].date()))
        confirmation = THIRD_PARTY_CONFIRMATIONS.get(key)
        if confirmation is None:
            continue
        third_party_gap = float(confirmation["next_open"]) / float(row["open"]) - 1
        if abs(third_party_gap - float(row["open_to_next_open"])) <= 0.005:
            audited.at[index, "audit_class"] = "THIRD_PARTY_CONFIRMED"
        audited.at[index, "third_party_price_source"] = confirmation["price_source"]
        audited.at[index, "event_source"] = confirmation["event_source"]
        audited.at[index, "event_explanation"] = confirmation["event"]
    predictions = pd.read_parquet(PREDICTIONS)
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], utc=True)
    audited = audited.merge(
        predictions[["timestamp", "symbol", "rank", "label"]],
        on=["timestamp", "symbol"], how="left", validate="one_to_one",
    )
    audited["selected_top10"] = audited["rank"].notna()
    audited["selected_top1"] = audited["rank"].eq(1)
    audited = audited.sort_values("open_to_next_open", key=lambda value: value.abs(), ascending=False)
    audited.to_csv(REPORTS / "investable_extreme_gap_cross_source_audit.csv", index=False)

    selected = audited.loc[audited["selected_top10"]]
    selected_top1 = audited.loc[audited["selected_top1"]]
    summary = {
        "audited_at_utc": datetime.now(UTC).isoformat(),
        "scope": "investable abs(open_t_to_open_t_plus_1) >= 30%",
        "rows": int(len(audited)),
        "audit_classes": audited["audit_class"].value_counts().to_dict(),
        "maximum_cross_source_gap_difference": float(
            audited.loc[available, "gap_difference"].max()
        ) if available.any() else None,
        "median_cross_source_gap_difference": float(
            audited.loc[available, "gap_difference"].median()
        ) if available.any() else None,
        "maximum_signal_volume_relative_difference": float(
            audited.loc[available, "volume_relative_difference"].max()
        ) if available.any() else None,
        "selected_top10_extreme_rows": int(len(selected)),
        "selected_top1_extreme_rows": int(len(selected_top1)),
        "selected_top10_cross_source_confirmed": int(
            selected["audit_class"].eq("CROSS_SOURCE_CONFIRMED").sum()
        ),
        "selected_top1_cross_source_confirmed": int(
            selected_top1["audit_class"].eq("CROSS_SOURCE_CONFIRMED").sum()
        ),
        "third_party_confirmed": int(
            audited["audit_class"].eq("THIRD_PARTY_CONFIRMED").sum()
        ),
        "selected_top10_label_sum": float(selected["label"].sum()),
        "selected_top1_label_sum": float(selected_top1["label"].sum()),
        "decision_rule": {
            "confirmed": "absolute Yahoo-vs-Futu return difference <= 50 bps",
            "source_mismatch": "quarantine pending corporate-action repair",
            "confirmed_large_move": "retain; magnitude alone is not a data error",
        },
    }
    (REPORTS / "investable_extreme_gap_cross_source_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
