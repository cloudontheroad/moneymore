"""Cross-sectional factor efficacy audit for the corrected open-to-open label."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/us_overnight_corrected/features/historical_panel_bundle_investable_120d_20m_5.parquet"
OUT_CSV = ROOT / "data/us_overnight_corrected/reports/corrected_open2open_factor_ic.csv"
OUT_JSON = ROOT / "data/us_overnight_corrected/reports/corrected_open2open_factor_ic.json"
LABEL = "label_next_open_net"

EXCLUDE = {
    "timestamp", "symbol", "feature_asof", LABEL, "open", "high", "low", "close",
    "volume", "next_open", "next_dividend", "listing_sessions", "eligible",
}


def factor_family(name: str) -> str:
    if name.startswith(("market_", "macro_", "vix", "tnx")):
        return "market_macro"
    if name.startswith(("option_", "iv_")):
        return "options"
    if name.startswith(("rating_", "analyst_")):
        return "analyst"
    if any(token in name for token in ("volume", "dollar", "turnover", "liquidity")):
        return "volume_liquidity"
    if any(token in name for token in ("rsi", "macd", "boll", "kdj", "atr", "adx", "stoch")):
        return "technical"
    if any(token in name for token in ("ret", "momentum", "reversal", "trend", "breakout", "gap")):
        return "price_trend"
    if any(token in name for token in ("event", "earn", "calendar", "distance", "news")):
        return "event_calendar"
    return "other"


def main() -> int:
    panel = pd.read_parquet(PANEL)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    numeric = panel.select_dtypes(include=[np.number, "bool"]).columns
    features = [column for column in numeric if column not in EXCLUDE and not column.startswith("label")]
    dates = sorted(panel.timestamp.unique())
    midpoint = dates[len(dates) // 2]
    rows = []
    for feature in features:
        subset = panel[["timestamp", feature, LABEL]].replace([np.inf, -np.inf], np.nan).dropna()
        if subset.empty or subset[feature].nunique() < 2:
            continue
        subset = subset.assign(
            ranked_x=subset.groupby("timestamp")[feature].rank(pct=True),
            ranked_y=subset.groupby("timestamp")[LABEL].rank(pct=True),
        )
        ic = subset.groupby("timestamp", sort=False).apply(
            lambda group: group.ranked_x.corr(group.ranked_y),
            include_groups=False,
        ).dropna()
        if len(ic) < 60:
            continue
        first = ic[ic.index < midpoint]
        second = ic[ic.index >= midpoint]
        std = ic.std(ddof=1)
        rows.append({
            "feature": feature,
            "family": factor_family(feature),
            "available_rows": int(len(subset)),
            "sessions": int(len(ic)),
            "mean_ic": float(ic.mean()),
            "ic_ir": float(ic.mean() / std * np.sqrt(252)) if std > 0 else 0.0,
            "positive_session_rate": float((ic > 0).mean()),
            "first_half_ic": float(first.mean()),
            "second_half_ic": float(second.mean()),
            "same_sign_halves": bool(first.mean() * second.mean() > 0),
        })
    result = pd.DataFrame(rows)
    result["abs_mean_ic"] = result.mean_ic.abs()
    result = result.sort_values(["same_sign_halves", "abs_mean_ic"], ascending=False)
    result.to_csv(OUT_CSV, index=False)
    stable = result[result.same_sign_halves]
    family = stable.groupby("family").agg(
        tested=("feature", "count"),
        best_abs_ic=("abs_mean_ic", "max"),
        median_abs_ic=("abs_mean_ic", "median"),
        features_ic_ge_005=("abs_mean_ic", lambda value: int((value >= 0.005).sum())),
    ).sort_values("best_abs_ic", ascending=False)
    report = {
        "label": "open(t) to open(t+1), net 20 bps",
        "features_tested": int(len(result)),
        "stable_sign_features": int(result.same_sign_halves.sum()),
        "stable_abs_ic_ge_005": int((stable.abs_mean_ic >= 0.005).sum()),
        "stable_abs_ic_ge_01": int((stable.abs_mean_ic >= 0.01).sum()),
        "top_stable_features": stable.head(25).to_dict("records"),
        "family_summary": family.reset_index().to_dict("records"),
        "note": "Diagnostic only; no feature is adopted from this full-sample audit.",
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "features_tested", "stable_sign_features", "stable_abs_ic_ge_005",
        "stable_abs_ic_ge_01", "family_summary")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
