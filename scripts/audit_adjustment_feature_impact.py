"""Measure input-feature deltas only; no labels, fitting or return selection."""

from pathlib import Path

import numpy as np
import pandas as pd

from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.research.baseline_v2_release import digest, read_json


def features(prices):
    returns = prices.pct_change(fill_method=None)
    return {
        "momentum_252_21": prices.shift(21) / prices.shift(252) - 1,
        "return_120": prices / prices.shift(120) - 1,
        "volatility_60": returns.rolling(60, min_periods=60).std(ddof=1) * np.sqrt(242),
        "drawdown_120": prices / prices.rolling(120, min_periods=120).max() - 1,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    pointer = root / "state/baseline-v2-release/adjustment-overlay.json"
    report = read_json(pointer)
    directory = Path(report["output_directory"])
    source = directory / "native-with-derived-factors.parquet"
    if report["artifact_hashes"][str(source)] != digest(source):
        raise ValueError("OVERLAY_SOURCE_CHANGED")
    frame = pd.read_parquet(source)
    before, after = frame.copy(), frame.copy()
    before["price"] = before.close * before.adj_factor
    after["price"] = after.close * after.derived_adj_factor
    panels = []
    for data in (before, after):
        panel = data.pivot(index="trade_date", columns="ts_code", values="price").sort_index()
        panels.append(panel.where(np.isfinite(panel) & panel.gt(0)))
    original, corrected = features(panels[0]), features(panels[1])
    results = []
    for name in original:
        a, b = original[name], corrected[name]
        comparable = a.notna() & b.notna()
        delta = (b - a).where(comparable)
        changed = delta.abs().gt(1e-10)
        values = delta.to_numpy()
        results.append(
            {
                "factor": name,
                "comparable_stock_days": int(comparable.to_numpy().sum()),
                "changed_stock_days": int(changed.to_numpy().sum()),
                "changed_securities": int(changed.any(axis=0).sum()),
                "maximum_absolute_feature_delta": float(np.nanmax(np.abs(values))),
            }
        )
        pd.DataFrame(
            {"trade_date": changed.index, "changed_stock_days": changed.sum(axis=1).to_numpy()}
        ).to_parquet(directory / (name + "-daily-impact.parquet"), index=False)
    output = {
        "status": "PARTIAL_OVERLAY_INPUT_IMPACT_NOT_MODEL_VALIDATION",
        "scope": "FULL_UNION_FEATURES_NOT_EXACT_TRAINING_SAMPLE_MEMBERSHIP",
        "verified_steps_removed": report["verified_steps_removed"],
        "factors": results,
        "labels_used": False,
        "model_refit": False,
        "source_sha256": digest(source),
        "script_sha256": digest(__file__),
        "production_changed": False,
    }
    write_diagnostic(directory / "feature-impact.json", output)
    write_diagnostic(root / "state/baseline-v2-release/adjustment-feature-impact.json", output)
    print(output)
