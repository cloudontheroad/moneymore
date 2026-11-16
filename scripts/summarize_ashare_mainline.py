"""Close N2 and freeze causal monthly calibration inputs for N3."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from moneymore.research.ashare_mainline import assert_keys, prediction_diagnostics, sha256

ROOT = Path(__file__).resolve().parents[1]


def block_mean(values, block=20, samples=1000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"mean": None, "ci95": [None, None], "standard_error": None}
    rng = np.random.default_rng(42)
    starts = rng.integers(len(values), size=(samples, (len(values) + block - 1) // block))
    indices = (starts[..., None] + np.arange(block)) % len(values)
    means = values[indices.reshape(samples, -1)[:, : len(values)]].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "standard_error": float(means.std(ddof=1)),
    }


def causal_calibration(predictions, cutoff):
    f = predictions.loc[
        predictions.label_end_20.notna()
        & predictions.label_end_20.lt(cutoff)
        & predictions.date.lt(cutoff)
    ].copy()
    rows = []
    for day, g in f.groupby("date"):
        z = (g.score - g.score.mean()) / g.score.std(ddof=0)
        z = z.clip(-3, 3)
        good = np.isfinite(z) & np.isfinite(g.label_20)
        z = z.loc[good]
        y = g.loc[good, "label_20"]
        if len(z) < 100 or float(z @ z) < 1e-10:
            continue
        y = y - y.mean()
        rows.append(
            {
                "date": day,
                "beta": float(z @ y / (z @ z)),
                "last_label_end": str(g.loc[good, "label_end_20"].max()),
                "assets": len(z),
            }
        )
    daily = pd.DataFrame(rows)
    if len(daily) < 60:
        raise ValueError("INSUFFICIENT_MATURE_OOF_CALIBRATION")
    estimate = block_mean(daily.beta)
    return {
        "cutoff_exclusive": cutoff,
        "horizon": 20,
        "oof_days": len(daily),
        "last_signal": daily.date.max(),
        "last_label_end": daily.last_label_end.max(),
        "beta": estimate["mean"],
        "standard_error_block20": estimate["standard_error"],
        "beta_conservative": max(0.0, estimate["mean"] - estimate["standard_error"]),
        "score_transform": "current_cross_section_zscore_clip3",
        "unit": "20_session_relative_return_decimal",
        "automatic_sign_flip": False,
    }, daily


def main(source, output):
    if output.exists():
        raise FileExistsError("IMMUTABLE_N2_SUMMARY")
    manifest = json.loads((source / "results.json").read_bytes())
    config = json.loads((source / "protocol.json").read_bytes())
    expected = {(fold, case) for fold in config["folds"] for case in config["cases"]}
    actual = {(r["fold"], r["case"]) for r in manifest["results"]}
    if (
        not manifest["full_registered_matrix_complete"]
        or expected != actual
        or len(manifest["results"]) != len(expected)
    ):
        raise ValueError("INCOMPLETE_OR_DUPLICATED_MATRIX")
    output.mkdir(parents=True)
    records, summaries, calibrations = [], [], []
    for record in manifest["results"]:
        path = source / record["fold"] / record["case"]
        for name, digest in record["hashes"].items():
            if sha256(path / name) != digest:
                raise ValueError("MATRIX_ARTIFACT_CHANGED:" + str(path / name))
        contract = json.loads((path.parent / "time-contract.json").read_bytes())
        if (
            contract["train"]["last_label_end"] >= contract["valid"]["first"]
            or contract["valid"]["last_label_end"] >= contract["test"]["first"]
        ):
            raise ValueError("CROSS_BOUNDARY_LABEL")
        p = pd.read_parquet(path / "predictions.parquet")
        p["case"] = record["case"]
        p["fold"] = record["fold"]
        records.append(p)
    combined = pd.concat(records, ignore_index=True)
    assert_keys(combined, ("case", "date", "symbol"))
    combined.to_parquet(output / "predictions.parquet", index=False)
    diagnostics = []
    for case, p in combined.groupby("case"):
        for year in ("2025", "2026"):
            end = "20251231" if year == "2025" else "20260930"
            daily = prediction_diagnostics(p.loc[p.date.str.startswith(year)], end)
            daily["case"], daily["year"] = case, year
            diagnostics.append(daily)
            estimate = block_mean(daily.rank_ic)
            summaries.append(
                {
                    "case": case,
                    "year": year,
                    "mean_rank_ic": estimate["mean"],
                    "rank_ic_block20_ci95": estimate["ci95"],
                    "mature_days": int(daily.rank_ic.notna().sum()),
                    "top50_growth_board": float(daily.top50_chinext_star_fraction.mean()),
                    "top50_largest_industry": float(daily.top50_largest_industry_fraction.mean()),
                }
            )
        for month in pd.date_range("2026-01-01", "2026-09-01", freq="MS"):
            cal, _ = causal_calibration(p, month.strftime("%Y%m%d"))
            calibrations.append({"case": case, "month": month.strftime("%Y%m"), **cal})
    pd.concat(diagnostics, ignore_index=True).to_parquet(
        output / "daily-diagnostics.parquet", index=False
    )
    (output / "calibration.json").write_text(json.dumps(calibrations, indent=2), encoding="utf-8")
    result = {
        "status": "N2_COMPLETE_N3_COSTED_REPLAY_PENDING",
        "verified_cases": len(expected),
        "prediction_rows": len(combined),
        "summaries": summaries,
        "calibration_count": len(calibrations),
        "cross_quarter_mature_labels_included": True,
        "source_results_sha256": sha256(source / "results.json"),
        "hashes": {
            name: sha256(output / name)
            for name in ("predictions.parquet", "daily-diagnostics.parquet", "calibration.json")
        },
        "net_performance_measured": False,
        "production_modified": False,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=ROOT / "state/ashare-mainline-20261112/registered-matrix-v1"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "state/ashare-mainline-20261112/n2-summary-v1"
    )
    args = parser.parse_args()
    main(args.source, args.output)
