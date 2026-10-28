"""Independent saved-panel arithmetic, ranks and coverage check. No model fitting."""

import hashlib
import json
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from moneymore.challenger_diagnostics import write_diagnostic

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    base = root / "state/baseline-v2-validation/daily-inputs"
    report = json.loads((base / "daily-scores.json").read_text(encoding="utf-8"))
    directory = Path(report["output_directory"])
    path = directory / "daily-scores.parquet"
    if hashlib.sha256(path.read_bytes()).hexdigest() != report["panel_sha256"]:
        raise ValueError("SAVED_SCORE_PANEL_HASH_MISMATCH")
    frozen = json.loads((directory / "frozen-inputs.json").read_text(encoding="utf-8"))
    pool = Path(frozen["identity_audit"]["output_directory"]) / "identity-top1000-preview.parquet"
    if hashlib.sha256(pool.read_bytes()).hexdigest() != frozen["pool_sha256"]:
        raise ValueError("FROZEN_POOL_CHANGED")
    coverage = frozen["coverage_audit"]
    exclusions = Path(coverage["output_directory"]) / "sample-exclusions.parquet"
    if hashlib.sha256(exclusions.read_bytes()).hexdigest() != coverage["details_sha256"]:
        raise ValueError("FROZEN_COVERAGE_CHANGED")
    coverage_detail = pd.read_parquet(exclusions)
    expected = coverage_detail.loc[coverage_detail.reason.eq("ELIGIBLE"), ["date", "symbol"]]
    panel = pd.read_parquet(path)
    panel["date_key"] = pd.to_datetime(panel.date).dt.strftime("%Y%m%d")
    if panel.duplicated(["date_key", "symbol"]).any():
        raise ValueError("DUPLICATE_SCORE_IDENTITY_DATE")
    reconciliation = panel[["date_key", "symbol"]].merge(
        expected,
        left_on=["date_key", "symbol"],
        right_on=["date", "symbol"],
        how="outer",
        indicator=True,
    )
    if not reconciliation._merge.eq("both").all():
        raise ValueError("SAVED_SCORE_COVERAGE_MISMATCH")
    weights = frozen["weight_report"]
    names = list(frozen["policy"]["factors"])
    fixed = np.array([weights["fixed_weights"][n] for n in names], dtype=float)
    learned = np.array(
        [weights["diagnostic_coefficients_not_released"][n] for n in names], dtype=float
    )
    if not np.isfinite(learned).all() or np.abs(learned).sum() <= 1e-12:
        raise ValueError("INVALID_FROZEN_LEARNED_WEIGHTS")
    errors = {}
    for variant, coefficients in [("fixed", fixed / fixed.sum()), ("learned", learned)]:
        predicted = np.einsum("ij,j->i", panel[names].to_numpy(), coefficients)
        error = np.abs(predicted - panel["score_" + variant].to_numpy())
        if not np.isfinite(error).all() or error.max() > 1e-12:
            raise ValueError("SCORE_ARITHMETIC_MISMATCH:" + variant)
        errors[variant] = float(error.max())
        for _, daily in panel.groupby("date_key"):
            ordered = daily.sort_values(["score_" + variant, "symbol"], ascending=[False, True])
            if not np.array_equal(
                ordered["rank_" + variant].to_numpy(), np.arange(1, len(ordered) + 1)
            ):
                raise ValueError("SAVED_SCORE_RANK_MISMATCH:" + variant)
    reasons = coverage_detail.set_index(["date", "symbol"]).reason.to_dict()
    by_date = {d: frame for d, frame in panel.groupby("date_key")}
    dates = sorted(by_date)
    continuity = []
    for previous, current in pairwise(dates):
        eligible = set(by_date[current].symbol)
        for variant in ("fixed", "learned"):
            top = by_date[previous].loc[by_date[previous]["rank_" + variant].le(10), "symbol"]
            for symbol in sorted(set(top) - eligible):
                continuity.append(
                    {
                        "variant": variant,
                        "previous_signal": previous,
                        "current_signal": current,
                        "symbol": symbol,
                        "reason": reasons.get((current, symbol), "OUTSIDE_CURRENT_TOP1000"),
                    }
                )
    result = {
        "status": "SAVED_DAILY_RANKINGS_AUDITED_NOT_RELEASED",
        "score_run": str(directory),
        "days": panel.date_key.nunique(),
        "rows": len(panel),
        "maximum_score_error": errors,
        "coverage_and_deterministic_ranks": "PASS",
        "diagnostic_previous_top10_missing_next_day": len(continuity),
        "real_orders_enabled": False,
        "limitations": [
            "NOT_AN_INDEPENDENT_REBUILD_OF_RAW_FACTOR_PREPROCESSING",
            "NOT_DAILY_EXECUTION_VALIDATION",
            "CONTINUITY_CHECK_USES_DIAGNOSTIC_TOP10_NOT_ACTUAL_HOLDINGS",
        ],
    }
    destination = base / "score-audits" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    destination.mkdir(parents=True)
    pd.DataFrame(
        continuity, columns=["variant", "previous_signal", "current_signal", "symbol", "reason"]
    ).to_parquet(destination / "top10-input-continuity.parquet", index=False)
    result["output_directory"] = str(destination)
    write_diagnostic(destination / "report.json", result)
    write_diagnostic(base / "daily-score-audit.json", result)
    print(json.dumps(result))
