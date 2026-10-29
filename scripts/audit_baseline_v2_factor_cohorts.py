"""Read-only cohort/reversal hypotheses; never normalizes adjustment factors."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.research.baseline_v2_release import digest, read_json

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    pointer = read_json(root / "state/baseline-v2-release/latest.json")
    directory = Path(pointer["directory"])
    # This independent diagnostic may also use the last complete source audit
    # while a fresh release bundle is still being built. Source hashes identify it.
    if not (directory / "full-union-action-alignment.parquet").exists():
        directories = sorted(
            (root / "state/baseline-v2-release/runs").glob("*/full-union-action-alignment.parquet")
        )
        directory = directories[-1].parent
    source = directory / "full-union-native-prices.parquet"
    alignment = directory / "full-union-action-alignment.parquet"
    native = pd.read_parquet(source, columns=["ts_code", "trade_date", "adj_factor"]).sort_values(
        ["ts_code", "trade_date"]
    )
    native["ratio"] = native.groupby("ts_code").adj_factor.pct_change(fill_method=None) + 1
    native["next_ratio"] = native.groupby("ts_code").ratio.shift(-1)
    native["next_observed_bar_date"] = native.groupby("ts_code").trade_date.shift(-1)
    issues = pd.read_parquet(alignment)
    issues = issues.loc[
        issues.action_alignment.eq("NO_IMPLEMENTED_EVENT_IN_CACHE_UNRESOLVED")
        & np.isclose(issues.pre_close, issues.previous_close, rtol=0, atol=1e-8)
    ]
    cohort = issues.merge(
        native[["ts_code", "trade_date", "ratio", "next_ratio", "next_observed_bar_date"]],
        on=["ts_code", "trade_date"],
        validate="one_to_one",
    )
    cohort["returns_to_prior_factor_next_observed_bar"] = np.isclose(
        cohort.ratio * cohort.next_ratio, 1, rtol=0, atol=1e-10
    )
    rows = []
    for date, group in cohort.groupby("trade_date"):
        rows.append(
            {
                "date": str(date),
                "stocks": len(group),
                "immediate_reversal_candidates": int(
                    group.returns_to_prior_factor_next_observed_bar.sum()
                ),
                "median_step_bps": float((group.ratio - 1).median() * 10000),
                "minimum_step_bps": float((group.ratio - 1).min() * 10000),
                "maximum_step_bps": float((group.ratio - 1).max() * 10000),
            }
        )
    output = root / "state/baseline-v2-release/factor-cohort-audits" / digest(alignment)[:20]
    output.mkdir(parents=True, exist_ok=True)
    detail = output / "cohorts.parquet"
    cohort.to_parquet(detail, index=False)
    result = {
        "status": "DIAGNOSTIC_COHORTS_NOT_INDEPENDENT_PROOF",
        "pairs": len(cohort),
        "cohorts": sorted(rows, key=lambda x: (-x["stocks"], x["date"])),
        "immediate_reversal_candidates": int(
            cohort.returns_to_prior_factor_next_observed_bar.sum()
        ),
        "artifact_hashes": {str(p): digest(p) for p in (source, alignment, detail)},
        "limitations": [
            "MASS_DATE_CLUSTER_NOT_ROOT_CAUSE_PROOF",
            "NEXT_OBSERVED_BAR_NOT_GUARANTEED_NEXT_EXCHANGE_SESSION",
        ],
        "source_normalized": False,
        "release_authorized": False,
    }
    write_diagnostic(root / "state/baseline-v2-release/factor-cohort-audit.json", result)
    print(json.dumps(result))
