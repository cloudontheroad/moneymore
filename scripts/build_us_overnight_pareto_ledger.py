"""Create the reproducible experiment ledger for the corrected US panel."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "data/us_overnight_paper_only/models"
OUT = ROOT / "data/us_overnight_corrected/reports"
EXPERIMENTS = [
    (0, "reproducible_corrected_baseline", "pareto_00_baseline_replay", "BASELINE"),
    (1, "day_night_decomposition", "pareto_01_day_night", "REJECT"),
    (2, "cross_sectional_residual_target", "pareto_02_residual_target", "REJECT"),
    (4, "right_side_trend", "pareto_04_right_side_trend", "REJECT"),
    (5, "market_regime", "pareto_05_market_regime", "REJECT"),
    (6, "sector_etf_regime", "pareto_06_sector_regime", "REJECT"),
    (7, "calendar", "pareto_07_calendar", "REJECT"),
    (8, "option_information", "pareto_08_options", "REJECT"),
    (9, "rating_change_proxy", "experiment_event_ratings_v1", "REJECT"),
    (10, "forward_earnings_proxy", "experiment_forward_earnings_v1", "REJECT_INVALID_HISTORICAL_VINTAGE"),
]


def main() -> int:
    rows = []
    for order, module, prefix, decision in EXPERIMENTS:
        payload = json.loads((MODELS / f"{prefix}_ranker.json").read_text(encoding="utf-8"))
        top1 = payload["validation_rank_slot_metrics"][0]
        rows.append({
            "order": order, "module": module, "model_prefix": prefix, "decision": decision,
            "top10_mean_net_bps": payload["validation_top_n_mean_net"] * 10_000,
            "top10_win_rate": payload["validation_top_n_win_rate"],
            "top10_sharpe": payload["validation_top_n_sharpe"],
            "top10_max_drawdown": payload["validation_top_n_max_drawdown"],
            "top1_mean_net_bps": top1["mean_net_bps"], "top1_win_rate": top1["win_rate"],
            "top1_sharpe": top1["sharpe"], "top1_max_drawdown": top1["max_drawdown"],
            "monotonic_violations": payload["validation_monotonic_violations"],
        })
    ledger = pd.DataFrame(rows).sort_values("order")
    OUT.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(OUT / "pareto_experiment_ledger.csv", index=False)
    baseline = ledger.iloc[0].to_dict()
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "panel_contract": "corrected 2026-11-06 panel only; old 2026-11-05 champion invalidated",
        "acceptance": "strict Pareto: improve at least one core metric and worsen none, including Top1",
        "trained_rounds": int((ledger.decision != "BASELINE").sum()),
        "accepted_rounds": int(ledger.decision.str.startswith("ACCEPT").sum()),
        "final_baseline": baseline,
        "confidence_gate": json.loads((MODELS / "pareto_03_confidence_gate.json").read_text()),
        "not_backtest_eligible": {
            "analyst_estimate_revisions": "no historical daily consensus revision snapshots",
            "social_sentiment": "immutable snapshots start now; no complete timestamped history",
            "industry_event_diffusion": "no point-in-time historical company relationship graph",
        },
        "data_built": {
            "option_symbols": 975, "option_panel_symbols": 974,
            "social_snapshot_contract": "configs/us_social_expectation_websearch.yaml",
        },
    }
    (OUT / "pareto_final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
