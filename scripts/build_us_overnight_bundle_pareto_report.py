"""Record bundle-aware Pareto search and the research-only champion."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "data/us_overnight_paper_only/models"
OUT = ROOT / "data/us_overnight_corrected/reports"
BUNDLES = [
    "right_market", "right_ratings", "market_ratings",
    "right_market_ratings", "right_market_options", "right_market_ratings_calendar",
    "right_market_daynight", "right_market_sector", "right_market_calendar",
    "right_market_ratings_options", "right_market_sector_ratings", "right_market_options_calendar",
    "right_market_options_calendar_ratings", "right_market_options_calendar_sector",
    "right_market_options_calendar_sector_ratings",
]
ACCEPTED = {"right_market": 1, "right_market_options_calendar": 2}


def metrics(prefix: str) -> dict:
    data = json.loads((MODELS / f"{prefix}_ranker.json").read_text(encoding="utf-8"))
    top1 = data["validation_rank_slot_metrics"][0]
    return {
        "top10_mean_net_bps": data["validation_top_n_mean_net"] * 10_000,
        "top10_win_rate": data["validation_top_n_win_rate"],
        "top10_sharpe": data["validation_top_n_sharpe"],
        "top10_max_drawdown": data["validation_top_n_max_drawdown"],
        "top1_mean_net_bps": top1["mean_net_bps"], "top1_win_rate": top1["win_rate"],
        "top1_sharpe": top1["sharpe"], "top1_max_drawdown": top1["max_drawdown"],
    }


def main() -> int:
    baseline = metrics("pareto_00_baseline_replay")
    rows = []
    for name in BUNDLES:
        rows.append({
            "bundle": name, "accepted_stage": ACCEPTED.get(name),
            "decision": "ACCEPT" if name in ACCEPTED else "REJECT",
            **metrics(f"bundle_{name}"),
        })
    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "bundle_pareto_experiments.csv", index=False)
    champion_name = "right_market_options_calendar"
    champion = metrics(f"bundle_{champion_name}")
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "objective": "joint improvement of Top1 and aggregate Top10; rank monotonicity retired",
        "acceptance": "modules may be accepted as a package when the package improves Top1 and Top10 mean net return and Sharpe",
        "bundle_experiments": len(frame), "accepted_packages": len(ACCEPTED),
        "accepted_chain": ["right_market", champion_name],
        "baseline": baseline, "research_champion": {"bundle": champion_name, **champion},
        "improvement": {key: champion[key] - baseline[key] for key in champion},
        "trade_status": "BLOCKED_TOP10_NET_RETURN_STILL_NEGATIVE",
        "model_json": str(MODELS / f"bundle_{champion_name}_ranker.json"),
        "model_joblib": str(MODELS / f"bundle_{champion_name}_ranker.joblib"),
    }
    (OUT / "bundle_pareto_final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (MODELS / "research_champion_bundle.json").write_text(json.dumps(report["research_champion"], indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
