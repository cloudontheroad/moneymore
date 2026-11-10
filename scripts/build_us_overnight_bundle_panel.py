"""Assemble causally compatible feature modules for bundle Pareto experiments."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data/us_overnight_corrected/features"


def main() -> int:
    factor = pd.read_parquet(FEATURES / "historical_panel_factor_groups_lag1.parquet")
    option = pd.read_parquet(FEATURES / "historical_panel_61_plus_options.parquet")
    event = pd.read_parquet(FEATURES / "historical_panel_61_plus_event_v1.parquet")
    groups = json.loads((FEATURES / "factor_groups.json").read_text(encoding="utf-8"))
    option_columns = json.loads((FEATURES / "feature_columns_61_plus_options.json").read_text(encoding="utf-8"))
    option_columns = [column for column in option_columns if column not in groups["base"]]
    rating_columns = [
        "event_rating_direction", "event_rating_target_change",
        "event_rating_age_days", "event_rating_decay_20d",
    ]
    keys = ["timestamp", "symbol"]
    panel = factor.merge(option[keys + option_columns], on=keys, how="left", validate="one_to_one")
    panel = panel.merge(event[keys + rating_columns], on=keys, how="left", validate="one_to_one")
    target = FEATURES / "historical_panel_bundle_modules.parquet"
    panel.to_parquet(target, index=False)
    module_columns = {
        "day_night": groups["overnight_history"],
        "right_side": groups["right_side_trend"],
        "market": groups["market_regime"],
        "sector": groups["sector_regime"],
        "calendar": groups["calendar"],
        "options": option_columns,
        "ratings": rating_columns,
    }
    bundles = {
        "right_market": ["right_side", "market"],
        "right_ratings": ["right_side", "ratings"],
        "market_ratings": ["market", "ratings"],
        "right_market_ratings": ["right_side", "market", "ratings"],
        "daynight_right": ["day_night", "right_side"],
        "right_market_options": ["right_side", "market", "options"],
        "right_market_ratings_calendar": ["right_side", "market", "ratings", "calendar"],
        "right_market_daynight": ["right_side", "market", "day_night"],
        "right_market_sector": ["right_side", "market", "sector"],
        "right_market_calendar": ["right_side", "market", "calendar"],
        "right_market_ratings_options": ["right_side", "market", "ratings", "options"],
        "right_market_sector_ratings": ["right_side", "market", "sector", "ratings"],
        "right_market_options_calendar": ["right_side", "market", "options", "calendar"],
        "right_market_options_calendar_ratings": ["right_side", "market", "options", "calendar", "ratings"],
        "right_market_options_calendar_sector": ["right_side", "market", "options", "calendar", "sector"],
        "right_market_options_calendar_sector_ratings": ["right_side", "market", "options", "calendar", "sector", "ratings"],
    }
    for name, modules in bundles.items():
        columns = list(groups["base"])
        for module in modules:
            columns.extend(module_columns[module])
        (FEATURES / f"feature_columns_bundle_{name}.json").write_text(
            json.dumps(columns, indent=2), encoding="utf-8"
        )
    audit = {
        "panel": str(target), "rows": len(panel), "modules": {key: len(value) for key, value in module_columns.items()},
        "bundles": bundles,
        "acceptance_targets": ["top1_mean_net_bps", "top1_sharpe", "top10_mean_net_bps", "top10_sharpe"],
        "monotonic_ranking_required": False,
    }
    (FEATURES / "bundle_modules_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
