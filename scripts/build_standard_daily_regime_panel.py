"""Merge screened standard daily factors with causal market/sector regime state."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STANDARD = ROOT / "data/us_overnight_corrected/features/standard_daily/panel.parquet"
REGIME = ROOT / "data/us_overnight_corrected/features/historical_panel_factor_groups_lag1.parquet"
GROUPS = ROOT / "data/us_overnight_corrected/features/factor_groups.json"
OUT = ROOT / "data/us_overnight_corrected/features/standard_daily_regime"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    groups = json.loads(GROUPS.read_text(encoding="utf-8"))
    regime_columns = [*groups["market_regime"], *groups["sector_regime"]]
    standard = pd.read_parquet(STANDARD)
    regime = pd.read_parquet(REGIME, columns=["timestamp", "symbol", *regime_columns])
    panel = standard.merge(regime, on=["timestamp", "symbol"], how="left")
    selected = json.loads(
        (STANDARD.parent / "selected_features.json").read_text(encoding="utf-8")
    )
    (OUT / "standard26_market.json").write_text(
        json.dumps([*selected, *groups["market_regime"]], indent=2), encoding="utf-8"
    )
    (OUT / "standard26_sector.json").write_text(
        json.dumps([*selected, *groups["sector_regime"]], indent=2), encoding="utf-8"
    )
    (OUT / "standard26_all_regime.json").write_text(
        json.dumps([*selected, *regime_columns], indent=2), encoding="utf-8"
    )
    panel.to_parquet(OUT / "panel.parquet", index=False)
    print(json.dumps({"rows": len(panel), "standard": len(selected), "regime": len(regime_columns)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
