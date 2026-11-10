"""Add point-in-time cross-sectional residual targets without changing P&L labels."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/us_overnight_corrected/features/historical_panel_61_lag1.parquet"
TARGET = ROOT / "data/us_overnight_corrected/features/historical_panel_61_residual_targets.parquet"


def main() -> int:
    panel = pd.read_parquet(SOURCE)
    daily = panel.groupby("timestamp")["label_next_open_net"]
    panel["label_cross_sectional_residual"] = (
        panel["label_next_open_net"] - daily.transform("median")
    )
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(TARGET, index=False)
    print({"target": str(TARGET), "rows": len(panel), "sessions": panel["timestamp"].nunique()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
