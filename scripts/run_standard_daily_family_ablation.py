"""Ablate mature daily-factor families with one fixed LightGBM protocol."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/Scripts/python.exe"
TRAINER = ROOT / "scripts/train_us_overnight_lightgbm.py"
FEATURE_ROOT = ROOT / "data/us_overnight_corrected/features/standard_daily"
PANEL = FEATURE_ROOT / "panel.parquet"
MODELS = ROOT / "data/us_overnight_corrected/models/standard_daily_families"
STATE = ROOT / "state/us-overnight-standard-daily-families"

GROUPS = {
    "trend": ["logret_60", "trix15", "macd_signal_norm", "ppo", "macd_norm",
              "trend_d2_20", "aroon_up_25", "adx_d1", "macd_hist_norm",
              "logret_5", "trend_d1_5", "logret_20", "trend_d1_20", "aroon_down_25"],
    "volatility_channel": ["boll_width_60", "boll_width_10", "boll_pos_60",
                           "boll_width_20", "boll_width_d1_20", "keltner_pos_20"],
    "oscillator": ["rsi_28", "rsi_14"],
    "candle": ["lower_shadow", "upper_shadow", "candle_body"],
    "volume_price": ["cmf_20"],
}


def main() -> int:
    MODELS.mkdir(parents=True, exist_ok=True)
    STATE.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, features in GROUPS.items():
        feature_file = STATE / f"{name}.json"
        feature_file.write_text(json.dumps(features, indent=2), encoding="utf-8")
        command = [
            str(PYTHON), str(TRAINER), "--panel-path", str(PANEL), "--model-dir", str(MODELS),
            "--feature-columns-json", str(feature_file), "--output-prefix", name,
            "--num-leaves", "15", "--min-child-samples", "400", "--colsample-bytree", "0.75",
            "--reg-lambda", "3", "--learning-rate", "0.025", "--n-estimators", "350",
            "--huber-alpha", "0.9", "--weight-half-life", "126",
            "--rolling-train-sessions", "504", "--validation-sessions", "126",
            "--retrain-every", "21", "--min-child-weight", "50", "--extra-trees",
            "--max-bin", "63",
        ]
        subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        payload = json.loads((MODELS / f"{name}_ranker.json").read_text(encoding="utf-8"))
        top1 = payload["validation_rank_slot_metrics"][0]
        rows.append({
            "family": name, "feature_count": len(features),
            "top1_mean_bps": top1["mean_net_bps"], "top1_win_rate": top1["win_rate"],
            "top1_sharpe": top1["sharpe"], "top1_cumulative": top1["cumulative_return"],
            "top1_drawdown": top1["max_drawdown"],
            "top10_mean_bps": payload["validation_top_n_mean_net"] * 10_000,
            "top10_win_rate": payload["validation_top_n_win_rate"],
            "top10_sharpe": payload["validation_top_n_sharpe"],
            "top10_drawdown": payload["validation_top_n_max_drawdown"],
            "violations": payload["validation_monotonic_violations"],
        })
        print(json.dumps(rows[-1]), flush=True)
    (STATE / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
