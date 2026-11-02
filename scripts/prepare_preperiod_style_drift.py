"""Pre-evaluation point-available universe feature drift, no fitted risk budget."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from moneymore.challenger_diagnostics import trading_dates
from moneymore.data.store import ParquetStore
from moneymore.data.fundamental_research import research_financial_frame
from moneymore.research.style_replay_inputs import daily_risk_snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("immutable calibration inputs")
    root = Path(__file__).resolve().parents[1]; store = ParquetStore(root/"data")
    start, end, evaluation_start = "20260902", "20260930", "20261012"
    universe_path = store.curated/"strategy_universe.parquet"
    universe = store.read("strategy_universe")
    selected = universe.loc[universe.effective_date.astype(str).eq("20260901")].copy()
    if len(selected) != 1000 or selected.symbol.duplicated().any() or selected.created_at.nunique() != 1:
        raise ValueError("HISTORICAL_UNIVERSE_NOT_UNAMBIGUOUS")
    captured = pd.to_datetime(selected.created_at.iloc[0], utc=True)
    if captured >= pd.Timestamp(start+" 09:30", tz="Asia/Shanghai").tz_convert("UTC") or end >= evaluation_start:
        raise ValueError("CALIBRATION_OR_UNIVERSE_USES_FUTURE")
    symbols = selected.symbol.tolist(); calendar = trading_dates(store, end)
    dates = [d for d in calendar if start <= d <= end]
    risk_start = calendar[calendar.index(start)-120]
    filters = [("trade_date", ">=", risk_start), ("trade_date", "<=", end), ("ts_code", "in", symbols)]
    daily = store.read("daily", columns=["ts_code", "trade_date", "open", "close", "amount"], filters=filters)
    daily = daily.merge(store.read("adj_factor", columns=["ts_code", "trade_date", "adj_factor"], filters=filters), on=["trade_date", "ts_code"], how="left", validate="one_to_one")
    daily["total_price"] = daily.close*daily.adj_factor
    basic = store.read("daily_basic", filters=[("trade_date", ">=", start), ("trade_date", "<=", end), ("ts_code", "in", symbols)])
    finances = research_financial_frame(store); membership = store.read("baseline_v2_industry_membership")
    args.output.mkdir(parents=True)
    selected.to_parquet(args.output/"historical-universe.parquet", index=False)
    previous = None; rows = []; coverage = []
    for date in dates:
        # Constant entries identify the historical pool only; never presented
        # as model forecasts. Covariance fitting is unnecessary here.
        raw, controls, _, _, _, wb, _, missing = daily_risk_snapshot(date, pd.Series(0., index=symbols), {}, calendar,
            daily, basic, finances, membership, estimate_covariance=False)
        centered = controls-(wb @ controls)
        centered.to_parquet(args.output/f"centered-controls-{date}.parquet")
        raw.to_parquet(args.output/f"observations-{date}.parquet")
        coverage.append({"date":date, "eligible":len(controls), "unknown_counts":missing["missing_counts"]})
        if previous is not None:
            prior_date, prior = previous
            common_assets = prior.index.intersection(centered.index)
            common_factors = prior.columns.intersection(centered.columns)
            change = centered.loc[common_assets, common_factors]-prior.loc[common_assets, common_factors]
            for factor in common_factors:
                values = change[factor]
                if not np.isfinite(values).all():
                    raise ValueError("NONFINITE_COEFFICIENT_DRIFT")
                rows.append({"from_date":prior_date, "to_date":date, "factor":factor, "observations":len(values),
                    "median_absolute_asset_change":float(abs(values).median()), "p95_absolute_asset_change":float(abs(values).quantile(.95)),
                    "maximum_absolute_asset_change":float(abs(values).max()),
                    "eligible_added":len(centered.index.difference(prior.index)), "eligible_removed":len(prior.index.difference(centered.index))})
        previous = (date, centered)
        print(json.dumps({"date":date,"eligible":len(controls)}), flush=True)
    pd.DataFrame(rows).to_parquet(args.output/"prior-period-factor-drift.parquet", index=False)
    report = {"status":"PREPERIOD_INPUTS_PREPARED_NOT_BUFFER_CALIBRATED", "orders_enabled":False, "start":start,"end":end,
        "evaluation_start":evaluation_start, "sessions":len(dates), "transitions":max(0,len(dates)-1),
        "historical_universe_effective_date":"20260901", "historical_universe_captured_at":str(captured),
        "source_universe_file_sha256":hashlib.sha256(universe_path.read_bytes()).hexdigest(), "coverage":coverage,
        "selected_historical_universe_sha256":hashlib.sha256((args.output/"historical-universe.parquet").read_bytes()).hexdigest(),
        "implementation_hashes":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(),
            root/"src/moneymore/research/style_replay_inputs.py",root/"src/moneymore/research/style_price_features.py",root/"src/moneymore/research/style_missing_controls.py"]},
        "no_model_training":True, "no_buffer_selected":True,
        "limitations":["September coefficient drift, not portfolio next-day error or 95% coverage guarantee",
            "prior effective-date industry/financial visibility but supplier original vintages not proven",
            "current November pool not used; Sep1 frozen pool does not track later cap-rank reconstitution",
            "missing mandatory risk fields excluded and counted; no price forward-fill",
            "no October evaluation returns or scores used to choose margins"]}
    (args.output/"results.json").write_text(json.dumps(report, ensure_ascii=False,indent=2,allow_nan=False), encoding="utf-8")
    print(json.dumps({"status":report["status"],"sessions":len(dates),"transitions":len(dates)-1}))


if __name__ == "__main__":
    main()
