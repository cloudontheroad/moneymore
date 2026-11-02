"""Measurement-only scale ablation. No trading or new buffer selection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from moneymore.data.store import ParquetStore
from moneymore.research.frozen_risk_scale import fit_scale, apply_scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError("immutable scale ablation")
    root=Path(__file__).resolve().parents[1]
    calibration=root/"state/style-balance/preperiod-drift-202609-b"
    files=sorted(calibration.glob("observations-*.parquet"))
    frames={p.stem.split("-")[-1]:pd.read_parquet(p) for p in files}
    artifact=fit_scale(frames)
    base=root/"state/style-balance/expanded-style-10bps-20261102-b"
    continued=root/"state/style-balance/continued-style-20261027-b"
    report=json.loads((continued/"results.json").read_text(encoding="utf-8"))
    recorded_exposures={e["date"]:e["actual_style"]["active_controls"] for e in report["events"]
        if e.get("actual_style",{}).get("active_controls")}
    dates=sorted(report["reconstructed_holdings"])
    symbols=sorted({s for h in report["reconstructed_holdings"].values() for s in h})
    close=ParquetStore(root/"data").read("daily",columns=["trade_date","ts_code","close"],filters=[("trade_date",">=",dates[0]),("trade_date","<=",dates[-1]),("ts_code","in",symbols)]).set_index(["trade_date","ts_code"]).close
    args.output.mkdir(parents=True)
    (args.output/"frozen-scale.json").write_text(json.dumps(artifact,indent=2,allow_nan=False),encoding="utf-8")
    rows=[]; input_paths=list(files)+[continued/"results.json"]
    for date in dates:
        holdings=report["reconstructed_holdings"][date]
        if not holdings:
            continue
        directory=continued if date >= "20261026" else base
        raw_path=directory/f"observations-{date}.parquet"
        original_path=directory/f"controls-{date}.parquet"
        raw=pd.read_parquet(raw_path)
        daily,fixed,ratio=apply_scale(raw,artifact)
        original=pd.read_parquet(original_path)
        pd.testing.assert_frame_equal(daily,original,check_exact=True)
        # Reconstruction uses each day's observed raw caps on the SAME risk
        # pool. Existing positions outside it cannot silently acquire 0 risk.
        if set(holdings)-set(raw.index):
            raise ValueError("HELD_ASSET_NOT_IN_RISK_POOL")
        benchmark=np.exp(raw.log_size); benchmark/=benchmark.sum()
        value=pd.Series(holdings)*close.loc[date].reindex(holdings)
        if not np.isfinite(value).all():
            raise ValueError("MISSING_HELD_CLOSE")
        active=(value/value.sum()).reindex(raw.index,fill_value=0.)-benchmark
        a=active @ daily; b=active @ fixed
        if abs(a-pd.Series(recorded_exposures[date]).reindex(a.index)).max()>1e-8:
            raise ValueError("RECONSTRUCTED_SOURCE_EXPOSURE_MISMATCH")
        if abs(b-a*ratio).max()>1e-10:
            raise ValueError("MEASUREMENT_SCALE_IDENTITY_FAILED")
        for factor in daily.columns:
            limit=.01 if factor.startswith("unknown_") else .03 if factor.startswith(("industry_","board_")) else .1
            # Tolerance transforms too; equivalence is in physical raw units.
            original_alert=abs(a[factor])>limit+1e-6
            converted_alert=abs(b[factor])>(limit+1e-6)*ratio[factor]
            if original_alert != converted_alert:
                raise ValueError("RAW_RISK_REGION_WAS_CHANGED")
            rows.append({"trade_date":date,"factor":factor,"daily_scale_exposure":float(a[factor]),"frozen_scale_exposure":float(b[factor]),
                "ratio":float(ratio[factor]),"daily_limit":limit,"same_raw_budget_frozen_limit":float(limit*ratio[factor]),
                "daily_alert":bool(original_alert),"frozen_same_number_alert":bool(abs(b[factor])>limit+1e-6),
                "frozen_same_raw_budget_alert":bool(converted_alert)})
        input_paths.extend([raw_path,original_path])
    table=pd.DataFrame(rows)
    table.to_parquet(args.output/"scale-comparison.parquet",index=False)
    counts={name:{"factor_alerts":int(table[name].sum()),"days":int(table.loc[table[name],"trade_date"].nunique())}
        for name in ["daily_alert","frozen_same_number_alert","frozen_same_raw_budget_alert"]}
    summary={"status":"MEASUREMENT_ABLATION_COMPLETED_NOT_BUFFER_OR_STRATEGY_APPROVAL","orders_enabled":False,
        "calibration_start":min(frames),"calibration_end":max(frames),"evaluation_start":min(dates),"evaluation_end":max(dates),
        "counts":counts,"same_physical_budget_alerts_identical":True,"holdings_and_returns_unchanged":True,
        "input_hashes":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths},
        "limitations":["same numerical limit with new scale changes the physical risk budget; fewer alerts are not improvement evidence",
            "daily missing-value estimates and all category/unknown controls intentionally unchanged to isolate scale",
            "sample period only 20 sessions; no 95% portfolio guarantee","no buffer or optimizer/account update","revised historical supplier fundamentals, not original-vintage proof"]}
    (args.output/"results.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"counts":counts,"same_physical_budget_alerts_identical":True}))


if __name__=="__main__":
    main()
