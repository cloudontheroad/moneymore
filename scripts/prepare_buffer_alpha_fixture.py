"""Earliest archived-alpha engineering pair, explicitly NOT September holdout."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import yaml
from qlib.model.riskmodel import StructuredCovEstimator
from moneymore.data.store import ParquetStore
from moneymore.challenger_diagnostics import trading_dates
from moneymore.research.frozen_risk_scale import CORE
from moneymore.research.style_missing_controls import uncertain_controls
from moneymore.research.portfolio_error_envelope import robust_bounds
from moneymore.research.style_replay_inputs import optimize_daily,prospective_capture


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError("immutable engineering inputs")
    root=Path(__file__).resolve().parents[1];store=ParquetStore(root/"data")
    source=root/"state/style-balance/expanded-style-10bps-20261102-b"
    signal,date="20261012","20261013"
    archive=root/f"state/qlib-candidate-shadow/20261010/{signal}.json";record=json.loads(archive.read_text(encoding="utf-8"))
    if not prospective_capture(signal,record.get("created_at"),date):raise ValueError("NONPROSPECTIVE_ALPHA")
    raw=pd.read_parquet(source/f"observations-{signal}.parquet");controls=uncertain_controls(raw,CORE)[0]
    pd.testing.assert_frame_equal(controls,pd.read_parquet(source/f"controls-{signal}.parquet"),check_exact=True)
    wb=np.exp(raw.log_size);wb/=wb.sum()
    limits=pd.Series({f:.01 if f.startswith("unknown_") else .03 if f.startswith(("industry_","board_")) else .1 for f in controls.columns})
    calendar=trading_dates(store,date);start=calendar[calendar.index(signal)-120]
    filters=[("trade_date",">=",start),("trade_date","<=",date),("ts_code","in",list(raw.index))]
    bars=store.read("daily",columns=["ts_code","trade_date","open","close"],filters=filters)
    bars=bars.merge(store.read("adj_factor",columns=["ts_code","trade_date","adj_factor"],filters=filters),on=["ts_code","trade_date"],how="left",validate="one_to_one")
    bars["total_price"]=bars.close*bars.adj_factor
    returns=bars.pivot(index="trade_date",columns="ts_code",values="total_price").reindex(index=[d for d in calendar if start<=d<=signal],columns=raw.index).pct_change(fill_method=None).iloc[1:]
    if not np.isfinite(returns.to_numpy()).all():raise ValueError("RISK_PRICE_GAP")
    np.random.seed(17);F,cov,specific=StructuredCovEstimator(num_factors=10,factor_model="pca",scale_return=False).predict(returns,is_price=False,return_decomposed_components=True)
    scores=pd.DataFrame(record["scores"]).set_index("instrument").score
    base=optimize_daily(scores,raw,controls,F,cov,specific,wb,limits,pd.Series(dtype=float))
    recorded=pd.read_parquet(source/f"weights-{signal}.parquet").target
    difference=float(abs(base-recorded).max())
    if difference>1e-6:raise ValueError("ARCHIVED_BASE_TARGET_NOT_REPRODUCED")
    envelope_path=root/"state/style-balance/portfolio-buffer-probes-20261102-c/observed-error-envelope.json"
    envelope=json.loads(envelope_path.read_text(encoding="utf-8"));lo,hi=robust_bounds(limits,envelope)
    buffered=optimize_daily(scores,raw,controls,F,cov,specific,wb,limits,pd.Series(dtype=float),lower_limits=lo,upper_limits=hi)
    args.output.mkdir(parents=True);(args.output/"gates").mkdir()
    for day in [signal,date]:
        shutil.copy2(source/f"observations-{day}.parquet",args.output/f"observations-{day}.parquet")
    bars.to_parquet(args.output/"source-price-history.parquet",index=False)
    pd.DataFrame({"unbuffered":base,"buffered":buffered}).to_parquet(args.output/f"weights-{signal}-archived_alpha_1010.parquet")
    parent=json.loads((source/"results.json").read_text(encoding="utf-8"));gate_rows=[]
    preflight=json.loads((root/"state/enhanced-indexing-research/replay-preflight.json").read_text(encoding="utf-8"))
    for endpoint in ["stk_limit","suspend_d"]:
        original=Path(preflight["evidence_directory"])/f"{endpoint}-{date}.parquet"
        if str(original) in parent["input_hashes"] and hashlib.sha256(original.read_bytes()).hexdigest()!=parent["input_hashes"][str(original)]:raise ValueError("GATE_CHANGED")
        target=args.output/"gates"/original.name;shutil.copy2(original,target)
        gate_rows.append({"file":target.name,"sha256":hashlib.sha256(target.read_bytes()).hexdigest()})
    (args.output/"gates/manifest.json").write_text(json.dumps({"status":"READY","requests":gate_rows},indent=2),encoding="utf-8")
    protocol=yaml.safe_load((root/"configs/style_buffer_integer_v1.yaml").read_text(encoding="utf-8"))
    protocol.update(scope="known_october_earliest_archived_alpha_engineering_not_holdout",probe="archived_alpha_1010",dates=[date])
    protocol["excluded_claims"].append("September_out_of_sample_efficacy_proven_by_this_engineering_case")
    (args.output/"protocol.yaml").write_text(yaml.safe_dump(protocol,allow_unicode=True),encoding="utf-8")
    result={"source_envelope_sha256":hashlib.sha256(envelope_path.read_bytes()).hexdigest(),"transition_counts":[{"from_date":signal,"to_date":date}],
        "archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),"base_reproduction_max_difference":difference,
        "positive_targets":{name:int((w>1e-6).sum()) for name,w in [("unbuffered",base),("buffered",buffered)]},"orders_enabled":False}
    (args.output/"results.json").write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps(result))


if __name__=="__main__":main()
