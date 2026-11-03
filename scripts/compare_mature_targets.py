"""P3 current-close comparison; no accounts or orders created here."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from moneymore.data.store import ParquetStore
from moneymore.research.mature_multifactor import current_risk_inputs, topk_dropout_targets
from moneymore.research.enhanced_indexing import EnhancedPolicy, optimize_snapshot


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_TARGET_COMPARISON')
    args.output.mkdir(parents=True)
    root=Path(__file__).resolve().parents[1];manifest=json.loads((args.model/'results.json').read_bytes())
    for name,digest in manifest['artifact_hashes'].items():
        if hashlib.sha256((args.model/name).read_bytes()).hexdigest()!=digest:raise ValueError('MODEL_ARTIFACT_CHANGED')
    inputs=Path(manifest['inputs']);bars=pd.read_parquet(inputs/'bars-input.parquet')
    prediction=pd.read_parquet(args.model/'latest-predictions.parquet').score
    day=manifest['data_cutoff'];scores=prediction.xs(pd.Timestamp(day),level='datetime')
    store=ParquetStore(root/'data');returns,size,raw,controls,benchmark,limits,audit=current_risk_inputs(store,bars,scores,day)
    eligible=scores.reindex(controls.index)
    control=topk_dropout_targets(eligible,pd.Series(dtype=float))
    print(json.dumps({'stage':'NATIVE_STYLE_OPTIMIZATION','eligible':len(eligible),'controls':len(controls.columns)}),flush=True)
    balanced=optimize_snapshot(eligible,returns,size,pd.Series(dtype=float),EnhancedPolicy(objective_preconditioning=True),
        style_controls=controls,style_limits=limits,benchmark_weights=benchmark,controls_prestandardized=True,allow_cash_bootstrap=True,reject_turnover_relaxation=True,solver_max_iters=1000)
    (args.output/'optimizer.json').write_text(json.dumps(balanced,indent=2,allow_nan=False))
    if balanced['status']!='PASS_OFFLINE_ONLY':raise ValueError('NATIVE_STYLE_OPTIMIZER_BLOCKED:'+','.join(balanced['failures']))
    def metrics(target):
        w=pd.Series(target).reindex(controls.index,fill_value=0.);active=(w-benchmark)@controls
        board=raw.reindex(w.index).listing_board
        return {'targets':target,'effective_holdings':float(1/(w@w)), 'names_above_1bp':int((w>.0001).sum()),
            'chinext_weight':float(w.loc[board.eq('CHINEXT')].sum()),'star_weight':float(w.loc[board.eq('STAR')].sum()),
            'industry_weights':w.groupby(raw.reindex(w.index).industry).sum().to_dict(),
            'active_styles':active.to_dict(), 'maximum_normalized_band_use':float(max(abs(active[c])/limits[c] for c in controls))}
    comparison={'raw':metrics(control),'balanced':metrics(balanced['weights']),'benchmark':metrics(benchmark.to_dict())}
    result={'status':'P3_CONTINUOUS_TARGETS_READY_NOT_YET_INTEGER_EXECUTION','signal_date':day,'model':str(args.model.resolve()),
        'same_prediction':True,'same_risk_eligible_pool':True,'comparison':comparison,'input_audit':audit,
        'reference_caps':size.to_dict(),'style_limits':limits,'real_orders_enabled':False,'promotion_authorized':False,
        'model_manifest_sha256':hashlib.sha256((args.model/'results.json').read_bytes()).hexdigest()}
    controls.to_parquet(args.output/'controls.parquet');returns.to_parquet(args.output/'returns.parquet')
    (args.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({name:{k:v for k,v in value.items() if k in ['effective_holdings','names_above_1bp','chinext_weight','star_weight','maximum_normalized_band_use']} for name,value in comparison.items()}),flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):main()
