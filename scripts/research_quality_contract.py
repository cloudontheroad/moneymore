"""Quality-vs-style contract comparison: geometry and unchanged-target audit only."""
import argparse
import hashlib
import json
from pathlib import Path
import cvxpy as cp
import numpy as np
import pandas as pd
from moneymore.research.portfolio_error_envelope import robust_bounds
from moneymore.research.quality_contract import quality_upper_only_bounds
from moneymore.research.style_missing_controls import uncertain_controls
from moneymore.research.frozen_risk_scale import CORE


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_RESEARCH')
    root=Path(__file__).resolve().parents[1]
    source=root/'state/style-balance/real-archive-group-decomposition-20261102-b'
    report=json.loads((source/'results.json').read_text())
    for path,digest in report['input_hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('INPUT_CHANGED')
    for name,digest in report['derived_input_hashes'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:raise ValueError('DERIVED_INPUT_CHANGED')
    envelope=json.loads((source/'observed-error-envelope.json').read_text())
    errors=pd.read_parquet(source/'portfolio-exposure-errors.parquet')
    panel=pd.read_parquet(source/'registered-liquidity.parquet').set_index(['date','symbol']).turnover_mean_20
    args.output.mkdir(parents=True)
    protocol={'scope':'RESEARCH_CONTRACT_REVISION_NOT_ADOPTED','orders_enabled':False,
        'economic_bounds':'unchanged_original_plus_full_error_buffer',
        'quality_upper':'original_0.01_minus_full_positive_error_including_label_deterioration',
        'quality_lower':'physical_zero_absolute_missing_weight_not_legacy_symmetric_floor',
        'sample':'five_fixed_heldout_signal_dates_plus_first_October_snapshot',
        'not_claimed':['new_alpha','integer_execution','return_improvement','unchanged_legacy_contract']}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    dates=sorted(errors.loc[errors.to_date.ge('20260923'),'from_date'].unique())+['20261012']
    for date in dates:
        if date=='20261012':
            c=pd.read_parquet(source/'evaluation-controls-20261012.parquet')
            wb=pd.read_parquet(root/'state/style-balance/expanded-style-10bps-20261102-b/weights-20261012.parquet').benchmark
        else:
            raw=pd.read_parquet(root/f'state/style-balance/preperiod-drift-202609-b/observations-{date}.parquet')
            raw=raw.drop(columns=['turnover_rate']).assign(turnover_mean_20=panel.loc[date].reindex(raw.index))
            c=uncertain_controls(raw,CORE)[0]
            for field in raw.select_dtypes(include='number').columns:
                if field not in CORE:c['unknown_'+field]=raw[field].replace([np.inf,-np.inf],np.nan).isna().astype(float)
            wb=np.exp(raw.log_size);wb/=wb.sum()
        wb=wb.reindex(c.index)
        limits=pd.Series({f:.01 if f.startswith('unknown_') else .03 if f.startswith(('industry_','board_')) else .1 for f in c.columns})
        for variant in ['legacy_symmetric','quality_upper_only_v1']:
            lo,hi=robust_bounds(limits,envelope) if variant=='legacy_symmetric' else quality_upper_only_bounds(c,wb,limits,envelope)
            crossed=lo.index[lo>hi+1e-10].tolist();row={'date':date,'contract':variant,'crossed_intervals':crossed}
            if crossed:row['status']='INFEASIBLE_INTERVALS'
            else:
                w=cp.Variable(len(c));active=c.to_numpy().T@(w-wb.to_numpy())
                problem=cp.Problem(cp.Minimize(cp.sum_squares(w-wb.to_numpy())),
                    [w>=0,w<=.15,cp.sum(w)==1,active>=lo.reindex(c.columns).to_numpy(),active<=hi.reindex(c.columns).to_numpy()])
                problem.solve(solver=cp.CLARABEL,max_iter=300,tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10)
                row['status']=problem.status
                if problem.status=='optimal':
                    actual=c.to_numpy().T@(w.value-wb.to_numpy())
                    if (not np.isfinite(w.value).all() or w.value.min() < -1e-7 or w.value.max()>.15+1e-7
                        or abs(w.value.sum()-1)>1e-7 or (actual<lo.to_numpy()-1e-7).any() or (actual>hi.to_numpy()+1e-7).any()):
                        raise ValueError('INDEPENDENT_WITNESS_CHECK_FAILED')
                    pd.DataFrame({'reference':wb,'witness':w.value}).to_parquet(args.output/f'witness-{date}-{variant}.parquet')
            rows.append(row)
    # Same ORIGINAL targets; contract audit does not retrade or fabricate returns.
    quality=errors.loc[errors.factor.str.startswith('unknown_')].copy()
    quality['upper_violation']=quality.outcome.gt(.01+1e-6)
    quality['legacy_lower_violation']=quality.outcome.lt(-.01-1e-6)
    quality.to_parquet(args.output/'unchanged-target-quality-audit.parquet',index=False)
    summary={}
    for split,mask in [('calibration',quality.to_date.le('20260922')),('heldout',quality.to_date.ge('20260923'))]:
        x=quality.loc[mask]
        summary[split]={'rows':len(x),'upper_violations':int(x.upper_violation.sum()),'legacy_lower_violations':int(x.legacy_lower_violation.sum()),
            'not_an_execution_or_return_comparison':True}
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source/'results.json',source/'observed-error-envelope.json',source/'portfolio-exposure-errors.parquet']}
    result={'status':'COMPLETED_RESEARCH_NOT_ADOPTED','geometry':rows,'unchanged_target_audit':summary,
        'source_hashes':hashes,'orders_enabled':False,'contract_adopted':False,
        'legacy_lower_bound_removed_explicit_contract_revision':True,
        'unknown_positive_buffer_includes_all_observed_label_changes':True}
    (args.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result))


if __name__=='__main__':main()
