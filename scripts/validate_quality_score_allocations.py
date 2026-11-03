"""Fixed heldout real-score portfolios under explicit quality contract revisions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from qlib.model.riskmodel import StructuredCovEstimator
from moneymore.data.store import ParquetStore
from moneymore.challenger_diagnostics import trading_dates
from moneymore.research.frozen_risk_scale import CORE
from moneymore.research.style_missing_controls import uncertain_controls
from moneymore.research.style_replay_inputs import optimize_daily,prospective_capture
from moneymore.research.quality_contract import quality_upper_only_bounds
from moneymore.research.input_versions import require_input_version


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path);parser.add_argument('--inputs',type=Path);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_RESEARCH')
    root=Path(__file__).resolve().parents[1];source=args.source or root/'state/style-balance/real-archive-group-decomposition-20261102-b'
    report=json.loads((source/'results.json').read_text())
    require_input_version(report,args.inputs)
    hashes={**report['input_hashes'],**report['archive_source_hashes'],
        **{str(source/name):digest for name,digest in report['derived_input_hashes'].items()}}
    hashes[str(source/'source-price-history.parquet')]=report['price_history_sha256']
    for path,digest in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('FROZEN_SOURCE_CHANGED')
    envelope=json.loads((source/'observed-error-envelope.json').read_text())
    liquidity=pd.read_parquet(source/'registered-liquidity.parquet').set_index(['date','symbol']).turnover_mean_20
    bars=pd.read_parquet(source/'source-price-history.parquet');quote=bars.set_index(['trade_date','ts_code'])
    prices=bars.pivot(index='trade_date',columns='ts_code',values='total_price')
    calendar=trading_dates(ParquetStore(root/'data'),'20260930')
    frames={};controls={};references={}
    for path in sorted((args.inputs or root/'state/style-balance/preperiod-drift-202609-b').glob('observations-*.parquet')):
        date=path.stem.split('-')[-1];raw=pd.read_parquet(path)
        raw=raw.drop(columns=['turnover_rate']).assign(turnover_mean_20=liquidity.loc[date].reindex(raw.index))
        c=uncertain_controls(raw,CORE)[0]
        for field in raw.select_dtypes(include='number').columns:
            if field not in CORE:c['unknown_'+field]=raw[field].replace([np.inf,-np.inf],np.nan).isna().astype(float)
        frames[date]=raw;controls[date]=c;wb=np.exp(raw.log_size);references[date]=wb/wb.sum()
    args.output.mkdir(parents=True)
    protocol={'scope':'REAL_ARCHIVED_SCORE_FRACTIONAL_CONTRACT_COMPARISON_NOT_ADOPTED',
        'calibration_end':'20260922','validation_start':'20260923','validation_end':'20260930',
        'variants':['legacy_unbuffered','quality_unbuffered','quality_buffered'],
        'orders_enabled':False,'turnover':'independent_cash_entry_not_continuous_trades',
        'reference':'same_cap_reference_and_same_daily_risk_model_all_variants',
        'quality_revision':'explicit_no_unknown_minimum_holding; original_unknown_upper_preserved',
        'limitations':['no_integer_lots_fees_or_liquidity','no_return_or_predictive_improvement_claim','five_correlated_market_days']}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[];factor_rows=[]
    for case in report['transition_counts']:
        date,next_date=case['from_date'],case['to_date']
        if next_date<'20260923':continue
        raw=frames[date];c=controls[date];new=controls[next_date];wb=references[date];new_wb=references[next_date]
        limits=pd.Series({f:.01 if f.startswith('unknown_') else .03 if f.startswith(('industry_','board_')) else .1 for f in c.columns})
        zero={f:{'upper':0.,'lower':0.} for f in c.columns}
        days=[d for d in calendar if d<=date][-121:]
        returns=prices.reindex(index=days,columns=raw.index).pct_change(fill_method=None).iloc[1:]
        if not np.isfinite(returns.to_numpy()).all():raise ValueError('REQUIRED_RISK_HISTORY_GAP')
        np.random.seed(17)
        F,cov,specific=StructuredCovEstimator(num_factors=10,factor_model='pca',scale_return=False).predict(returns,is_price=False,return_decomposed_components=True)
        original=pd.read_parquet(source/f'diagnostic-weights-{date}.parquet')
        for probe,directory in [('archived_main_gru',root/'state/qlib-challenger-shadow'),('archived_candidate_20260812',root/'state/qlib-candidate-shadow/20260812')]:
            archive=json.loads((directory/f'{date}.json').read_text(encoding='utf-8'))
            if archive['trade_date']!=date or not prospective_capture(date,archive['created_at'],next_date):raise ValueError('NONPROSPECTIVE_ARCHIVE')
            score=pd.DataFrame(archive['scores']).set_index('instrument').score
            targets={}
            for variant in protocol['variants']:
                row={'from_date':date,'to_date':next_date,'probe':probe,'variant':variant}
                try:
                    if variant=='legacy_unbuffered':target=original[probe].reindex(raw.index)
                    else:target=optimize_daily(score,raw,c,F,cov,specific,wb,limits,pd.Series(dtype=float),research_quality_envelope=zero if variant=='quality_unbuffered' else envelope)
                except ValueError as error:
                    rows.append({**row,'status':'BLOCKED','reason':str(error)});continue
                lo,hi=(-limits,limits) if variant=='legacy_unbuffered' else quality_upper_only_bounds(c,wb,limits,zero if variant=='quality_unbuffered' else envelope)
                forecast=(target-wb) @ c
                if (not np.isfinite(target).all() or target.min() < -1e-8 or target.max()>.15+1e-6 or abs(target.sum()-1)>1e-8
                    or (forecast<lo-1e-6).any() or (forecast>hi+1e-6).any()):raise ValueError('INDEPENDENT_TARGET_POSTCHECK_FAILED')
                targets[variant]=target
                support=target[target>1e-8].copy()
                if target.sum()-support.sum()>1e-6:raise ValueError('MASS_CANNOT_BE_DROPPED')
                support/=support.sum();q=quote.loc[next_date].reindex(support.index)
                if set(support.index)-set(new.index) or set(c.columns)-set(new.columns) or not np.isfinite(q[['open','close']]).all().all() or (q[['open','close']]<=0).any().any():
                    rows.append({**row,'status':'UNMEASURABLE_NEXT_INPUTS'});continue
                value=support*q.close/q.open;end=value/value.sum()
                outcome=end @ (new-(new_wb @ new)).reindex(index=support.index,columns=c.columns)
                economic=[];quality=[];legacy_lower=[]
                for field in c.columns:
                    is_quality=field.startswith('unknown_')
                    upper=max(0.,float(outcome[field]-limits[field]))
                    lower=max(0.,float(-limits[field]-outcome[field]))
                    breach=(upper>1e-6) if is_quality and variant!='legacy_unbuffered' else max(upper,lower)>1e-6
                    (quality if is_quality else economic).append(breach)
                    if is_quality:legacy_lower.append(lower>1e-6)
                    factor_rows.append({**row,'factor':field,'outcome':float(outcome[field]),'upper_excess':upper,'legacy_lower_excess':lower,'contract_breach':breach})
                rows.append({**row,'status':'MEASURED','holdings':len(support),'economic_breaches':sum(economic),'quality_breaches':sum(quality),
                    'unknown_upper_breaches':sum(float(outcome[f]-limits[f])>1e-6 for f in c.columns if f.startswith('unknown_')),
                    'legacy_unknown_lower_breaches':sum(legacy_lower),'target_l1_vs_legacy':float(abs(target-original[probe]).sum())})
            pd.DataFrame(targets).to_parquet(args.output/f'weights-{date}-{probe}.parquet')
        print(json.dumps({'to_date':next_date,'completed':True}),flush=True)
    table=pd.DataFrame(rows);table.to_parquet(args.output/'portfolio-outcomes.parquet',index=False)
    pd.DataFrame(factor_rows).to_parquet(args.output/'factor-outcomes.parquet',index=False)
    summary={}
    for variant in protocol['variants']:
        x=table.loc[table.variant.eq(variant) & table.status.eq('MEASURED')]
        summary[variant]={'measured_cases':len(x),'economic_breach_cases':int(x.economic_breaches.gt(0).sum()) if len(x) else None,
            'quality_breach_cases':int(x.quality_breaches.gt(0).sum()) if len(x) else None,
            'unknown_upper_rows':int(x.unknown_upper_breaches.sum()) if len(x) else None,
            'all_contracts_passed_cases':int((x.economic_breaches.eq(0)&x.quality_breaches.eq(0)).sum()) if len(x) else None}
    result={'status':'COMPLETED_NOT_ADOPTED','input_version':str(args.inputs) if args.inputs else None,'summary':summary,'unmeasured':table.loc[~table.status.eq('MEASURED')].fillna('').to_dict('records'),
        'orders_enabled':False,'adopted':False,'input_hashes':hashes,
        'envelope_sha256':hashlib.sha256((source/'observed-error-envelope.json').read_bytes()).hexdigest()}
    (args.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(summary))


if __name__=='__main__':main()
