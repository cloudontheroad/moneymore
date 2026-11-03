"""P3 independent reconstructed position-risk audit plus fixed bias counterexamples."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from moneymore.research.replay_attribution import reconstruct_positions
from moneymore.research.enhanced_indexing import EnhancedPolicy, optimize_snapshot


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--targets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_STYLE_AUDIT')
    ledger=json.loads((args.study/'independent-audit.json').read_bytes())
    if ledger['status']!='PASS_LEDGER_AUDIT_NOT_STRATEGY_ACCEPTANCE':raise ValueError('INDEPENDENT_CASH_LEDGER_REQUIRED')
    results=json.loads((args.study/'results.json').read_bytes());binding=json.loads((args.study/'episode-binding.json').read_bytes());evidence=Path(binding['evidence'])
    market=pd.read_parquet(args.study/'market-history.parquet').set_index(['trade_date','ts_code'])
    actions=pd.read_parquet(Path(binding['audit'])/'official-actions.parquet');rows=[]
    for case in results['cases']:
        folder=args.study/case['account'];database=folder/'paper.sqlite3';before=hashlib.sha256(database.read_bytes()).hexdigest()
        with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as connection:
            fills=pd.read_sql('SELECT * FROM fills',connection);corporate=pd.read_sql('SELECT * FROM corporate_action_ledger',connection)
        days=pd.read_parquet(folder/'equity.parquet').date.tolist();positions=reconstruct_positions(fills,corporate,days)
        prior=market.reset_index();prior=prior.loc[prior.trade_date.lt(days[0])].sort_values('trade_date');last=prior.groupby('ts_code').tail(1).set_index('ts_code').close.to_dict()
        for event in case['events']:
            day=event['date'];controls=pd.read_parquet(folder/f'{day}-controls.parquet');benchmark=pd.read_parquet(folder/f'{day}-benchmark.parquet').weight
            open_marks=market.loc[day].open.dropna().to_dict();closing=market.loc[day].close.dropna().to_dict()
            proof=pd.read_parquet(evidence/f'suspend_d-{day}.parquet');halted=set(proof.loc[proof.suspend_type.eq('S')&(proof.suspend_timing.isna()|proof.suspend_timing.eq('')),'ts_code']) if not proof.empty else set()
            for symbol in set(positions[day])-set(open_marks):
                if symbol not in halted or symbol not in last:raise ValueError('UNCONFIRMED_HELD_PRICE_GAP')
                value=last[symbol]
                for action in actions.loc[actions.symbol.eq(symbol)&actions.ex_date.eq(pd.Timestamp(day))].itertuples():value=(value-action.cash_per_share)/(1+action.stock_ratio)
                open_marks[symbol]=value;closing[symbol]=value
            quantity=pd.Series(positions[day],dtype=float);values=quantity*pd.Series(open_marks).reindex(quantity.index)
            weights=(values/values.sum()).reindex(controls.index,fill_value=0.)
            active=(weights-benchmark)@controls
            row={'account':case['account'],'date':day,'status':event['status'],'star_weight':float(weights.loc[weights.index.str.startswith('68')].sum()),
                 'chinext_weight':float(weights.loc[weights.index.str.startswith('30')].sum())}
            if case['account']=='balanced' and event['status']=='PASS_REFERENCE_QUOTES_ONLY':
                plan=json.loads((folder/f'{day}-integer.json').read_bytes());limits=pd.Series(plan['style_active_upper_limits'])
                difference=max(abs(active-pd.Series(plan['active_style_exposures'])))
                excess=max(abs(active)-limits)
                if difference>1e-8 or excess>1e-5:raise ValueError('ACTUAL_POST_FILL_STYLE_NOT_FROZEN_BAND')
                row.update(maximum_excess=float(excess),independent_active_error=float(difference))
            row['close_active_styles']=(((quantity*pd.Series(closing).reindex(quantity.index))/(quantity*pd.Series(closing).reindex(quantity.index)).sum()).reindex(controls.index,fill_value=0.)-benchmark).dot(controls).to_dict()
            rows.append(row);last.update(closing)
        if hashlib.sha256(database.read_bytes()).hexdigest()!=before:raise ValueError('AUDIT_CHANGED_LEDGER')
    target=json.loads((args.targets/'results.json').read_bytes());returns=pd.read_parquet(args.targets/'returns.parquet');controls=pd.read_parquet(args.targets/'controls.parquet')
    size=pd.Series(target['reference_caps']);wb=pd.Series(target['comparison']['benchmark']['targets']);limits=target['style_limits'];extremes=[]
    # No tuning: both predeclared opposite preferences must satisfy the same bands.
    bank_columns=[c for c in controls if c.startswith('industry_801780')]
    if len(bank_columns)!=1:raise ValueError('BANK_INDUSTRY_CONTROL_NOT_IDENTIFIED')
    bank=controls[bank_columns[0]]
    dual=pd.Series([float(s.startswith(('30','68'))) for s in returns.columns],index=returns.columns)
    for name,scores in [('extreme_dual_board',dual),('extreme_bank',bank)]:
        solved=optimize_snapshot(scores,returns,size,pd.Series(dtype=float),EnhancedPolicy(objective_preconditioning=True),
            style_controls=controls,style_limits=limits,benchmark_weights=wb,controls_prestandardized=True,
            allow_cash_bootstrap=True,reject_turnover_relaxation=True,solver_max_iters=1000)
        if solved['status']!='PASS_OFFLINE_ONLY':raise ValueError('EXTREME_PREFERENCE_CONSTRAINT_FAILED')
        extremes.append({'name':name,'status':'PASS','maximum_band_use':max(abs(v['after'])/limits[k] for k,v in solved['named_style_active_exposures'].items()),
            'dual_board_weight':sum(w for s,w in solved['weights'].items() if s.startswith(('30','68')))})
    result={'status':'P3_ACTUAL_FILL_RISK_AND_COUNTEREXAMPLES_AUDITED','rows':rows,'extreme_preferences':extremes,
        'all_fill_risk_tests_passed':True,'close_drift_is_reported_not_guaranteed_zero':True,'legacy_or_live_accounts_modified':False,
        'source_ledger_audit_sha256':hashlib.sha256((args.study/'independent-audit.json').read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({'status':result['status'],'account_days':len(rows),'counterexamples':extremes}))


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):main()
