"""Original 44 B-blocked dates, exact-state diagnostics; no production writes."""
import json,sqlite3,hashlib
from collections import Counter
import pandas as pd
from threadpoolctl import threadpool_limits
from replay_cost_ab_v2 import ROOT,OUT,STUDY,StudyStore
from prepare_four_strategy_2026 import save
from moneymore.research.replay_attribution import reconstruct_positions,reconstruct_cash
from moneymore.research.mature_multifactor import current_risk_inputs
from moneymore.research.cost_risk import cost_risk_inputs
from moneymore.research.cost_integer import plan_integer_execution
from moneymore.research.cost_rebalance import cost_targets
from moneymore.research.quote_permissions import quote_permissions
from moneymore.research.replay_actions import official_replay_actions
from moneymore.config import BacktestConfig

def main():
    base=STUDY/'parallel/balanced';book=base/'balanced';events=json.loads((base/'checkpoint.json').read_bytes())['events']['balanced']
    days=[e['date'] for e in events]
    with sqlite3.connect(f'file:{(book/"paper.sqlite3").as_posix()}?mode=ro',uri=True) as db:
        fills=pd.read_sql_query('SELECT * FROM fills',db);ledger=pd.read_sql_query('SELECT * FROM corporate_action_ledger',db);entitlements=pd.read_sql_query('SELECT * FROM corporate_entitlements',db)
    positions=reconstruct_positions(fills,ledger,days);cash_history=reconstruct_cash(fills,ledger,days,1e7)
    source=ROOT/'state/mature-multifactor/p1-native-20261103-a';bars=pd.read_parquet(source/'bars-input.parquet');bars.trade_date=bars.trade_date.astype(str)
    for extra in [STUDY/'official/held-exit-002405-202609.parquet',*sorted((OUT/'quote-cache').glob('*.parquet'))]:
        if extra.exists():bars=pd.concat([bars,pd.read_parquet(extra)],ignore_index=True).drop_duplicates(['trade_date','ts_code'],keep='first')
    membership=pd.read_parquet(source/'membership.parquet');membership.datetime=pd.to_datetime(membership.datetime)
    predictions=pd.read_parquet(STUDY/'model/predictions.parquet').score
    store=StudyStore();proofs=[];store.basic=[];rows=[]
    config=BacktestConfig.model_validate({**BacktestConfig.from_yaml(ROOT/'configs/default.yaml').model_dump(),'initial_cash':1e7,'slippage_bps':10})
    calibration=json.loads((OUT/'t2/historical.json').read_bytes())
    action_rows=pd.concat([pd.read_parquet(p) for p in sorted((STUDY/'official').glob('dividend-*.parquet'))],ignore_index=True).drop_duplicates()
    action_map=official_replay_actions(action_rows.loc[action_rows.ts_code.isin(set(membership.instrument))]).set_index('action_key')
    all_official=sorted((STUDY/'official').glob('daily_basic-*.parquet'))
    for event in events:
        day=event['date'];signal=event['signal_date']
        if event['status']=='PASS_REFERENCE_QUOTES_ONLY':continue
        target=OUT/'t4/dates'/(day+'.json')
        if target.exists():rows.append(json.loads(target.read_bytes()));continue
        store.day=signal;store.basic=[pd.read_parquet(p) for p in all_official if p.stem.split('-')[-1]<=signal]
        proof=pd.concat([pd.read_parquet(p) for p in sorted((STUDY/'official').glob('suspend_d-*.parquet')) if p.stem.split('-')[-1]<=signal],ignore_index=True)
        held=dict(positions[days[days.index(day)-1]]) if days.index(day)>0 else {};cash=cash_history[days[days.index(day)-1]] if days.index(day)>0 else 1e7
        for item in ledger.loc[ledger.trade_date.eq(day)].itertuples():
            if item.action_type=='CASH_DIVIDEND':cash+=item.cash_amount
            if item.action_type=='STOCK_DIVIDEND':held[item.symbol]=held.get(item.symbol,0)+item.share_quantity
        members=set(membership.loc[membership.datetime.eq(pd.Timestamp(signal)),'instrument']);scores=predictions.xs(pd.Timestamp(signal),level='datetime').reindex(sorted(members)).dropna()
        returns,size,raw,styles,benchmark,limits,audit=current_risk_inputs(store,bars,scores.reindex(sorted(set(scores.index)|set(held))),signal,suspension_proof=proof,candidate_members=members)
        eligible=set(styles.index)&set(scores.index);missing_forecast=(set(held)&members)-eligible;missing_risk=set(held)-set(styles.index)
        entitled=[e for e in entitlements.itertuples() if e.record_date<=signal and e.action_key in action_map.index and action_map.loc[e.action_key,'ex_date']<=pd.Timestamp(day)]
        pending={e.symbol:int(e.entitled_quantity*e.stock_ratio) for e in entitled if e.stock_ratio>0 and e.stock_list_date>day}
        receivable_cash=sum(e.entitled_quantity*e.cash_per_share for e in entitled if e.cash_per_share>0 and e.cash_pay_date>day)
        row={'date':day,'signal_date':signal,'original_status':event['status'],'original_reason':event.get('reason'),
            'missing_member_forecasts':sorted(missing_forecast),'missing_risk_inputs':sorted(missing_risk),'pending_stock_rights':pending,
            'exited_members':sorted(set(held)-members),'scope':'ORIGINAL_LEDGER_STATE_NOT_DIVERGENT_NEW_ACCOUNT'}
        if event['status']=='INPUT_HOLD':
            if missing_risk:
                repaired=cost_risk_inputs(store,bars,scores.reindex(sorted(set(scores.index)|set(held))),signal,suspension_proof=proof,candidate_members=members)
                row['repaired_missing_risk_inputs']=sorted(set(held)-set(repaired[3].index))
                row['proven_window_start_seeds']=repaired[-1]['cost_window_start_proven_halt_seeds']
                if not row['repaired_missing_risk_inputs']:
                    row['resolution']='PROVEN_PRIOR_HALT_ANCHOR_RESTORED_FREEZE_MISSING_FORECAST_ONLY'
                    save(target,row);rows.append(row);continue
            row['resolution']='FREEZE_ONLY_MISSING_FORECAST_DIMENSIONS' if missing_forecast and not missing_risk else 'ACCOUNT_PENDING_RIGHTS_AS_LOCKED_SHARES' if pending and not missing_risk else 'REMAIN_BLOCKED_MISSING_REQUIRED_RISK' if missing_risk else 'REQUIRES_ADDITIONAL_INPUT_DIAGNOSIS'
        elif event['status']=='BLOCKED':
            original=json.loads((book/(day+'-integer.json')).read_bytes());row['original_solver']=original.get('solver_diagnostics')
            solved=json.loads((book/(day+'-continuous.json')).read_bytes())
            marks=bars.loc[bars.trade_date.eq(day)].set_index('ts_code').open.dropna().to_dict()
            susp=pd.read_parquet(STUDY/'official'/('suspend_d-'+day+'.parquet'));halted=set(susp.loc[susp.suspend_type.eq('S')&(susp.suspend_timing.isna()|susp.suspend_timing.eq('')),'ts_code'])
            gate=pd.read_parquet(STUDY/'official'/('stk_limit-'+day+'.parquet')).set_index('ts_code');buy,sell=quote_permissions(marks,halted,gate,10)
            if set(held)-set(marks):row['resolution']='REMAIN_BLOCKED_UNOBSERVED_REQUIRED_QUOTES'
            else:
                equity=cash+sum(q*marks[s] for s,q in held.items())+sum(q*marks[s] for s,q in pending.items())+receivable_cash
                result=plan_integer_execution(solved['weights'],marks,held,held,cash,equity,config,styles,limits,locked_shares=pending,
                    can_buy=buy,can_sell=sell,benchmark_weights=benchmark,style_controls_prestandardized=True,time_limit=60,solver_threads=4,mip_relative_gap=.01,
                    research_secondary_time_limit=10,accept_validated_incumbent=True,stock_active_bound=max(0.,.15-benchmark.max()),forced_sell_symbols=(set(held)-members)-halted)
                save(OUT/'t4/plans'/(day+'.json'),result)
                row.update(repaired_status=result['status'],repaired_reason=result.get('reason'),repaired_solver=result.get('solver_diagnostics'),
                    resolution='EXACT_POSTCHECKED_TARGET_PLAN' if result['status']=='PASS_REFERENCE_QUOTES_ONLY' else 'REMAIN_EXPLICITLY_BLOCKED_NO_VALIDATED_SOLUTION')
        else:
            row['resolution']='NATIVE_TURNOVER_RELAXATION_REJECTED_NOT_AUTHORIZED_TO_EXECUTE'
        save(target,row);rows.append(row)
        save(OUT/'t4/progress.json',{'completed':len(rows),'total':44,'latest_date':day})
    assert len(rows)==44
    save(OUT/'t4/results.json',{'status':'ORIGINAL_44_DATES_DIAGNOSED','rows':rows,'resolutions':dict(Counter(r['resolution'] for r in rows)),
        'all_blocked_days_forced_to_trade':False,'risk_constraints_silently_relaxed':False,'real_orders_enabled':False})
if __name__=='__main__':
    with threadpool_limits(limits=4):main()
