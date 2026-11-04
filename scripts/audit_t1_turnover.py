"""T1: factual order classification and signal/position alignment, no decisions."""
from pathlib import Path
import json,sqlite3
import pandas as pd
import numpy as np
from moneymore.research.replay_attribution import reconstruct_positions

ROOT=Path(__file__).resolve().parents[1];STUDY=ROOT/'state/four-strategy-2026-jan-sep-v1'
OUT=ROOT/'state/cost-aware-ab-v2/t1';OUT.mkdir(parents=True,exist_ok=True)

if __name__=='__main__':
    membership=pd.read_parquet(ROOT/'state/mature-multifactor/p1-native-20261103-a/membership.parquet')
    membership['day']=pd.to_datetime(membership.datetime).dt.strftime('%Y%m%d')
    pools={d:set(g.instrument) for d,g in membership.groupby('day')}
    scores=pd.read_parquet(STUDY/'model/predictions.parquet').score
    bars=pd.read_parquet(ROOT/'state/mature-multifactor/p1-native-20261103-a/bars-input.parquet');bars.trade_date=bars.trade_date.astype(str)
    extra=pd.read_parquet(STUDY/'official/held-exit-002405-202609.parquet')
    bars=pd.concat([bars,extra],ignore_index=True).drop_duplicates(['trade_date','ts_code'])
    prices=bars.set_index(['trade_date','ts_code']).open
    report={}
    for kind in ['raw','balanced']:
        folder=STUDY/'parallel'/kind
        with sqlite3.connect(f'file:{(folder/kind/"paper.sqlite3").as_posix()}?mode=ro',uri=True) as c:
            fills=pd.read_sql_query('SELECT * FROM fills',c);actions=pd.read_sql_query('SELECT * FROM corporate_action_ledger',c)
        eq=pd.read_parquet(folder/kind/'equity.parquet');snapshots=reconstruct_positions(fills,actions,eq.date.tolist())
        events=json.loads((folder/'checkpoint.json').read_bytes())['events'][kind]
        before={};rows=[];alignment=[];last_side={};cycles={};durations=[]
        for i,event in enumerate(events):
            day=event['date'];signal=event['signal_date'];after=snapshots[day]
            prediction=scores.xs(pd.Timestamp(signal),level='datetime').sort_values(ascending=False)
            top=set(prediction.head(50).index);rank=prediction.rank(ascending=False)
            values={s:q*float(prices.get((day,s),np.nan)) for s,q in after.items()}
            total=sum(v for v in values.values() if np.isfinite(v))
            alignment.append({'date':day,'top50_value_fraction':sum(v for s,v in values.items() if s in top and np.isfinite(v))/total,
                'unscored_value_fraction':sum(v for s,v in values.items() if s not in prediction.index and np.isfinite(v))/total,
                'weighted_prediction_rank':sum(v*float(rank[s]) for s,v in values.items() if s in rank.index and np.isfinite(v))/sum(v for s,v in values.items() if s in rank.index and np.isfinite(v))})
            for f in fills.loc[fills.trade_date.eq(day)].itertuples():
                q0=before.get(f.symbol,0);q1=after.get(f.symbol,0)
                category=('INITIAL_ALLOCATION' if not before else 'CONFIRMED_POOL_EXIT' if f.side=='SELL' and f.symbol not in pools[signal]
                    else 'ALPHA_ENTRY' if f.side=='BUY' and q0==0 else 'ALPHA_EXIT' if f.side=='SELL' and q1==0 else 'RETAINED_RESIZE')
                prior=last_side.get(f.symbol);reversal=prior is not None and prior[0]!=f.side and i-prior[1]<=5
                reference=float(prices[(day,f.symbol)]);slippage=(f.price-reference)*f.quantity*(1 if f.side=='BUY' else -1)
                rows.append({'date':day,'symbol':f.symbol,'side':f.side,'quantity':f.quantity,'notional':f.price*f.quantity,'fee':f.fee,
                    'slippage':slippage,'category':category,'reversal_within_5_sessions':reversal,
                    'less_than_1bp_initial_capital':f.price*f.quantity<1000,'rank':float(rank.get(f.symbol,np.nan))})
                last_side[f.symbol]=(f.side,i)
            for s in set(before)|set(after):
                if s not in before and s in after:cycles[s]=i
                if s in before and s not in after:durations.append(i-cycles.pop(s,0))
            before=after
        detail=pd.DataFrame(rows);align=pd.DataFrame(alignment)
        detail.to_parquet(OUT/(kind+'-orders.parquet'),index=False);align.to_parquet(OUT/(kind+'-alignment.parquet'),index=False)
        groups=detail.groupby('category').agg(orders=('quantity','count'),notional=('notional','sum'),fees=('fee','sum'),slippage=('slippage','sum')).reset_index()
        assert len(detail)==len(fills) and abs(detail.fee.sum()-fills.fee.sum())<.0001
        report[kind]={'categories':groups.to_dict('records'),'total_orders':len(fills),'total_notional':float(detail.notional.sum()),
            'retained_resize_notional_fraction':float(detail.loc[detail.category.eq('RETAINED_RESIZE'),'notional'].sum()/detail.notional.sum()),
            'five_session_reversal_notional_fraction':float(detail.loc[detail.reversal_within_5_sessions,'notional'].sum()/detail.notional.sum()),
            'sub_1000_yuan_order_fraction':float(detail.less_than_1bp_initial_capital.mean()),
            'mean_top50_value_fraction':float(align.top50_value_fraction.mean()),'mean_prediction_rank':float(align.weighted_prediction_rank.mean()),
            'mean_unscored_value_fraction':float(align.unscored_value_fraction.mean()),'closed_cycles':len(durations),
            'median_closed_cycle_sessions':float(np.median(durations)),'mean_closed_cycle_sessions':float(np.mean(durations)),
            'notes':['Resize is a factual quantity change, NOT automatically useless','Reversals include sensible partial adjustments, NOT a causal loss estimate',
                'Historical replay values incumbents at next open; production prepares previous-close targets. T5 must lock one timeline for every component comparison.']}
    (OUT/'results.json').write_text(json.dumps({'status':'T1_FACTUAL_AUDIT_COMPLETE','accounts':report,'production_modified':False},indent=2),encoding='utf-8')
    print(json.dumps(report))
