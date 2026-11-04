"""Read-only ledger loss attribution; exact PnL first, lagged peer diagnosis second."""
from pathlib import Path
import json,sqlite3,hashlib,argparse
from collections import Counter
import numpy as np
import pandas as pd
from moneymore.research.replay_attribution import reconstruct_positions,reconstruct_cash,stock_wealth_change,economic_daily_return
from moneymore.research.replay_actions import official_replay_actions
from moneymore.research.style_price_features import named_price_risk_exposures
from moneymore.research.baseline_v2_industry import industry_as_of
from moneymore.challenger_attribution import matched_day,linked_attribution
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];STUDY=ROOT/'state/four-strategy-2026-jan-sep-v1'
OUT=STUDY/'loss-attribution';OUT.mkdir(exist_ok=True)

def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def inputs():
    source=ROOT/'state/mature-multifactor/p1-native-20261103-a'
    bars=pd.read_parquet(source/'bars-input.parquet');bars.trade_date=bars.trade_date.astype(str)
    extra=STUDY/'official/held-exit-002405-202609.parquet'
    if extra.exists():bars=pd.concat([bars,pd.read_parquet(extra)],ignore_index=True).drop_duplicates(['trade_date','ts_code'],keep='first')
    membership=pd.read_parquet(source/'membership.parquet');membership['day']=pd.to_datetime(membership.datetime).dt.strftime('%Y%m%d')
    members=set(membership.instrument)
    raw=pd.concat([pd.read_parquet(p) for p in sorted((STUDY/'official').glob('dividend-2026*.parquet'))],ignore_index=True)
    actions=official_replay_actions(raw.loc[raw.ts_code.isin(members)])
    meta=pd.read_parquet(ROOT/'data/processed/instruments.parquet').drop_duplicates('ts_code').set_index('ts_code')
    industry=pd.read_parquet(ROOT/'data/processed/baseline_v2_industry_membership.parquet')
    close=bars.pivot(index='trade_date',columns='ts_code',values='close').sort_index()
    opening=bars.set_index(['trade_date','ts_code']).open
    adjusted=bars.assign(adjusted=bars.close*bars.adj_factor).pivot(index='trade_date',columns='ts_code',values='adjusted').sort_index()
    return bars,membership,actions,meta,industry,close,opening,adjusted

def exact(kind,data,folder=None):
    bars,membership,actions,meta,industry,close,opening,adjusted=data
    folder=STUDY/'parallel'/kind if folder is None else folder;database=folder/kind/'paper.sqlite3'
    with sqlite3.connect(f'file:{database.as_posix()}?mode=ro',uri=True) as connection:
        fills=pd.read_sql_query('SELECT * FROM fills',connection)
        ledger=pd.read_sql_query('SELECT * FROM corporate_action_ledger',connection)
        entitlements=pd.read_sql_query('SELECT * FROM corporate_entitlements',connection)
    equity=pd.read_parquet(folder/kind/'equity.parquet');days=equity.date.astype(str).tolist()
    positions=reconstruct_positions(fills,ledger,days);cash=reconstruct_cash(fills,ledger,days,1e7)
    ref=opening.reindex(pd.MultiIndex.from_frame(fills[['trade_date','symbol']])).to_numpy()
    assert np.isfinite(ref).all()
    fills['slippage']=(fills.price-ref)*fills.quantity*np.where(fills.side.eq('BUY'),1,-1)
    assert fills.slippage.ge(-1e-6).all()
    action_map=actions.set_index('action_key');prior_values={};prior_equity=1e7;prior_holdings={}
    records=[];daily=[];lagged={};maximum_error=0.;maximum_cash_error=0.
    events=json.loads((folder/'checkpoint.json').read_bytes())['events'][kind]
    event_map={r['date']:r for r in events}
    for day in days:
        q=positions[day]
        for symbol in q:
            if pd.notna(close.loc[day].get(symbol,np.nan)):continue
            suspension=pd.read_parquet(STUDY/'official'/('suspend_d-'+day+'.parquet'))
            halted=set(suspension.loc[suspension.suspend_type.eq('S')&(suspension.suspend_timing.isna()|suspension.suspend_timing.eq('')),'ts_code']) if not suspension.empty else set()
            assert symbol in halted,('UNPROVEN_MISSING_HELD_MARK',kind,day,symbol)
            previous=close.loc[close.index<day,symbol].dropna()
            assert len(previous),('NO_PRIOR_HELD_MARK',symbol)
            carried=float(previous.iloc[-1])
            for action in actions.loc[actions.symbol.eq(symbol)&actions.ex_date.eq(pd.Timestamp(day))].itertuples():carried=(carried-action.cash_per_share)/(1+action.stock_ratio)
            close.loc[day,symbol]=carried
        mark=close.loc[day];values={s:quantity*float(mark[s]) for s,quantity in q.items()}
        economic_holdings=dict(q)
        for ent in entitlements.itertuples():
            if ent.record_date>day:continue
            action=action_map.loc[ent.action_key]
            if pd.Timestamp(day)<action.ex_date:continue
            if ent.cash_per_share>0 and ent.cash_pay_date>day:
                values[ent.symbol]=values.get(ent.symbol,0)+ent.entitled_quantity*ent.cash_per_share
            if ent.stock_ratio>0 and ent.stock_list_date>day:
                shares=int(ent.entitled_quantity*ent.stock_ratio)
                values[ent.symbol]=values.get(ent.symbol,0)+shares*float(mark[ent.symbol])
                economic_holdings[ent.symbol]=economic_holdings.get(ent.symbol,0)+shares
        row=equity.loc[equity.date.eq(day)].iloc[0]
        dayfills=fills.loc[fills.trade_date.eq(day)];dayledger=ledger.loc[ledger.trade_date.eq(day)]
        pnls=stock_wealth_change(prior_values,values,dayfills,dayledger)
        error=sum(pnls.values())-(row.equity-prior_equity);maximum_error=max(maximum_error,abs(error))
        maximum_cash_error=max(maximum_cash_error,abs(cash[day]-row.cash))
        assert abs(error)<.01 and abs(cash[day]-row.cash)<.01, ('PNL_OR_CASH_IDENTITY',kind,day,error)
        industry_map=industry_as_of(industry,day).l1_name
        for symbol,pnl in pnls.items():
            fee=float(dayfills.loc[dayfills.symbol.eq(symbol),'fee'].sum());slip=float(dayfills.loc[dayfills.symbol.eq(symbol),'slippage'].sum())
            records.append({'date':day,'symbol':symbol,'name':meta.name.get(symbol,symbol),'industry':industry_map.get(symbol,'未知'),
                'board':'科创板' if symbol.startswith('68') else '创业板' if symbol.startswith('30') else '主板',
                'net_pnl':pnl,'fees':fee,'slippage':slip,'gross_cost_addback':pnl+fee+slip})
        fee=float(dayfills.fee.sum());slip=float(dayfills.slippage.sum())
        daily.append({'date':day,'equity':float(row.equity),'actual_return':float(row.equity/prior_equity-1),
            'net_pnl':float(row.equity-prior_equity),'fees':fee,'slippage':slip,'status':event_map[day]['status'],
            'reason':str(event_map[day]['reason']) if event_map[day].get('reason') is not None else None,'gross_cost_addback':float(row.equity-prior_equity+fee+slip)})
        lagged[day]={'holdings':prior_holdings,'equity':prior_equity}
        prior_values=values;prior_equity=float(row.equity);prior_holdings=economic_holdings
    detail=pd.DataFrame(records);d=pd.DataFrame(daily)
    detail.to_parquet(OUT/(kind+'-stock-daily.parquet'),index=False);d.to_parquet(OUT/(kind+'-daily.parquet'),index=False)
    totals=detail.groupby(['symbol','name']).agg(net_pnl=('net_pnl','sum'),fees=('fees','sum'),slippage=('slippage','sum'),gross_cost_addback=('gross_cost_addback','sum')).reset_index()
    totals['contribution_pp']=totals.net_pnl/1e5;totals=totals.sort_values('net_pnl');totals.to_parquet(OUT/(kind+'-stocks.parquet'),index=False)
    monthly=[];previous=1e7
    for month,g in d.groupby(d.date.str[:6]):
        end=float(g.equity.iloc[-1]);monthly.append({'month':month,'return':end/previous-1,'pnl':float(g.net_pnl.sum()),'contribution_pp':float(g.net_pnl.sum()/1e5),'fees':float(g.fees.sum()),'slippage':float(g.slippage.sum())});previous=end
    groups={name:detail.groupby(name)[['net_pnl','fees','slippage','gross_cost_addback']].sum().reset_index().sort_values('net_pnl').to_dict('records') for name in ['industry','board']}
    losses=float(-totals.loc[totals.net_pnl.lt(0),'net_pnl'].sum());profits=float(totals.loc[totals.net_pnl.gt(0),'net_pnl'].sum())
    result={'strategy':kind,'status':'EXACT_LEDGER_RECONCILED','net_pnl':float(d.net_pnl.sum()),'fees':float(d.fees.sum()),'slippage':float(d.slippage.sum()),
        'gross_cost_addback':float(d.gross_cost_addback.sum()),'turnover_total_notional_multiple':float((fills.price*fills.quantity).sum()/1e7),
        'monthly':monthly,'groups':groups,'worst_stocks':totals.head(15).to_dict('records'),'best_stocks':totals.tail(10).sort_values('net_pnl',ascending=False).to_dict('records'),
        'losing_names_total_pnl':-losses,'winning_names_total_pnl':profits,'maximum_daily_pnl_error':maximum_error,'maximum_cash_error':maximum_cash_error,
        'execution_state_pnl':d.groupby('status')[['net_pnl','fees','slippage']].sum().reset_index().to_dict('records'),
        'protection_reasons':dict(Counter(str(r.get('reason')) for r in events if r['status']!='PASS_REFERENCE_QUOTES_ONLY')),
        'no_cost_addback_is_not_a_reoptimized_counterfactual':True,'protection_day_pnl_is_not_the_causal_effect_of_protection':True}
    dump(OUT/(kind+'-exact.json'),result)
    return result,lagged,daily

def peers(kind,data,lagged,daily):
    from moneymore.data.fundamental_research import research_financial_frame
    from moneymore.data.store import ParquetStore
    from moneymore.research.baseline_v2_financials import common_period_features
    from moneymore.research.style_risk_inputs import signed_inverse
    bars,membership,actions,meta,industry,close,opening,adjusted=data
    financial=research_financial_frame(ParquetStore(ROOT/'data'))
    names=['market','style','selection','unmatched_stock','trading_and_other','costs'];samples={n:[] for n in [3,5,10]}
    days=close.index.tolist();pool_map={d:g.instrument.tolist() for d,g in membership.groupby('day')}
    for i,row in enumerate(daily):
        day=row['date'];signal=days[days.index(day)-1];pool=pool_map[signal]
        positions=lagged[day]['holdings'];equity=lagged[day]['equity']
        prior=close.loc[signal].copy();current=close.loc[day].copy()
        ex=actions.loc[actions.ex_date.eq(pd.Timestamp(day))]
        stock=ex.groupby('symbol').stock_ratio.sum();cash=ex.groupby('symbol').cash_per_share.sum()
        returns=economic_daily_return(prior,current,stock,cash)
        basic=pd.read_parquet(STUDY/'official'/('daily_basic-'+signal+'.parquet')).set_index('ts_code')
        scope=sorted(set(pool)|set(positions));history=adjusted.loc[:signal].tail(121).reindex(columns=scope)
        features,_=named_price_risk_exposures(history,basic.total_mv.reindex(scope)*10000)
        features=features[['log_size','momentum60','volatility20','market_beta120','residual_volatility120']]
        finance=common_period_features(financial,signal).set_index('ts_code')
        additional={'book_to_price':signed_inverse(basic.pb),'earnings_yield':signed_inverse(basic.pe_ttm),
            'dividend_yield':basic.dv_ttm,'quality_roe':finance.roe,'growth_sales_yoy':finance.q_sales_yoy}
        for name,series in additional.items():
            values=series.reindex(scope).replace([np.inf,-np.inf],np.nan)
            # Matching-only neutral rank plus an explicit missingness dimension;
            # never impute raw missing accounting values as genuine zero earnings.
            features[name]=values.rank(pct=True).fillna(.5)
            features['missing__'+name]=values.isna().astype(float)
        industries=industry_as_of(industry,signal).l1_code
        for n in samples:
            matched=matched_day([{'symbol':s,'quantity':q} for s,q in positions.items()],equity,row['actual_return'],row['fees']+row['slippage'],prior,returns,features,pool,n,industries,allow_partial=True)
            if matched['status'] not in ['MATCHED','PARTIALLY_MATCHED']:
                costs=-(row['fees']+row['slippage'])/equity
                matched={**matched,'actual_return':row['actual_return'],**{k:0. for k in names},'costs':costs,'trading_and_other':row['actual_return']-costs,
                    'prior_exposure':0.,'total_known_prior_exposure':None,'unmatched':[]}
            matched['date']=day;samples[n].append(matched)
        dump(OUT/(kind+'-peer-progress.json'),{'completed':i+1,'total':len(daily),'latest_date':day})
    result={}
    for n,rows in samples.items():
        linked=linked_attribution(rows,names);assert abs(linked['reconciliation_error'])<1e-9
        pd.DataFrame([{k:v for k,v in r.items() if k not in ['matches','unmatched']} for r in rows]).to_parquet(OUT/(kind+f'-peers-{n}-daily.parquet'),index=False)
        dump(OUT/(kind+f'-peers-{n}-matches.json'),rows)
        known=sum((r.get('total_known_prior_exposure') or r.get('prior_exposure') or 0) for r in rows)
        matched_weight=sum((r.get('matched_prior_exposure') or r.get('prior_exposure') or 0) for r in rows)
        result[str(n)]={'linked_contribution_pp':{k:v*100 for k,v in linked.items()},'statuses':dict(Counter(r['status'] for r in rows)),
            'average_matched_prior_exposure':matched_weight/len(rows),'matched_fraction_of_known_exposure':matched_weight/known if known else None,
            'interpretation':'lagged same industry + listing board peers; size/momentum/volatility/beta/residual-vol plus valuation/dividend/quality/growth and explicit missing flags; descriptive, not causal alpha; unmatched and trading residual kept explicit'}
    dump(OUT/(kind+'-peers.json'),result);return result

def signal_diagnostics(data):
    industry=data[4]
    source=ROOT/'state/mature-multifactor/p1-native-20261103-a'
    scores=pd.read_parquet(STUDY/'model/predictions.parquet').score
    labels=pd.read_parquet(source/'alpha158-panel.parquet').economic_label
    calendar=pd.DatetimeIndex(pd.to_datetime((source/'provider/calendars/day.txt').read_text().splitlines()))
    last=calendar[calendar<=pd.Timestamp('20260930')][-6]
    frame=pd.concat([scores.rename('score'),labels.reindex(scores.index).rename('label')],axis=1).dropna()
    frame=frame.loc[frame.index.get_level_values('datetime')<=last]
    rows=[]
    for date,g in frame.groupby(level='datetime'):
        symbols=g.index.get_level_values('instrument')
        sector=industry_as_of(industry,date.strftime('%Y%m%d')).l1_code.reindex(symbols)
        group=sector.astype(str)+'_'+pd.Series(['STAR' if s.startswith('68') else 'CHINEXT' if s.startswith('30') else 'MAIN' for s in symbols],index=symbols)
        groups=pd.Series(group.to_numpy(),index=g.index)
        allowed=pd.Series(sector.notna().to_numpy(),index=g.index)&groups.map(groups.value_counts()).ge(5)
        valid=g.loc[allowed];within=groups.loc[allowed]
        ranks=valid.rank(pct=True);residuals=ranks-ranks.groupby(within).transform('mean')
        rows.append({'date':date.strftime('%Y%m%d'),'rank_ic':float(g.score.corr(g.label,method='spearman')),
            'same_subset_rank_ic':float(valid.score.corr(valid.label,method='spearman')),
            'industry_board_partial_rank_correlation':float(residuals.score.corr(residuals.label)),
            'neutralized_rows':len(valid),'total_rows':len(g)})
    d=pd.DataFrame(rows);d.to_parquet(OUT/'signal-quality-daily.parquet',index=False)
    result={'mature_signal_days':len(d),'last_mature_signal':str(last.date()),'mean_rank_ic':float(d.rank_ic.mean()),
        'same_subset_mean_rank_ic':float(d.same_subset_rank_ic.mean()),'mean_industry_board_partial_rank_correlation':float(d.industry_board_partial_rank_correlation.mean()),
        'daily_positive_rank_ic_fraction':float(d.rank_ic.gt(0).mean()),'neutralized_row_coverage':float(d.neutralized_rows.sum()/d.total_rows.sum()),
        'method':'5-session economic label; per-day rank residuals after industry+listing-board group means; minimum group 5; not full-style or causal alpha; overlapping labels not independent'}
    dump(OUT/'signal-quality.json',result);print(json.dumps(result));return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--exact-only',action='store_true');args=parser.parse_args()
    with threadpool_limits(limits=4):
        data=inputs();results={}
        for kind in ['baseline','v2','raw','balanced']:
            result,lagged,daily=exact(kind,data)
            if kind in ['raw','balanced'] and not args.exact_only:result['peer_diagnosis']=peers(kind,data,lagged,daily)
            results[kind]=result
            print(json.dumps({'strategy':kind,'net_pnl':result['net_pnl'],'fees':result['fees'],'slippage':result['slippage'],'max_pnl_error':result['maximum_daily_pnl_error']}),flush=True)
        dump(OUT/('exact-summary.json' if args.exact_only else 'results.json'),{'status':'EXACT_LEDGER_AND_DESCRIPTIVE_ATTRIBUTION','strategies':results,'production_modified':False,
            'limits':['historical membership includes carried snapshots','cost addback holds actual quantities fixed','no causal claim from protection-day grouping','peer comparisons are descriptive with explicit unmatched exposures']})
