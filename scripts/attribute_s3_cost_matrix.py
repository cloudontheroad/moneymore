"""Read-only S3 cost and existing matched-style attribution; no winner adoption."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import yaml
from moneymore.data.store import ParquetStore
from moneymore.challenger_diagnostics import trading_dates
from moneymore.challenger_attribution import matched_day,linked_attribution
from moneymore.research.replay_attribution import reconstruct_positions
from moneymore.research.held_evidence import read_held_basic
from moneymore.research.style_price_features import named_price_risk_exposures
from moneymore.research.style_missing_controls import uncertain_controls
from moneymore.research.style_risk_inputs import signed_inverse
from moneymore.research.baseline_v2_industry import industry_as_of
from moneymore.research.fixed_order_cost import counterfactual_cash_delta


def main():
    p=argparse.ArgumentParser();p.add_argument('--matrix',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-partial-accounting',action='store_true');args=p.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_COST_ATTRIBUTION')
    manifest=args.matrix/'cost-matrix-complete.json'
    progress=json.loads((manifest if manifest.exists() else args.matrix/'progress.json').read_text())
    if progress['status']!='COST_MATRIX_EXECUTED_REQUIRES_ATTRIBUTION_AND_RELEASE_REVIEW' or len(progress['studies'])!=8:raise ValueError('EIGHT_FIXED_STUDIES_REQUIRED')
    root=Path(__file__).resolve().parents[1];store=ParquetStore(root/'data');calendar=trading_dates(store,'20251231')
    audit=json.loads((root/'state/style-balance/s1-feature-paths-20261102-a/audit.json').read_text());pool=pd.read_parquet(audit['historical_pool']['path']);pool['signal_date']=pool.signal_date.astype(str)
    if hashlib.sha256(Path(audit['historical_pool']['path']).read_bytes()).hexdigest()!=audit['historical_pool']['sha256']:raise ValueError('DATED_POOL_CHANGED')
    membership=store.read('baseline_v2_industry_membership');args.output.mkdir(parents=True);cases=[];bindings=[];market_cache={}
    for study_row in progress['studies']:
        study=Path(study_row['path']);protocol=yaml.safe_load((study/'protocol.yaml').read_text());source=json.loads((study/'results.json').read_text())
        binding=json.loads((study/'episode-binding.json').read_text()) if (study/'episode-binding.json').exists() else {'fold':0,'evidence':str(root/'state/style-balance/s3-cost-evidence-fold0-20261103-a')}
        fold=binding['fold'];basic_folder=root/f'state/style-balance/s3-attribution-basic-fold{fold}-20261103-a';basic=read_held_basic(basic_folder)
        bindings.append({'study':str(study),'results_sha256':hashlib.sha256((study/'results.json').read_bytes()).hexdigest(),
                         'ledger_audit_sha256':hashlib.sha256((study/'independent-audit.json').read_bytes()).hexdigest(),
                         'basic_manifest_sha256':hashlib.sha256((basic_folder/'manifest.json').read_bytes()).hexdigest()})
        market=pd.read_parquet(study/'market-history.parquet');dates=pd.read_parquet(study/source['cases'][0]['account']/'equity.parquet').date.astype(str).tolist()
        signals=[calendar[calendar.index(d)-1] for d in dates];symbols=sorted(pool.loc[pool.signal_date.isin(signals),'ts_code'].unique())
        # Full dated market pool, including assets without 120-day model features.
        if fold not in market_cache:
            extra=store.read('daily',columns=['ts_code','trade_date','close'],filters=[('ts_code','in',symbols),('trade_date','>=',market.trade_date.min()),('trade_date','<=',dates[-1])])
            extra=extra.merge(store.read('adj_factor',columns=['ts_code','trade_date','adj_factor'],filters=[('ts_code','in',symbols),('trade_date','>=',market.trade_date.min()),('trade_date','<=',dates[-1])]),on=['ts_code','trade_date'],validate='one_to_one')
            extra['total_price']=extra.close*extra.adj_factor;market_cache[fold]=extra
            extra.to_parquet(args.output/f'fold{fold}-market-input.parquet',index=False)
        extra=market_cache[fold]
        bindings[-1]['market_input_sha256']=hashlib.sha256((args.output/f'fold{fold}-market-input.parquet').read_bytes()).hexdigest()
        grid=[d for d in calendar if market.trade_date.min()<=d<=dates[-1]]
        raw_prices=extra.pivot(index='trade_date',columns='ts_code',values='close').reindex(grid)
        total_prices=extra.pivot(index='trade_date',columns='ts_code',values='total_price').reindex(grid)
        unfilled_risk_prices=total_prices.copy()
        checked=Path(binding.get('audit',str(root/f'state/style-balance/s3-cost-evidence-audit-fold{fold}-20261103-a')))
        actions=pd.read_parquet(checked/'official-actions.parquet')
        for day in dates:
            susp=pd.read_parquet(Path(binding['evidence'])/f'suspend_d-{day}.parquet')
            full=set(susp.loc[susp.suspend_type.eq('S')&(susp.suspend_timing.isna()|susp.suspend_timing.eq('')),'ts_code']) if len(susp) else set()
            prior=calendar[calendar.index(day)-1]
            for symbol in full&set(total_prices.columns):
                if pd.isna(total_prices.loc[day,symbol]) and pd.notna(total_prices.loc[prior,symbol]):total_prices.loc[day,symbol]=total_prices.loc[prior,symbol]
                if pd.isna(raw_prices.loc[day,symbol]) and pd.notna(raw_prices.loc[prior,symbol]):
                    value=raw_prices.loc[prior,symbol]
                    for action in actions.loc[actions.symbol.eq(symbol)&actions.ex_date.eq(pd.Timestamp(day))].itertuples():value=(value-action.cash_per_share)/(1+action.stock_ratio)
                    raw_prices.loc[day,symbol]=value
        contexts={}
        for case in source['cases']:
            account=case['account'];folder=study/account;eq=pd.read_parquet(folder/'equity.parquet').set_index('date')
            database=folder/'paper.sqlite3';digest=hashlib.sha256(database.read_bytes()).hexdigest()
            with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as con:
                fills=pd.read_sql_query('SELECT * FROM fills',con);ledger=pd.read_sql_query('SELECT * FROM corporate_action_ledger',con)
            positions=reconstruct_positions(fills,ledger,dates);days_by_neighbor={n:[] for n in [3,5,10]};prior_equity=float(protocol['capital_each_episode']);board_rows=[]
            pure_cost=None
            if protocol['slippage_bps_each_side']==10:
                cumulative=0.;counterfactual=[];reference=market.set_index(['trade_date','ts_code']).open
                for day in dates:
                    for fill in fills.loc[fills.trade_date.astype(str).eq(day)].to_dict('records'):
                        delta=counterfactual_cash_delta(fill,float(reference.loc[(day,fill['symbol'])]),protocol,2)
                        if delta<-1e-6:raise ValueError('LOWER_FRICTION_REDUCED_CASH_UNEXPECTEDLY')
                        cumulative+=delta
                    counterfactual.append({'date':day,'cash_saved':cumulative,'equity':float(eq.loc[day,'equity'])+cumulative})
                cf=pd.DataFrame(counterfactual);cf.to_parquet(args.output/f'fold{fold}-{account}-same-orders-2bps.parquet',index=False)
                pure_cost={'same_actual_order_schedule':True,'new_orders_or_predictions':False,'cash_saved':cumulative,
                    'net_return_2bps':float(cf.equity.iloc[-1]/protocol['capital_each_episode']-1),
                    'net_return_10bps':case['net_return'],'description':'held quantities and corporate rights unchanged; savings remain cash, no fictitious new fills'}
            for day in dates:
                prior=calendar[calendar.index(day)-1];held=positions.get(prior,{})
                current=pool.loc[pool.signal_date.eq(prior)].set_index('ts_code');ix=current.index.union(pd.Index(held));key=(prior,tuple(sorted(ix)))
                if key not in contexts:
                    attrs=basic.loc[basic.trade_date.astype(str).eq(prior)].set_index('ts_code').reindex(ix)
                    history=unfilled_risk_prices.loc[:prior].tail(121).reindex(columns=ix)
                    raw,_=named_price_risk_exposures(history,attrs.total_mv*10000)
                    raw['book_to_price']=signed_inverse(attrs.pb);raw['earnings_yield']=signed_inverse(attrs.pe_ttm);raw['dividend_yield']=attrs.dv_ttm/100
                    raw['industry']=ix.map(industry_as_of(membership,prior).l1_code).fillna('UNKNOWN');raw['listing_board']=['STAR' if s.startswith('68') else 'CHINEXT' if s.startswith('30') else 'MAIN' for s in ix]
                    c,eligible,_=uncertain_controls(raw,['log_size','momentum60','volatility20','market_beta120','residual_volatility120'])
                    contexts[key]=(c,raw.industry)
                c,industries=contexts[key];actual=float(eq.loc[day,'equity']/prior_equity-1);fee=float(fills.loc[fills.trade_date.astype(str).eq(day),'fee'].sum())
                returns=total_prices.loc[day]/total_prices.loc[prior]-1
                held_rows=[{'symbol':s,'quantity':q} for s,q in held.items()]
                for neighbors in [3,5,10]:
                    result=matched_day(held_rows,prior_equity,actual,fee,raw_prices.loc[prior],returns,c,current.index.tolist(),neighbors,industries,args.allow_partial_accounting)
                    days_by_neighbor[neighbors].append(dict(date=day,**result))
                now=positions[day];values=pd.Series(now,dtype=float)*raw_prices.loc[day].reindex(now)
                known=float(values.sum());tech=float(values.loc[values.index.str.startswith(('30','68'))].sum()) if len(values) else 0.
                complete_marks=bool(values.notna().all())
                board_rows.append({'date':day,'dual_board_stock_sleeve_weight':(tech/known if known else 0.) if complete_marks else None,
                    'raw_quote_coverage':int(values.notna().sum()),'held_names':len(now)})
                prior_equity=float(eq.loc[day,'equity'])
            sensitivity={}
            for n,rows in days_by_neighbor.items():
                complete=all(r['status']=='MATCHED' for r in rows)
                accounting_complete=all(r['status'] in ['MATCHED','PARTIALLY_MATCHED'] for r in rows)
                coverage=[r['matched_prior_exposure']/r['total_known_prior_exposure'] for r in rows if r.get('valuation_complete') and r.get('total_known_prior_exposure',0)>0]
                components=('market','style','selection','unmatched_stock','trading_and_other','costs') if args.allow_partial_accounting else None
                sensitivity[str(n)]={'matched_days':sum(r['status']=='MATCHED' for r in rows),'total_days':len(rows),'complete':complete,
                    'accounting_complete':accounting_complete,'matched_capital_coverage_mean':float(np.mean(coverage)) if coverage else None,
                    'linked':linked_attribution(rows,components) if accounting_complete else None,
                    'unmatched':[{'date':r['date'],'reason':r['status'],'symbol':r.get('symbol'),'positions':r.get('unmatched',[])} for r in rows if r['status']!='MATCHED']}
                (args.output/f'fold{fold}-{protocol["slippage_bps_each_side"]}bps-{account}-matches-{n}.json').write_text(json.dumps(rows,allow_nan=False))
            if hashlib.sha256(database.read_bytes()).hexdigest()!=digest:raise ValueError('SOURCE_LEDGER_CHANGED')
            cases.append({'fold':fold,'slippage_bps':protocol['slippage_bps_each_side'],'account':account,'model':case['model'],'variant':case['variant'],
                'net_return':case['net_return'],'max_drawdown':case['max_drawdown'],'fees':case['fees'],'fills':case['fills'],
                'passed_rebalances':case['passed_rebalances'],'planned_rebalances':case['planned_rebalances'],'attribution':sensitivity,'board_exposure':board_rows,
                'pure_cost_counterfactual':pure_cost})
        print(json.dumps({'completed_study':str(study),'cases':len(cases)}),flush=True)
    if len(cases)!=48:raise ValueError('FORTY_EIGHT_ACCOUNTS_REQUIRED')
    output={'status':'COST_AND_MATCHED_STYLE_REVIEW_NOT_RELEASE_APPROVAL','cases':cases,'bindings':bindings,'orders_enabled':False,'winner_adopted':False,
        'partial_accounting_enabled':args.allow_partial_accounting,'risk_history_price_filled':False,'only_official_halts_valuation_carried':True,
        'limitations':['four_short_exploratory_episodes_not_full_continuous_history','matched_selection_is_not_causal_alpha',
            'trading_and_other_includes_intraday_timing_and_corporate_rights','partial_accounting_bridge_is_not_full_style_attribution_or_causal_alpha',
            'matched_selection_only_applies_to_matched_capital_unmatched_stock_proxy_is_separate_not_alpha',
            '2_vs_10bps_return_difference_includes_changed_execution_paths_not_only_fee_savings','board_exposure_uses_observed_raw_marks_and_discloses_coverage']}
    (args.output/'results.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    pd.DataFrame([{k:v for k,v in c.items() if k not in ['attribution','board_exposure','pure_cost_counterfactual']} for c in cases]).to_parquet(args.output/'cost-comparison.parquet',index=False)


if __name__=='__main__':main()
