"""Independent N4 cash/share audits, paired outcomes and transparent source evidence."""
import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from moneymore.research.ashare_mainline import sha256, prediction_diagnostics
from moneymore.research.period_statistics import period_statistics
from report_ashare_mainline_replay import audit_ledger
from report_ashare_rank_preserving import assert_exact_replay_equal, daily_returns
from finalize_ashare_mainline_n3 import paired_intervals
from diagnose_ashare_rank_turnover import attach_industry, json_safe, position_trade_partition
from prepare_four_strategy_2026 import save

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'state/ashare-mainline-20261112'
SOURCES=['doubleensemble_full','fundamental','trend','short']


def attribution(folder, case, equity, fills, holdings, panel):
    with sqlite3.connect((folder/case/'paper.sqlite3').resolve().as_uri()+'?mode=ro',uri=True) as db:
        ledger=pd.read_sql_query('SELECT * FROM corporate_action_ledger',db)
    partition,quantities=position_trade_partition(fills,ledger,holdings,equity.date.tolist())
    values=holdings.merge(equity[['date','equity']],on='date')
    values['value']=values.weight*values.equity
    values=values.pivot(index='date',columns='symbol',values='value').reindex(equity.date).fillna(0)
    symbols=sorted(set(values.columns)|set(fills.symbol)|set(ledger.symbol))
    values=values.reindex(columns=symbols,fill_value=0)
    flows=pd.DataFrame(0.,index=values.index,columns=symbols)
    for r in fills.itertuples(): flows.loc[r.date,r.symbol]+=r.notional*(1 if r.side=='SELL' else -1)-r.fee
    for r in ledger.itertuples():
        if r.trade_date in flows.index: flows.loc[r.trade_date,r.symbol]+=r.cash_amount
    pnl=values-values.shift(1,fill_value=0)+flows
    daily=equity.set_index('date')
    navchange=daily.equity-daily.equity.shift(1).fillna(10_000_000.)
    residual=navchange-pnl.sum(axis=1)
    long=pnl.rename_axis(index='date',columns='symbol').stack().rename('net_pnl').reset_index()
    long=attach_industry(long,panel)
    industry=long.groupby('industry').net_pnl.sum().sort_values()
    stock=long.groupby('symbol').net_pnl.sum().sort_values()
    return {'max_daily_unattributed_rights_or_receivable_residual':float(residual.abs().max()),'total_unattributed_residual':float(residual.sum()),'industry_net_pnl':industry.to_dict(),'bottom10_stock_pnl':stock.head(10).to_dict(),'top10_stock_pnl':stock.tail(10).to_dict(),'trade_categories':partition.groupby('trade_category').agg(fills=('quantity','count'),notional=('notional','sum'),fees=('fee','sum'),slippage=('slippage','sum')).to_dict('index'),'cash_share_daily_positions_reconstructed':True}


def main(runs, output):
    output.mkdir(parents=True,exist_ok=False)
    config=json.loads((ROOT/'configs/ashare_n4_v1.json').read_bytes())
    training=json.loads((BASE/'n4-training-v1/results.json').read_bytes())
    if training['status']!='COMPLETE' or len(training['results'])!=18 or training['config_sha256']!=sha256(ROOT/'configs/ashare_n4_v1.json') or training['prediction_sha256']!=sha256(BASE/'n4-training-v1/predictions.parquet'):
        raise ValueError('INCOMPLETE_OR_CHANGED_TRAINING')
    for trained in training['results']:
        tc=trained['time_contract']
        if not tc['train']['last_label_end']<tc['valid']['first'] or not tc['valid']['last_label_end']<tc['test']['first'] or trained['reload_error']>1e-10:
            raise ValueError('TRAINING_TIME_OR_RELOAD_CONTRACT')
    panel=pd.read_parquet(BASE/'panel-v2/panel.parquet',columns=['date','symbol','industry'])
    rows,eqmap,checks,correlations,allocations=[],{},{},{},{}
    for bps in config['slippage_bps']:
        for period in ['history','evaluation','blends']:
            run=runs/f'{period}-cost{bps}'
            if not (run/'completed.json').exists(): raise ValueError('INCOMPLETE_RUN:'+str(run))
            manifest=json.loads((run/'completed.json').read_bytes())
            for p,h in {**manifest['source_hashes'],**manifest.get('intent_hashes',{})}.items():
                if sha256(Path(p))!=h: raise ValueError('REPLAY_INPUT_OR_SOURCE_CHANGED:'+p)
            folder=run/f'bps{bps}'
            result=json.loads((folder/'results.json').read_bytes())
            if result['smoke'] or not result['all_daily_reconciliations_pass']: raise ValueError('NOT_FULL_REPLAY')
            holding_all=pd.read_parquet(folder/'holdings.parquet')
            expected=[s+'--rank_cap' for s in SOURCES] if period!='blends' else config['allocation']['methods']
            if {r['strategy'] for r in result['cases']}!=set(expected): raise ValueError('CASE_MATRIX_INCOMPLETE')
            for case in expected:
                equity=pd.read_parquet(folder/case/'equity.parquet')
                fills=pd.read_parquet(folder/case/'fills.parquet')
                required=126 if period=='history' else 181
                if len(equity)!=required or equity.date.duplicated().any() or not equity.date.is_monotonic_increasing: raise ValueError('INCOMPLETE_CALENDAR')
                ledger=audit_ledger(folder/case/'paper.sqlite3',fills)
                stats=period_statistics(equity.equity,config['initial_cash'])
                engine_row=next(r for r in result['cases'] if r['strategy']==case)
                for field in ['net_return','sharpe','maximum_drawdown','annualized_volatility']:
                    if not np.isclose(stats[field],engine_row[field],atol=1e-10): raise ValueError('METRIC_MISMATCH')
                holdings=holding_all.loc[holding_all.kind.eq(case)]
                style=attach_industry(holdings,panel)
                if style.industry.eq('UNKNOWN').any(): raise ValueError('UNKNOWN_INDUSTRY')
                sectors=style.groupby(['date','industry']).weight.sum()
                previous=equity.set_index('date').equity.shift(1).fillna(config['initial_cash'])
                turnover=(fills.groupby('date').notional.sum().reindex(equity.date,fill_value=0)/previous/2).sum()*252/len(equity)
                decisions=[json.loads(p.read_bytes()) for p in sorted((run/'decisions'/case).glob('*.json'))]
                if len(decisions)!=len(equity): raise ValueError('MISSING_DECISIONS')
                for d in decisions:
                    if d['signal_date']>=d['execution_date']: raise ValueError('PLAN_DATE_LOOKAHEAD')
                    if d['status']=='OPEN_PLAN_VERIFIED':
                        if d.get('next_open_used_for_selection') is not False or not d['open_postcheck']['passed']: raise ValueError('OPEN_PLAN_AUDIT_FAILURE')
                        post=d['open_postcheck']
                        if post['maximum_stock_weight']>config['max_stock_weight']+1e-5 or any(w>post['industry_limits'][s]+1e-5 for s,w in post['industry_weights'].items()): raise ValueError('INDEPENDENT_POST_LIMIT_FAILURE')
                        if period=='blends' and (sum(d['target_weights'].values())>config['stock_exposure']+1e-9 or max(d['target_weights'].values(),default=0)>config['max_stock_weight']+1e-9): raise ValueError('AGGREGATE_TARGET_BUDGET')
                        if d['risk_audit']['risk_end']>d['signal_date']: raise ValueError('RISK_LOOKAHEAD')
                        if period=='blends' and (d['allocation']['return_end']>d['allocation']['signal_date'] or d['allocation']['signal_date']>=d['execution_date']): raise ValueError('ALLOCATION_LOOKAHEAD')
                row={'period':period,'case':case,'bps':bps,**stats,'mean_exposure':float(equity.exposure.mean()),'minimum_exposure':float(equity.exposure.min()),'mean_names':float(holdings.groupby('date').size().reindex(equity.date,fill_value=0).mean()),'mean_largest_industry':float(sectors.groupby('date').max().reindex(equity.date,fill_value=0).mean()),'max_industry':float(sectors.max()),'industry_mean_weights':(style.groupby('industry').weight.sum()/len(equity)).to_dict(),'growth_board_mean_weight':float(style.loc[style.symbol.str.startswith(('30','68'))].weight.sum()/len(equity)),'turnover_annual_half_gross':float(turnover),'fees':float(fills.fee.sum()),'slippage':float(fills.slippage.sum()),'fills':len(fills),'trading_days':int(equity.fills.gt(0).sum()),'quarters':engine_row['quarters'],'decision_reasons':dict(Counter(d['reason'] for d in decisions if d.get('reason'))),'ledger':ledger}
                row['netting']={k:sum(d.get('netting',{}).get(k,0) for d in decisions) for k in ['virtual_gross_notional','net_notional_before_lot_cash_and_permissions','crossed_notional_both_sides']}
                row['attribution']=attribution(folder,case,equity,fills,holdings,panel)
                rows.append(row)
                eqmap[(period,bps,case)]=equity
                if period=='evaluation' and case==SOURCES[0]+'--rank_cap':
                    for name in ['equity.parquet','fills.parquet']:
                        reference=BASE/f'n3r-rank-v2/cost{bps}/bps{bps}'/case/name
                        current=folder/case/name
                        assert_exact_replay_equal(pd.read_parquet(current),pd.read_parquet(reference),f'N4:{bps}:{name}')
                        checks[f'{bps}:{name}']={'exact_equal':True,'binary_equal':sha256(current)==sha256(reference)}
        all_returns={}
        for case in [s+'--rank_cap' for s in SOURCES]+config['allocation']['methods']:
            period='evaluation' if '--' in case else 'blends'
            all_returns[case]=daily_returns(eqmap[(period,bps,case)],config['initial_cash'])
        returns=pd.DataFrame(all_returns)
        core=returns[SOURCES[0]+'--rank_cap']
        residuals=returns.apply(lambda s:s-np.cov(s,core,ddof=1)[0,1]/core.var(ddof=1)*core)
        correlations[bps]={'pearson':returns.corr().to_dict(),'core_down_days':returns.loc[core<0].corr().to_dict(),'core_beta_residual_correlation_diagnostic_only':residuals.drop(columns=[SOURCES[0]+'--rank_cap']).corr().to_dict()}
        allocation=json.loads((runs/f'allocation-cost{bps}.json').read_bytes())
        allocations[bps]=allocation['records']
    comparisons=[]
    for row in rows:
        if row['period']=='history': continue
        bps=row['bps']
        ref=next(r for r in rows if r['period']=='evaluation' and r['bps']==bps and r['case']==SOURCES[0]+'--rank_cap')
        if row['case']==ref['case']:continue
        left=daily_returns(eqmap[(row['period'],bps,row['case'])],config['initial_cash'])
        right=daily_returns(eqmap[('evaluation',bps,ref['case'])],config['initial_cash'])
        comparisons.append({'case':row['case'],'bps':bps,'return_difference':row['net_return']-ref['net_return'],'sharpe_difference':row['sharpe']-ref['sharpe'],'drawdown_difference':row['maximum_drawdown']-ref['maximum_drawdown'],'cost_difference':row['fees']+row['slippage']-ref['fees']-ref['slippage'],**paired_intervals(left,right)})
    predictions=pd.read_parquet(BASE/'n4-training-v1/predictions.parquet')
    diagnostics=[]
    for case,g in predictions.groupby('case'):
        for year in ['2025','2026']:
            f=g.loc[g.date.str.startswith(year)]
            daily=prediction_diagnostics(f,'20251231' if year=='2025' else '20260930')
            diagnostics.append({'case':case,'year':year,'horizon':int(f.horizon.iloc[0]),'mature_days':int(daily.rank_ic.notna().sum()),'rank_ic':float(daily.rank_ic.mean()),'min_unique_scores':int(daily.distinct_scores.min())})
    core_predictions=pd.read_parquet(BASE/'n2-summary-v1/predictions.parquet')
    core_predictions=core_predictions.loc[core_predictions.case.eq(SOURCES[0]),['case','date','symbol','score']]
    all_scores=pd.concat([predictions[['case','date','symbol','score']],core_predictions],ignore_index=True)
    scores=all_scores.loc[all_scores.date.between('20260101','20260930')].pivot(index=['date','symbol'],columns='case',values='score')
    signal_rows=[]
    proxies=pd.read_parquet(BASE/'panel-v2/panel.parquet',columns=['date','symbol','ep','mom6_1','ROC5']).set_index(['date','symbol'])
    for day,g in scores.groupby(level='date'):
        matched=g.dropna()
        corr=matched.corr(method='spearman')
        top={s:set(g[s].dropna().nlargest(50).index.get_level_values('symbol')) for s in SOURCES}
        proxy=proxies.reindex(g.index)
        signal_rows.append({'date':day,'score_correlations':corr.to_dict(),'top50_overlap_with_core':{s:len(top[s]&top[SOURCES[0]])/50 for s in SOURCES[1:]},'short_score_correlation_with_ROC5_reversal_proxy':float(g['short'].corr(proxy.ROC5,method='spearman')),'trend_score_correlation_with_mom6_1':float(g['trend'].corr(proxy.mom6_1,method='spearman')),'fundamental_score_correlation_with_EP':float(g['fundamental'].corr(proxy.ep,method='spearman'))})
    scope=json.loads((ROOT/'configs/ashare_n4_execution_scope_v1.json').read_bytes())
    payload=json_safe({'status':'N4_COMPLETE_ISOLATED_RESEARCH','cases':rows,'comparisons':comparisons,'correlations':correlations,'allocations':allocations,'prediction_diagnostics':diagnostics,'signal_independence':signal_rows,'core_exact_checks':checks,'execution_scope_limitation':scope,'production_modified':False,'historical_period_previously_examined':True})
    save(output/'results.json',payload)
    lines=['# N4成本后完整结果','','资本1000万元；2026-01-05—09-30，181交易日。单边滑点另加佣金、税费。历史研究，不是未来收益承诺。','','| 成本 | 策略 | 净收益 | 夏普 | 最大回撤 | 年化波动 | 平均仓位 | 费用＋滑点 |','|---|---|---|---|---|---|---|---|']
    for r in rows:
        if r['period']=='history':continue
        lines.append(f"| {r['bps']}bps | {r['case']} | {r['net_return']:.2%} | {r['sharpe']:.3f} | {r['maximum_drawdown']:.2%} | {r['annualized_volatility']:.2%} | {r['mean_exposure']:.2%} | {r['fees']+r['slippage']:,.0f}元 |")
    lines+=['','全部22个独立账户完成现金/股票账本核验；原core两档成本逐项精确复现。组合是真实股票净额撮合，理论跨来源抵消额不冒充实际节省费用。完整季度、相关性、持仓风格、归因、权重历史与不确定性见同目录results.json。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':payload['status'],'accounts':len(rows),'output':str(output)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    main(a.runs,a.output)
