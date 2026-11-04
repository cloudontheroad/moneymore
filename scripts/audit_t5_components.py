"""Independent exact PnL bridge, fees and decision counts for the frozen grid."""
import json,hashlib
from collections import Counter
import pandas as pd
from threadpoolctl import threadpool_limits
import attribute_four_strategy_2026 as attribution
from prepare_four_strategy_2026 import ROOT,save
OUT=ROOT/'state/cost-aware-ab-v2'

def main(partial=False):
    if partial:
        grid={'cases':[],'calibration':json.loads((OUT/'t2/historical.json').read_bytes())}
        for p in sorted((OUT/'t5').glob('*/results.json')):grid['cases'].extend(json.loads(p.read_bytes())['cases'])
        assert grid['cases']
    else:
        grid=json.loads((OUT/'t5-results.json').read_bytes());assert len(grid['cases'])==6
    frozen=json.loads((OUT/'t5-frozen-inputs.json').read_bytes())
    for p,h in frozen['hashes'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,('FROZEN_STUDY_CHANGED',p)
    attribution.OUT=OUT/'t5-audit';attribution.OUT.mkdir(exist_ok=True)
    data=attribution.inputs();bars=data[0]
    for p in sorted((OUT/'quote-cache').glob('*.parquet')):bars=pd.concat([bars,pd.read_parquet(p)],ignore_index=True).drop_duplicates(['trade_date','ts_code'],keep='first')
    opening=bars.set_index(['trade_date','ts_code']).open
    close=bars.pivot(index='trade_date',columns='ts_code',values='close').sort_index()
    adjusted=bars.assign(adjusted=bars.close*bars.adj_factor).pivot(index='trade_date',columns='ts_code',values='adjusted').sort_index()
    data=(bars,*data[1:5],close,opening,adjusted);rows=[]
    for case in grid['cases']:
        kind=case['strategy'];folder=OUT/'t5'/kind
        exact,_,_=attribution.exact(kind,data,folder=folder)
        events=json.loads((folder/'checkpoint.json').read_bytes())['events'][kind]
        assert exact['maximum_daily_pnl_error']<.01 and exact['maximum_cash_error']<.01
        assert abs(exact['net_pnl']/1e7-case['net_return'])<1e-10
        rows.append({**case,'total_friction':exact['fees']+exact['slippage'],'slippage':exact['slippage'],
            'gross_same_quantities_cost_addback':exact['gross_cost_addback']/1e7,'notional_multiple':exact['turnover_total_notional_multiple'],
            'decisions':dict(Counter(e.get('cost_decision') or e['status'] for e in events)),
            'days_with_orders':sum(e.get('orders',0)>0 for e in events),'blocked_reasons':exact['protection_reasons'],
            'maximum_pnl_error':exact['maximum_daily_pnl_error']})
    by={r['strategy']:r for r in rows};deltas=[]
    for first,second in [('a_reference','a_cost'),('b_reference','b_exec'),('b_reference','b_cost'),('b_reference','b_cost_exec'),('b_exec','b_cost_exec')]:
        if first not in by or second not in by:continue
        a,b=by[first],by[second]
        deltas.append({'reference':first,'version':second,'net_return_difference_pp':100*(b['net_return']-a['net_return']),
            'saved_friction':a['total_friction']-b['total_friction'],'fill_difference':b['fills']-a['fills']})
    result={'status':'PARTIAL_LEDGER_AUDIT_NOT_FINAL' if partial else 'SIX_COMPONENT_LEDGER_AUDIT_PASS','cases':rows,'deltas':deltas,
        'timeline':'prior_close_plan_next_open_execution','cost_addback_is_not_a_reoptimized_no_cost_strategy':True,
        'calibrated_alpha_enabled':grid['calibration']['optional_alpha_enabled'],'profit_proven':False,'real_orders_enabled':False}
    initial=OUT/'t5/b_reference/b_reference/20260105-continuous.json'
    if initial.exists():
        reference=json.loads(initial.read_bytes());wb=pd.Series(reference['benchmark_weights'])
        seed=json.loads((initial.parent/'20260105-integer.json').read_bytes())
        quantity=pd.Series(seed['target_quantities']);values=quantity*opening.loc['20260105'].reindex(quantity.index)
        active=(values/values.sum()).reindex(wb.index,fill_value=0)-wb
        result['shared_initial_setup']={'date':'20260105','new_incremental_policy_first_rebalance':'20260106',
            'legacy_reference_shares_copied_for_fair_start':True,'legacy_initial_stock_active_bound_excess':float(max(0,abs(active).max()-(.15-wb.max()))),
            'not_a_new_policy_bootstrap_risk_pass':True,'live_new_account_bootstrap_separately_verified':True}
    save(OUT/('t5-audit/partial-results.json' if partial else 't5-audit/results.json'),result)
    if partial:
        print(json.dumps(result,ensure_ascii=True),flush=True);return
    columns=['strategy','net_return','annualized_volatility','sharpe','maximum_drawdown','fills','total_friction','days_with_orders']
    lines=['# 成本感知 A/B：固定组件终验','','全部181日、同一冻结预测、1000万元、10bps单边滑点。前收盘规划、次开盘执行；不与旧异口径回放相减。',
        '','校准未证明可靠交易优势，改版属于成本/风险保护，不是新盈利模型。','','| 策略 | 收益 | 年化波动 | 夏普 | 最大回撤 | 成交 | 总摩擦(元) | 有成交日 |','|---|---|---|---|---|---|---|---|']
    for r in rows:lines.append(f"| {r['strategy']} | {r['net_return']:.2%} | {r['annualized_volatility']:.2%} | {r['sharpe']:.3f} | {r['maximum_drawdown']:.2%} | {r['fills']} | {r['total_friction']:.2f} | {r['days_with_orders']} |")
    lines+=['','首日采用相同的旧版参考配置，改版增量调仓自下一日开始。参考风险组初始主动个股权重存在约1.82bp离散误差，不能把共享初始配置说成新版本初建风控已通过；正式新模拟的初建另经严格整数风险预检。',
        '完整逐日/逐股会计桥接：state/cost-aware-ab-v2/t5-audit。残余阻塞按原因保留，不伪造成交。','成员历史含延续快照，财务历史非完整原始版本认证；上述为回顾性研究，不是前瞻收益。']
    (ROOT/'docs/research/COST_AWARE_AB_FINAL_20261104.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True),flush=True)
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    with threadpool_limits(limits=4):main(args.partial)
