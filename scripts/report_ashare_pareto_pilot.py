"""R4/R5 pilot report, explicitly excluding unimplemented production promotion."""
import hashlib,json
import numpy as np
import pandas as pd
from moneymore.execution.paper import PaperBroker
from moneymore.research.period_statistics import period_statistics
from moneymore.research.pareto_rebuild import pareto_front
from audit_ashare_pareto_inputs import ROOT,OUT

NAMES={'ridge_reference':'旧价量·每日','ep_buffer':'EP低换手·每日','r_old_ridge20':'旧价量·20日对照','r_value':'行业内价值质量代理','r_improvement':'行业内盈利改善代理','r_momentum':'行业内中期动量','r_equal':'三族等权信号','r_ridge_raw20':'滚动Ridge·原始20日目标','r_ridge_residual20':'滚动Ridge·残差20日目标','r_ridge_residual5':'滚动Ridge·残差5日目标','r_ridge_residual60':'滚动Ridge·残差60日目标','r_tree_residual20':'滚动浅树·残差20日目标','r_nonnegative_residual20':'非负组合·残差20日目标'}


def main():
    walk=json.loads((OUT/'walkforward/results.json').read_bytes())
    assert len(walk['fits'])==18
    for f in walk['fits']:
        assert f['last_label_end']<f['refit_date'] and f['last_label_end']<=f['prediction_start']
    panel=pd.read_parquet(OUT/'signals/panel.parquet')
    for field,value in [('roe_ann_date','roe'),('q_sales_yoy_ann_date','sales_yoy')]:
        known=pd.to_datetime(panel[field],format='%Y%m%d',errors='coerce')+pd.Timedelta(days=1)
        used=panel[value].notna()
        assert (known.loc[used]<=pd.to_datetime(panel.loc[used,'date'])).all()
    text=['# A股预测重构：阶段性可成交回放与帕累托结果','',
          'R0/R1/R2/R3已执行；本报告是R4的信号比较和R5评估器试运行，**尚未完成因子基线/V2的统一重放与Qlib风险约束组合，不是整个重构项目完成，也未发布生产策略。**','',
          '2026-01-05至09-30共181交易日，每账户1000万元。新组固定每20交易日检查一次常规换仓，每次最多替换5只，目标Top50；候选池退出每天处理。5/20/60日指训练目标期限，本轮执行节奏均为20日，避免同时改变两个变量。旧价量每日、EP缓冲每日、旧价量20日均保留为对照。','',
          '季度重拟合只使用当时已成熟标签，18次拟合均核验标签结束不晚于首个预测日。财报遵循公告日后可见规则。等权与非负组合均先合成信号、再生成股数和订单，不是平均旧净值。','']
    fronts={};audits=[];metrics={}
    for bps in [10,20]:
        folder=OUT/f'pilot-replays/bps{bps}';report=json.loads((folder/'results.json').read_bytes());records=[]
        for row in report['cases']:
            sub=folder/row['strategy'];nav=pd.read_parquet(sub/'equity.parquet');fills=pd.read_parquet(sub/'fills.parquet')
            assert len(nav)==181 and nav.date.is_unique and nav.date.is_monotonic_increasing
            broker=PaperBroker(sub/'paper.sqlite3');rec=broker.reconcile('isolated');assert rec.matched
            assert len(broker.fills('isolated'))==len(fills)==row['fills']
            independent=period_statistics(nav.equity,10_000_000)
            for key in ['net_return','sharpe','maximum_drawdown']:
                if row[key] is None:assert independent[key] is None
                else:assert abs(row[key]-independent[key])<1e-10
            turnover=float(fills.notional.sum()/2/nav.equity.mean()*252/181) if len(fills) else 0.
            records.append({**row,'window_start':'20260105','window_end':'20260930','initial_cash':10_000_000,'slippage_bps':bps,'execution_version':'uniform_prev_close_shares_paperbroker_v1','complete':True,'annualized_one_way_turnover':turnover})
            audits.append({'bps':bps,'strategy':row['strategy'],'reconciled':True,'fills':len(fills),'cash_difference':rec.cash_difference})
        front=pareto_front(records);fronts[str(bps)]=front;metrics[bps]={r['strategy']:r for r in front}
        text += [f'## 单边{bps}bps','', '| 策略 | 区间收益 | 年化收益 | 波动 | 夏普 | 最大回撤 | 年化单边换手倍数 | 帕累托 |','|---|---:|---:|---:|---:|---:|---:|---|']
        for r in front:
            sharp=f"{r['sharpe']:.3f}" if r['sharpe'] is not None else '不可定义'
            text.append(f"| {NAMES[r['strategy']]} | {r['net_return']:.2%} | {r['annualized_return']:.2%} | {r['annualized_volatility']:.2%} | {sharp} | {r['maximum_drawdown']:.2%} | {r['annualized_one_way_turnover']:.2f} | {r['pareto_status']} |")
        text.append('')
    text += ['## 相比20日旧价量对照的增量（10bps）','', '| 新方法 | 收益差（百分点） | 夏普差 | 回撤降低（百分点） | 两档成本都严格支配该对照 |','|---|---:|---:|---:|---|']
    base=metrics[10]['r_old_ridge20']
    robust=[]
    for key,r in metrics[10].items():
        if key in ['ridge_reference','ep_buffer','r_old_ridge20']:continue
        dominates=all(key in metrics[b]['r_old_ridge20']['dominated_by'] for b in [10,20])
        if dominates:robust.append(key)
        sharp=r['sharpe']-base['sharpe'] if r['sharpe'] is not None and base['sharpe'] is not None else np.nan
        text.append(f"| {NAMES[key]} | {(r['net_return']-base['net_return'])*100:+.2f} | {sharp:+.3f} | {(base['maximum_drawdown']-r['maximum_drawdown'])*100:+.2f} | {'是' if dominates else '否'} |")
    text += ['', '## 季度与风格（10bps）','', '| 策略 | Q1 | Q2 | Q3 | 创业板＋科创板平均NAV权重 | 最大行业平均NAV权重 |','|---|---:|---:|---:|---:|---|']
    for key,r in metrics[10].items():
        sectors=sorted(r['industry_mean_weights'].items(),key=lambda p:-p[1]);top=sectors[0] if sectors else ('无',0.)
        text.append('| '+NAMES[key]+' | '+' | '.join(f"{r['quarters'][q]['net_return']:.2%}" for q in ['2026Q1','2026Q2','2026Q3'])+f" | {r['growth_board_mean_weight']:.2%} | {top[0]} {top[1]:.2%} |")
    text += ['', '## 解释及剩余工作','',
             '- 帕累托使用净年化收益↑、夏普↑、最大回撤↓、换手↓。非支配不是全面优于旧模型；严格支配是本样本点估计，不代表统计显著。相对20日旧价量对照的支配也不能替代相对因子基线/V2的比较。',
             '- 这不是完整QMJ或盈利超预期模型。信号仍有历史供应商版本、候选池延续和未来标签缺失选择偏差风险；财务可见性校验不等于原始版本完整PIT认证。',
             '- 同一2026区间已经看过，不是全新盲测。真实新增前瞻必须在未来积累。',
             '- 执行模型未包含盘口排队和动态冲击，成本压力不是实盘成交保证；行业报告使用当前证券元数据，只作描述，不进入信号。信号计算用生效行业区间。',
             '- 仍需：因子基线/V2同口径回放、Qlib结构化风险组合、同样本风格和费用归因、分期/块bootstrap支配稳定性、参数邻域审查、候选冻结及独立模拟验收。不得提前称R4/R5/R6全完成。']
    report=ROOT/'docs/research/ASHARE_PARETO_PILOT_RESULTS_20261111.md';report.write_text('\n'.join(text)+'\n',encoding='utf-8')
    evidence={'status':'PILOT_COMPLETE_FULL_R4_R5_R6_PENDING','fits_audited':18,'account_audits':audits,'fronts_by_cost':fronts,'strictly_dominates_20day_old_ridge_both_costs':robust,'production_modified':False}
    (OUT/'pilot-summary.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print(report)
    for k,r in metrics[10].items():print(k,round(r['net_return']*100,2),r['sharpe'],round(r['maximum_drawdown']*100,2),r['pareto_status'])
    print('BOTH_COSTS_DOMINATE_20DAY_CONTROL',robust)


if __name__=='__main__':main()
