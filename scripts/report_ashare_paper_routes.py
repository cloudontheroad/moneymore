"""Independent ledger checks and a generated report for the predeclared experiments."""
from pathlib import Path
import hashlib
import json
import sqlite3
import numpy as np
import pandas as pd
from moneymore.execution.paper import PaperBroker
from moneymore.research.period_statistics import period_statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'state/ashare-paper-routes-20261111/controlled-drop5-v2'
NAMES={'ridge_reference':'价量Ridge对照','ep':'① EP价值信号','ridge_buffer':'② 原信号＋低换手','ep_buffer':'价值＋低换手','ridge_vol':'③ 原信号＋波动仓位','ep_buffer_vol':'价值＋低换手＋波动仓位','ridge_static':'事后等平均目标固定仓位'}


def main():
    model_root=ROOT/'state/four-strategy-2026-jan-sep-v1/model'
    model=json.loads((model_root/'results.json').read_bytes())
    assert model['label_available_by']=='20251231' and not model['latest_2026_model_reused']
    assert hashlib.sha256((model_root/'predictions.parquet').read_bytes()).hexdigest()==model['artifact_hashes']['predictions.parquet']
    reports={bps:json.loads((OUT/f'bps{bps}/results.json').read_bytes()) for bps in (10,20)}
    lines=['# A股三条论文机制实验结果（2026-11-11）','',
           '已完成14组×181个交易日。2026-01-05至09-30，初始1000万元，无风险收益为0，现金不计息。所有指标扣佣金、税费与滑点。',
           '', '这是预先固定规则的机制消融，不是论文完整复现，也不是已通过实盘验收的新策略。对照是截至2025年底训练的原生Alpha158/Ridge信号＋Top50/drop5，不是页面“因子基线”。所有组使用本次统一的前收盘股数规划和次开盘现金校验；未复用旧回测收益直接相减。', '']
    audit=[]
    for bps,report in reports.items():
        assert not report['smoke'] and report['all_daily_reconciliations_pass']
        lines += [f'## 单边{bps}bps','', '| 方法 | 区间收益 | 年化收益 | 年化波动 | 夏普 | 最大回撤 | 平均股票仓位 | 成交数 | 税费＋滑点（万元） |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for row in report['cases']:
            folder=OUT/f'bps{bps}'/row['strategy']
            equity=pd.read_parquet(folder/'equity.parquet');fills=pd.read_parquet(folder/'fills.parquet')
            broker=PaperBroker(folder/'paper.sqlite3')
            ledger=broker.fills('isolated')
            assert len(ledger)==len(fills)==row['fills']
            assert len(equity)==181 and equity.date.is_unique and equity.date.is_monotonic_increasing
            assert abs(sum(float(r['fee']) for r in ledger)-float(fills.fee.sum()))<.01
            stats=period_statistics(equity.equity,10_000_000)
            for key in ['net_return','annualized_return','annualized_volatility','sharpe','maximum_drawdown']:assert abs(stats[key]-row[key])<1e-10
            rec=broker.reconcile('isolated');assert rec.matched
            audit.append({'bps':bps,'strategy':row['strategy'],'fills':len(ledger),'cash_difference':rec.cash_difference,'quantity_difference':rec.quantity_difference})
            lines.append(f"| {NAMES[row['strategy']]} | {row['net_return']:.2%} | {row['annualized_return']:.2%} | {row['annualized_volatility']:.2%} | {row['sharpe']:.3f} | {row['maximum_drawdown']:.2%} | {row['mean_exposure']:.2%} | {row['fills']} | {(row['fees']+row['slippage'])/10000:.2f} |")
        lines.append('')
    cases={r['strategy']:r for r in reports[10]['cases']};base=cases['ridge_reference']
    lines += ['## 本轮结论','',
              '1. 单独EP价值替换主要降低风险暴露，并非强预测能力证明；价值＋低换手为本轮较好的组合，但仅约1.33%九个月收益、0.198夏普，不足以称成熟可发布模型。',
              '2. 原Ridge加缓冲基本无效：1862笔仅降到1860笔。排名变动多数已穿过持有缓冲区，固定缓冲无法修复信号稳定性。',
              '3. EP加缓冲由228笔降为60笔，收益增加约1.03个百分点，但最大回撤略增约0.61个百分点；不是所有指标都改善。',
              '4. 本版波动仓位不采用：与约66%实际平均仓位的固定仓位对照相比，收益少约6.64个百分点，回撤仅少约0.65个百分点。三条叠加也比价值＋低换手少约2.24个百分点。',
              '5. EP＋低换手平均银行权重62.35%，创业板＋科创板仅0.21%，明显未解决风格均衡；Q2亏9.82%、Q3赚12.52%，表现依赖市场阶段。禁止用其替换原生产策略或宣称解决了科技/银行集中问题。',
              '6. 两档成本的14组均完成，14项相关测试通过，所有日终对账及最终独立成交/费用/净值复核通过。代码和论文机制可复用，但没有由此证明新的强Alpha。','']
    lines += ['## 10bps下的改变量','', '| 方法对照 | 区间收益差（百分点） | 夏普差 | 最大回撤降低（百分点） |','|---|---:|---:|---:|']
    pairs=[(k,'ridge_reference') for k in ['ep','ridge_buffer','ridge_vol','ep_buffer','ep_buffer_vol']]+[('ep_buffer','ep'),('ridge_vol','ridge_static'),('ep_buffer_vol','ep_buffer')]
    for a,b in pairs:
        x,y=cases[a],cases[b]
        lines.append(f"| {NAMES[a]} − {NAMES[b]} | {(x['net_return']-y['net_return'])*100:+.2f} | {x['sharpe']-y['sharpe']:+.3f} | {(y['maximum_drawdown']-x['maximum_drawdown'])*100:+.2f} |")
    lines += ['', '## 季度区间收益（10bps）','', '| 方法 | Q1 | Q2 | Q3 |','|---|---:|---:|---:|']
    for row in cases.values():
        lines.append('| '+NAMES[row['strategy']]+' | '+' | '.join(f"{row['quarters'][q]['net_return']:.2%}" for q in ['2026Q1','2026Q2','2026Q3'])+' |')
    lines += ['', '## 风格与数据覆盖','', '| 方法 | 创业板＋科创板平均NAV权重 | 最大三个行业平均NAV权重 |','|---|---:|---|']
    for row in cases.values():
        sectors=sorted(row['industry_mean_weights'].items(),key=lambda x:-x[1])[:3]
        lines.append(f"| {NAMES[row['strategy']]} | {row['growth_board_mean_weight']:.2%} | "+'、'.join(f'{k} {v:.2%}' for k,v in sectors)+' |')
    coverage=pd.DataFrame(json.loads((OUT/'bps10/coverage.json').read_bytes()))
    lines += ['', f"EP每天有效分数 {coverage.ep_scored.min()}—{coverage.ep_scored.max()}；Ridge {coverage.ridge_scored.min()}—{coverage.ridge_scored.max()}。缺失不填零，持有且仍在池内的缺分证券冻结而非误认成退出。",'',
              '## 不确定性：配对10日块Bootstrap','',
              '固定随机种子，2000次循环块重采样，给出相对对照的“日均净收益差×252”区间（非复合年化收益）。保留同期相关性，但不修复反复研究同一测试期或历史数据版本偏差；不作为策略自动晋级门禁。','',
              '| 方法 | 年化均值差估计 | 95%经验区间 |','|---|---:|---|']
    def ret(kind):
        nav=pd.read_parquet(OUT/'bps10'/kind/'equity.parquet').equity.to_numpy()
        return nav/np.r_[10_000_000,nav[:-1]]-1
    reference=ret('ridge_reference');rng=np.random.default_rng(20261111);n=len(reference)
    index=np.concatenate([(rng.integers(n,size=(2000,1))+np.arange(10))%n for _ in range((n+9)//10)],axis=1)[:,:n]
    uncertainty={}
    for key in ['ep','ridge_buffer','ridge_vol','ep_buffer','ep_buffer_vol']:
        diff=ret(key)-reference;sample=diff[index].mean(axis=1)*252;lo,hi=np.quantile(sample,[.025,.975])
        uncertainty[key]={'annualized_mean_difference':float(diff.mean()*252),'interval95':[float(lo),float(hi)]}
        lines.append(f'| {NAMES[key]} | {diff.mean()*252:+.2%} | [{lo:+.2%}, {hi:+.2%}] |')
    lines += ['', '## 解释边界','',
              '- EP路线只是单因子替换，不能等同于已经完成价值、质量、成长的成熟多因子模型；银行/周期暴露变化不是纯选股Alpha。行业来自当前证券元数据，仅作描述，不进信号。',
              '- 低换手改变持仓路径，收益差不能全算节省成本；即使成本下降，保留弱势股票也可能亏得更多。',
              '- 波动仓位是上一完整月市场代理方差的倒数，标度在2025年底冻结；没有用测试收益挑参数。固定仓位组匹配的是事后平均目标，实际平均仓位会因跳空/价格漂移/未成交而略有偏差，仅作归因诊断。',
              '- 开盘价加固定滑点只是成交模型，未建模盘口排队/市场冲击/停牌外的瞬时流动性，不能视为真实成交保证。整手按100股保守处理，科创板初买不少于200股。',
              '- 现有历史池包含快照延续，历史财务估值字段未经全量原始版本认证；此期间此前被研究过，不是全新盲测。这些限制不会因14个测试和账本对账通过而消失。',
              '- 原生产策略、真实委托、生产净值均未修改。', '',
              '## 研究来源','',
              '- [Size and Value in China](https://www.nber.org/papers/w24458)：A股EP和小市值壳价值处理。',
              '- [A Taxonomy of Anomalies and Their Trading Costs](https://www.nber.org/papers/w20721)：买入/持有门槛分离。',
              '- [Volatility-Managed Portfolios](https://www.nber.org/papers/w22208)：历史方差倒数管理风险。']
    target=ROOT/'docs/research/ASHARE_PAPER_ROUTES_RESULTS_20261111.md'
    target.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    paths=[ROOT/'scripts/research_ashare_paper_routes.py',ROOT/'src/moneymore/research/paper_routes.py',ROOT/'state/four-strategy-2026-jan-sep-v1/model/predictions.parquet',ROOT/'state/mature-multifactor/p1-native-20261103-a/membership.parquet']
    evidence={'status':'PASS','ledger_checks':audit,'bootstrap':uncertainty,'model_cutoff':model['label_available_by'],'prediction_hash_matches_frozen_model':True,'hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (OUT/'audit.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print(target)
    print(json.dumps({k:{f:r[f] for f in ['net_return','sharpe','maximum_drawdown','mean_exposure','fills','fees','slippage']} for k,r in cases.items()},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
