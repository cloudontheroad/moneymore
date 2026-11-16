"""Summarize completed repair evidence, never imply unfinished Qlib/release work."""
import json
import pandas as pd
from audit_ashare_pareto_inputs import ROOT, OUT


def main():
    repair=json.loads((OUT/'label-repair-v2/results.json').read_bytes())
    old=pd.read_parquet(OUT/'signals/panel.parquet')
    new=pd.read_parquet(OUT/'label-repair-v2/panel.parquet')
    features=[c for c in old if not c.startswith(('label_','residual_label_'))]
    pd.testing.assert_frame_equal(old[features],new[features])
    text=['# R4 标签补充与统一回放续办结果','',
          '本报告只总结已有完整文件；不代替Qlib风险组合、发布校验或前瞻收益。原版面板和回放保留，未改生产账户。','',
          '## 端点修复','',
          '| 信号年 | 标签期限 | 端点已知样本 | 修复前缺失 | 新补回 | 仍缺失 |','|---|---:|---:|---:|---:|---:|']
    for r in repair['counts']:
        text.append(f"| {r['year']} | {r['horizon']} | {r['mature_endpoint_rows']} | {r['missing_before']} | {r['repaired']} | {r['missing_after']} |")
    text+=['',f"价格来源重叠核验拒绝 {repair['rejected_source_symbol_pairs']} 个来源/股票组合；不混接不一致价格或复权因子。没有用零收益、后向未来填充处理仍缺失样本。特征列逐值一致，重算残差标签是必要的：截面回归的有效样本变化也会影响原本不缺标签的股票。",'',
           '表中包括截至源行情末日能找到结束日的标签，不等同于截至9月30日可用于绩效诊断的成熟样本。训练仍严格要求label_end早于refit_date。','',
           '## 重拟合','']
    repaired_ends=[]
    for horizon in (5,20,60):
        repaired_mask=old[f'label_{horizon}'].isna()&new[f'label_{horizon}'].notna()
        repaired_ends.extend(new.loc[repaired_mask,f'label_end_{horizon}'].dropna().tolist())
    if repaired_ends and min(repaired_ends)>='20260701':
        text.append(f'- 补回标签最早结束日为{min(repaired_ends)}，晚于本次最后重拟合日2026-07-01；因此不能把这些未来成熟收益喂入前三季度模型。修复改善的是标签覆盖与后续训练输入，并不应改变这次前三季度预测/成交。')
    for directory,expected in [('validation2025',24),('walkforward-repaired',18)]:
        path=OUT/directory/'results.json'
        if not path.exists():text.append(f'- {directory}：尚未完成。');continue
        fitted=json.loads(path.read_bytes());assert len(fitted['fits'])==expected
        assert all(f['last_label_end']<f['refit_date'] for f in fitted['fits'])
        text.append(f'- {directory}：{expected}次拟合完成，所有成熟日检查通过。')
    text+=['','## 修复后与原回放对照','', '| 成本 | 方法 | 原收益 | 修复后收益 | 修复后夏普 | 修复后回撤 |','|---|---|---:|---:|---:|---:|']
    for bps in (10,20):
        path=OUT/f'repaired-replays/bps{bps}/results.json'
        if not path.exists():text.append(f'\n{bps}bps修复后回放尚未完成。\n');continue
        before={r['strategy']:r for r in json.loads((OUT/f'pilot-replays/bps{bps}/results.json').read_bytes())['cases']}
        for r in json.loads(path.read_bytes())['cases']:
            text.append(f"| {bps} | {r['strategy']} | {before[r['strategy']]['net_return']:.2%} | {r['net_return']:.2%} | {r['sharpe']:.3f} | {r['maximum_drawdown']:.2%} |")
    text+=['','## 原基线与V2统一回放','', '| 成本 | 方法 | 收益 | 夏普 | 回撤 | 成交笔数 |','|---|---|---:|---:|---:|---:|']
    for bps in (10,20):
        path=OUT/f'baseline-replays/bps{bps}/results.json'
        if not path.exists():text.append(f'\n{bps}bps基线回放尚未完成。\n');continue
        for r in json.loads(path.read_bytes())['cases']:
            text.append(f"| {bps} | {r['strategy']} | {r['net_return']:.2%} | {r['sharpe']:.3f} | {r['maximum_drawdown']:.2%} | {r['fills']} |")
    text+=['','## 剩余边界','',
           '- 本文件仅覆盖标签与回放修复。Qlib风险组合、分期归因、参数邻域及发布判定见ASHARE_PARETO_FINAL_RESULTS_20261111.md，不用本文件推断整体阶段状态。',
           '- 此前看过2026表现，所有2026九个月指标均为回顾研究；2025滚动验证选择也不等于未来收益保证。']
    path=ROOT/'docs/research/ASHARE_PARETO_REPAIR_RESULTS_20261111.md'
    path.write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(path,flush=True)


if __name__=='__main__':main()
