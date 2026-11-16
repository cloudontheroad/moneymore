"""Close the finite experiment only when every planned account exists and reconciles."""
import json
from collections import Counter
from audit_ashare_pareto_inputs import ROOT, OUT
import report_ashare_pareto_stability as stability
import attribute_ashare_pareto_returns as attribution
import report_ashare_pareto_repair as repair_report


def main():
    required=[]
    for bps in (10,20):
        required += [OUT/f'baseline-replays/bps{bps}/results.json',OUT/f'repaired-replays/bps{bps}/results.json']
        required += [OUT/f'qlib-replays/interval{i}/full-proof-v4/bps{bps}/results.json' for i in (15,20,25)]
    missing=[str(p) for p in required if not p.exists()]
    if missing:raise ValueError('FINAL_ACCOUNTS_NOT_COMPLETE:'+','.join(missing))
    stability.main(OUT/'repaired-replays');attribution.main();repair_report.main()
    report=json.loads((OUT/'repaired-replays/stability.json').read_bytes())
    maps={bps:{r['strategy']:r for r in report['costs'][str(bps)]['pareto']} for bps in (10,20)}
    selection=json.loads((OUT/'blend-selection/results.json').read_bytes())
    primary='validated_qlib_20'
    primary_positive=all(maps[b][primary]['net_return']>0 and maps[b][primary]['sharpe']>0 for b in (10,20))
    primary_dominates={reference:all(primary in maps[b][reference]['dominated_by'] for b in (10,20)) for reference in ('baseline','v2','r_old_ridge20')}
    diagnostic_dominators=[name for name in maps[10] if name not in ('baseline','v2','ridge_reference','ep_buffer','r_old_ridge20') and all(name in maps[b][reference]['dominated_by'] for b in (10,20) for reference in ('baseline','v2'))]
    # Do not promote a 2026 winner rejected by the declared 2025 selection rule.
    outcome='PRIMARY_REQUIRES_RELEASE_REVIEW' if primary_positive and all(primary_dominates.values()) else 'PRIMARY_NOT_QUALIFIED_NO_RELEASE'
    text=['# A股预测与组合重构：有限研究结项','',
          'R0—R5有限研究与对账完成；R6本轮判定见下文。仅隔离回放，没有真实交易、旧策略替换或历史净值改写。','',
          '统一2026年前九个月181个交易日、每账户1000万元、前收盘计划股数/次日开盘执行、10/20bps、真实费用、T+1、手数、涨跌停、公司行为。共36个正式回放账户（13信号×2成本、2基线×2成本、3风险换仓尺度×2成本），逐日及最终账本对账。','',
          '## 主结论','',f'- 主方案发布判定：`{outcome}`。',
          f'- 主方案两档成本均正收益且正夏普：{primary_positive}；相对各对照四目标严格支配：{primary_dominates}。',
          f'- 事后历史点估计中，两档成本都同时支配原基线与V2的方法：{diagnostic_dominators}。这与2025预先选择是否通过、增量置信区间是否跨零是不同问题；保留这条积极结果，但不偷换成已验证的新主方案。',
          '- 信号选择只读取2025滚动验证：非负残差20日组合胜出；浅树及半数混合Q4平均IC为负。不能事后因2026浅树回报较好就改选。',
          '- 15/25日只是预设邻域，不替换20日主方案，不根据2026表现选最佳换仓日。','',
          '## 统一指标','']
    names={'baseline':'原因子基线','v2':'基线V2','r_old_ridge20':'旧价量20日','r_improvement':'盈利改善代理','r_tree_residual20':'浅树20日（2025规则未选中）','r_nonnegative_residual20':'2025选中的非负信号','validated_qlib_20':'非负信号＋Qlib（主方案）','validated_qlib_15':'Qlib·15日邻域','validated_qlib_25':'Qlib·25日邻域'}
    for bps in (10,20):
        text += [f'### 单边{bps}bps','', '| 方法 | 收益 | 夏普 | 回撤 | 年化单边换手 | 科创＋创业平均权重 |','|---|---:|---:|---:|---:|---:|']
        for name,label in names.items():
            r=maps[bps][name];text.append(f"| {label} | {r['net_return']:.2%} | {r['sharpe']:.3f} | {r['maximum_drawdown']:.2%} | {r['annualized_one_way_turnover']:.2f} | {r['growth_board_mean_weight']:.2%} |")
        text.append('')
    text += ['## 成本与执行','']
    for bps in (10,20):
        r=maps[bps][primary]
        blocks=json.loads((OUT/f'qlib-replays/interval20/full-proof-v4/bps{bps}/validated_qlib/blocks.json').read_bytes())
        text.append(f"- {bps}bps主方案：{r['fills']}笔成交、{r['trading_days']}个成交日，费用{r['fees']:.2f}元、滑点{r['slippage']:.2f}元；阻塞分布{dict(Counter(x['reason'] for x in blocks))}。")
    text += ['', '## 不确定性、归因和限制','',
             '- 全部四目标帕累托、10/20/40日配对区块置信区间、收益相关矩阵：`state/ashare-pareto-rebuild-20261111/repaired-replays/stability.json`。区间跨零只能写未证明，不把点估计支配说成稳定优势。',
             '- 事后市场/价值质量/改善/动量收益回归与费用暴露：`return-attribution/results.json`。这是描述性回归，不是因果归因；截距含择时、现金、费用及个股剩余项，不是纯alpha。等平均仓位市场收益只作代理比较，不冒充可执行反事实。',
             '- 235只证券补取完成，修复3876条标签，20个来源/证券重叠核验不通过仍隔离；所有特征逐值不变、原有效标签一致。没有靠填零、美化缺失或调整历史收益提高结果。',
             '- Qlib规划采用实际风格上限80%的内侧余量，开盘物理裁剪后重验原上限；未取消20%换手约束。v3遗漏停牌证明的研究结果全部排除，仅v4进入前沿。',
             '- Qlib目标函数仍使用原生波动缩放；不能把它宣称为已校准的预期收益/风险/成本最优组合。降低风格暴露不保证提高预测效益。',
             '- 2026区间已被反复研究；历史候选池延续、供应商版本和可选财务字段不足仍存在，不是原始版本完整PIT或盲测证明。未来独立模拟才有新增证据。','',
             '## R6处置','',
             '主方案未达标则本轮不生成新的生产/模拟候选，不替换旧策略。禁止挑2026赢家、放宽成本、取消风险边界或加科技配额凑发布。若主方案达标，本脚本也只输出待验收状态，不自动提交委托。']
    dest=ROOT/'docs/research/ASHARE_PARETO_FINAL_RESULTS_20261111.md';dest.write_text('\n'.join(text)+'\n',encoding='utf-8')
    (OUT/'final-status.json').write_text(json.dumps({'status':outcome,'research_stages_complete':['R0','R1','R2','R3','R4','R5'],'R6':outcome,'accounts':36,'primary_positive_both_costs':primary_positive,'primary_dominates_both_costs':primary_dominates,'retrospective_both_baseline_dominators':diagnostic_dominators,'selection':selection['selected'],'production_modified':False,'candidate_released':False},ensure_ascii=False,indent=2),encoding='utf-8')
    print(dest,flush=True);print(outcome,flush=True)


if __name__=='__main__':main()
