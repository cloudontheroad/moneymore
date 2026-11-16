"""Final independent audits and return/style attribution, isolated files only."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from moneymore.research.period_statistics import period_statistics
from moneymore.research.ashare_mainline import sha256
from report_ashare_mainline_replay import audit_ledger
from report_ashare_rank_preserving import assert_exact_replay_equal,daily_returns
from report_ashare_n4 import attribution
from diagnose_ashare_rank_turnover import attach_industry
from prepare_four_strategy_2026 import save

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'state/ashare-simple-rules-20261113'
N=ROOT/'state/ashare-mainline-20261112'
NAMES={'doubleensemble_full--rank_cap':'原候选 rank_cap','raw_reversal--rank_cap':'原始5日反转','adjusted_reversal--rank_cap':'行业调整＋公告过滤反转','core_timing--rank_cap':'原候选＋最多3日执行延期','conservative_dividend--rules':'低波红利动量（适配版）','fcf_rules--rules':'自由现金流（CNI适配版）'}


def main(partial=False):
    spec=json.loads((ROOT/'configs/ashare_simple_rules_v1.json').read_bytes())
    panel=pd.read_parquet(N/'panel-v2/panel.parquet',columns=['date','symbol','industry'])
    rows=[];check={};equities={};details={};output=BASE/('partial-report-v1' if partial else 'final-v1')
    output.mkdir(parents=True,exist_ok=False)
    for bps in [10,20]:
        groups=['nonfcf'] if partial else ['nonfcf','fcf']
        for group in groups:
            run=BASE/f'{group}-cost{bps}-v1'
            done=json.loads((run/'completed.json').read_bytes())
            for p,h in done['source_hashes'].items():
                frozen=run/'sources'/Path(p).relative_to(ROOT)
                if sha256(frozen if frozen.exists() else Path(p))!=h:raise ValueError('SOURCE_CHANGED:'+p)
            folder=run/f'bps{bps}'
            result=json.loads((folder/'results.json').read_bytes())
            allhold=pd.read_parquet(folder/'holdings.parquet')
            if result['smoke'] or not result['all_daily_reconciliations_pass']:raise ValueError('REPLAY_NOT_FULL')
            for engine in result['cases']:
                case=engine['strategy'];e=pd.read_parquet(folder/case/'equity.parquet');f=pd.read_parquet(folder/case/'fills.parquet')
                if len(e)!=181 or e.date.duplicated().any() or not e.date.is_monotonic_increasing:raise ValueError('CALENDAR_INCOMPLETE')
                ledger=audit_ledger(folder/case/'paper.sqlite3',f)
                stats=period_statistics(e.equity,10_000_000)
                for name in ['net_return','sharpe','annualized_volatility','maximum_drawdown']:
                    if not np.isclose(stats[name],engine[name],atol=1e-10):raise ValueError('METRIC_MISMATCH')
                h=allhold.loc[allhold.kind.eq(case)]
                style=attach_industry(h,panel)
                sectors=style.groupby(['date','industry']).weight.sum()
                decisions=[json.loads(p.read_bytes()) for p in sorted((run/'decisions'/case).glob('*.json'))]
                if len(decisions)!=181:raise ValueError('DECISION_INCOMPLETE')
                for d in decisions:
                    if d['signal_date']>=d['execution_date']:raise ValueError('FUTURE_SIGNAL')
                    if d.get('next_open_used_for_selection',False):raise ValueError('OPEN_SELECTION')
                    if d.get('target_weights') and (sum(d['target_weights'].values())>.995+1e-9 or max(d['target_weights'].values())>.05+1e-9):raise ValueError('WEIGHT_CONSTRAINT')
                    if d.get('age',0)>=3 and d.get('delayed_symbols'):raise ValueError('DELAY_DEADLINE')
                row={'case':case,'name':NAMES[case],'bps':bps,**stats,'fills':len(f),'fees':float(f.fee.sum()),'slippage':float(f.slippage.sum()),'total_cost':float(f.fee.sum()+f.slippage.sum()),'mean_exposure':float(e.exposure.mean()),'min_exposure':float(e.exposure.min()),'mean_names':float(h.groupby('date').size().reindex(e.date,fill_value=0).mean()),'max_single_weight_after_market_moves':float(h.weight.max()),'growth_board_mean_weight':float(h.loc[h.symbol.str.startswith(('30','68'))].weight.sum()/181),'mean_largest_industry':float(sectors.groupby('date').max().mean()),'industry_mean_weights':(style.groupby('industry').weight.sum()/181).to_dict(),'quarters':engine['quarters'],'ledger':ledger,'decision_states':dict(Counter(d['status'] for d in decisions)),'timing_overrides':dict(Counter(d['override'] for d in decisions if d.get('override'))),'timing_delayed_symbol_sessions':sum(len(d.get('delayed_symbols',[])) for d in decisions),'rule_coverage':{d['signal_date']:d['coverage'] for d in decisions if d.get('coverage')}}
                row['attribution']=attribution(folder,case,e,f,h,panel)
                rows.append(row);equities[(bps,case)]=e
                details[f'{bps}:{case}']={'equity_sha256':sha256(folder/case/'equity.parquet'),'fills_sha256':sha256(folder/case/'fills.parquet'),'decisions_sha256':{p.name:sha256(p) for p in (run/'decisions'/case).glob('*.json')}}
                reference=None
                if case.startswith('doubleensemble_full'):reference=N/f'n3r-rank-v2/cost{bps}/bps{bps}'/case
                if case.startswith('raw_reversal'):reference=N/f'n4-matrix-v2/style-controls-v2-cost{bps}/bps{bps}/factor_reversal--rank_cap'
                if reference:
                    for name in ['equity.parquet','fills.parquet']:
                        assert_exact_replay_equal(pd.read_parquet(folder/case/name),pd.read_parquet(reference/name),f'{bps}:{case}:{name}')
                        check[f'{bps}:{case}:{name}']={'exact_equal':True,'binary_equal':sha256(folder/case/name)==sha256(reference/name)}
    correlations={}
    for bps in [10,20]:
        core=next(r for r in rows if r['bps']==bps and r['case'].startswith('doubleensemble_full'))
        for row in [r for r in rows if r['bps']==bps]:
            row['net_return_delta_core_pp']=(row['net_return']-core['net_return'])*100
            row['sharpe_delta_core']=row['sharpe']-core['sharpe']
            row['drawdown_delta_core_pp']=(row['maximum_drawdown']-core['maximum_drawdown'])*100
        returns=pd.DataFrame({c:daily_returns(e,10_000_000) for (b,c),e in equities.items() if b==bps})
        correlations[str(bps)]=returns.corr().to_dict()
    expected=(5 if partial else 6)*2
    if len(rows)!=expected:raise ValueError('MATRIX_INCOMPLETE')
    data_audit=None if partial else json.loads((BASE/'data-audit-v1.json').read_bytes())
    if data_audit:
        for item in data_audit['files']:
            if sha256(BASE/'data-v1'/item['file'])!=item['sha256']:raise ValueError('AUDITED_DATA_CHANGED')
    save(output/'results.json',{'status':'PARTIAL_NONFCF_COMPLETE' if partial else 'COMPLETE','cases':rows,'exact_controls':check,'correlations':correlations,'evidence':details,'data_audit_sha256':None if partial else sha256(BASE/'data-audit-v1.json'),'limitations':spec['limitations'],'production_modified':False})
    lines=['# 成熟简单规则回测结果','',f'状态：{"前两路线完成，现金流仍待补数" if partial else "三路线全部完成"}。2026-01-05—2026-09-30，181交易日，1000万元；实际A股撮合、佣金/印花税及单边滑点。夏普无风险利率0、252日年化。','',
           '| 策略 | 滑点bps | 九个月净收益 | 年化波动 | 夏普 | 最大回撤 | 成交数 | 成本万元 | 双创平均仓位 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in sorted(rows,key=lambda x:(x['bps'],spec['cases'].index(x['case']))):
        lines.append(f"| {r['name']} | {r['bps']} | {r['net_return']:.2%} | {r['annualized_volatility']:.2%} | {r['sharpe']:.3f} | {r['maximum_drawdown']:.2%} | {r['fills']} | {r['total_cost']/10000:.2f} | {r['growth_board_mean_weight']:.2%} |")
    lines+=['','全部账户逐日对账；独立从成交、公司行为账重建最终现金/股份；core与原反转两档控制净值及成交逐笔复现。','',
            '## 适配和限制','',
            '- 低波组用股息率代替净派息率，没有完整回购/增发现金净额，不能称Robeco原版复现。',
            '- 自由现金流用真实Tushare企业FCFF、净债务、季度ROE及现金流/利润，采用既有Top1000、申万行业和5%单股上限；不是国证指数复制，无官方定性审查及历史指数名单。',
            '- 反转是行业调整及财报公告排除，不等同完整新闻调整或FF3残差模型。最长3日延期遇风险保护会被覆盖，不能为等时机无限保留旧计划。',
            '- 历史股票池是已有重建版本；财务按公告日期过滤，但供应商修订记录不保证原始未修订版本。2026窗口已多次观察，结论是回顾性诊断，不是新的样本外证明。',
            '- 新月线与旧日线的月末原价一致，复权因子版本存在差异（剔除单股常数缩放后最大约8bps）。新低波规则按已登记的新官方月线缓存计算，未改旧模型快照；不能宣称历史信息版本完全相同。',
            '- 两条100只慢规则继承既有退池强制退出、T+1与现金保护，未套原core的行业主动约束；5%为调仓目标/买入上限，持有期间价格变化可使权重漂移。',
            '- 尚未替换生产、发布新账户或改历史收益。']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps([{k:r[k] for k in ['case','bps','net_return','sharpe','maximum_drawdown','total_cost']} for r in rows]),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args();main(a.partial)
