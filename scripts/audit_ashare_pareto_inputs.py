"""R1: monthly point-in-time coverage; no downloads or production mutations."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from moneymore.data.store import ParquetStore
from moneymore.data.fundamental_research import research_financial_frame
from moneymore.research.baseline_v2_financials import common_period_features,required_period
from moneymore.research.baseline_v2_industry import industry_as_of
from moneymore.research.pareto_rebuild import pareto_front

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'state/ashare-pareto-rebuild-20261111'


def load_basic(store, dates=None):
    parts=[store.read('daily_basic'),store.read('baseline_v2_daily_basic')]
    for path in (ROOT/'state/baseline-v2-validation/daily-inputs').glob('*.parquet'):
        if dates is not None and path.stem not in dates:continue
        marker=path.with_suffix('.json')
        if not marker.exists():continue
        proof=json.loads(marker.read_bytes())
        if hashlib.sha256(path.read_bytes()).hexdigest()!=proof['sha256']:raise ValueError('HISTORICAL_BASIC_HASH_MISMATCH:'+path.stem)
        parts.append(pd.read_parquet(path))
    parts += [pd.read_parquet(p) for p in (ROOT/'state/four-strategy-2026-jan-sep-v1/official').glob('daily_basic-*.parquet') if dates is None or p.stem.split('-')[-1] in dates]
    basic=pd.concat(parts,ignore_index=True).drop_duplicates(['trade_date','ts_code'],keep='last')
    basic.trade_date=basic.trade_date.astype(str)
    if 'source_trade_date' in basic:basic=basic.loc[basic.source_trade_date.isna()|basic.source_trade_date.astype(str).eq(basic.trade_date)]
    return basic if dates is None else basic.loc[basic.trade_date.isin(dates)]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    store=ParquetStore(ROOT/'data');source=ROOT/'state/mature-multifactor/p1-native-20261103-a'
    membership=pd.read_parquet(source/'membership.parquet');membership.datetime=pd.to_datetime(membership.datetime)
    dates=pd.DatetimeIndex(sorted(membership.datetime.unique()))
    dates=dates[dates>=pd.Timestamp('2024-01-01')]
    sample=pd.Series(dates,index=dates).groupby(dates.to_period('M')).max().tolist()
    bars=pd.read_parquet(source/'bars-input.parquet');bars.trade_date=bars.trade_date.astype(str)
    financials=research_financial_frame(store)
    full=store.read('fina_indicator_research_v2')
    industries=store.read('baseline_v2_industry_membership')
    basic=load_basic(store,{d.strftime('%Y%m%d') for d in sample})
    prices=bars.assign(adjusted=bars.close*bars.adj_factor).pivot(index='trade_date',columns='ts_code',values='adjusted').sort_index()
    counts=prices.notna().rolling(252,min_periods=1).sum()
    columns=['date','members','daily_close','valuation_ep','valuation_bp','size','roe_common_period','sales_growth_common_period','industry_effective','price_252_complete','grossprofit_margin','roa','debt_to_assets']
    rows=[]
    for date in sample:
        day=date.strftime('%Y%m%d');members=sorted(membership.loc[membership.datetime.eq(date),'instrument'])
        b=basic.loc[basic.trade_date.eq(day)].set_index('ts_code').reindex(members)
        fin=common_period_features(financials,day).set_index('ts_code').reindex(members)
        ind=industry_as_of(industries,day).reindex(members)
        available=pd.to_datetime(full.ann_date,format='%Y%m%d',errors='coerce')+pd.Timedelta(days=1)
        raw=full.loc[(available<=date)&full.end_date.astype(str).eq(required_period(day))].sort_values('ann_date').drop_duplicates('ts_code',keep='last').set_index('ts_code').reindex(members)
        record=dict(date=day,members=len(members),daily_close=int(prices.loc[day].reindex(members).notna().sum()),valuation_ep=int((b.pe_ttm.notna()&b.pe_ttm.ne(0)).sum()),valuation_bp=int((b.pb.notna()&b.pb.gt(0)).sum()),size=int(b.total_mv.gt(0).sum()),roe_common_period=int(fin.roe.notna().sum()),sales_growth_common_period=int(fin.q_sales_yoy.notna().sum()),industry_effective=int(ind.l1_code.notna().sum()),price_252_complete=int(counts.loc[day].reindex(members).eq(252).sum()))
        for key in ['grossprofit_margin','roa','debt_to_assets']:record[key]=int(raw[key].notna().sum()) if key in raw else 0
        rows.append(record)
    coverage=pd.DataFrame(rows)[columns];coverage.to_parquet(OUT/'input-coverage.parquet',index=False)
    artifacts=[source/'membership.parquet',source/'bars-input.parquet',store.curated/'fina_indicator_research_v2.parquet',store.curated/'baseline_v2_industry_membership.parquet']
    inventory={'status':'COVERAGE_AUDIT_COMPLETED_NOT_FULL_PIT_CERTIFICATION','months':len(coverage),'coverage':rows,'limitations':['membership snapshot carry','vendor historical financial revisions not original-vintage certified','historical core quality only ROE/revenue; optional quality fields sparse','price coverage checks do not certify suspension causes'],'hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}}
    (OUT/'input-audit.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf-8')
    report=['# R1 数据可用性核对','', '月末按当日历史池、公告日后可见的共同财报期、生效行业区间核对；不是今天的全库非空率。没有下载财报原文，没有改生产数据。','', '| 日期 | 池 | EP | BP | ROE | 单季营收同比 | 行业 | 完整252日价格 | 毛利率 | ROA | 杠杆 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:report.append('| '+' | '.join(str(r[k]) for k in ['date','members','valuation_ep','valuation_bp','roe_common_period','sales_growth_common_period','industry_effective','price_252_complete','grossprofit_margin','roa','debt_to_assets'])+' |')
    report += ['', '## 决策','', '- 先执行EP/BP＋共同期ROE质量代理、ROE/营收变化代理、中期动量；毛利/资产、ROA、杠杆和现金流质量不能声称已完整实现。','- 不向前填未来财报或行业；缺失保留标记，少数缺失不改变其他股票的公式权重。','- 严格252日全覆盖只是一项诊断；后续区分正式停牌和漏数，不能把缺失当零收益。','- 数据版本与候选池历史局限继续披露；研究可推进，不能冒充全量原始版本认证。']
    (ROOT/'docs/research/ASHARE_PARETO_INPUT_AUDIT_20261111.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    # Existing completed runs validate the evaluator, not a new model result.
    fronts={}
    for bps in [10,20]:
        folder=ROOT/f'state/ashare-paper-routes-20261111/controlled-drop5-v2/bps{bps}'
        data=json.loads((folder/'results.json').read_bytes());records=[]
        for row in data['cases']:
            nav=pd.read_parquet(folder/row['strategy']/'equity.parquet');fills=pd.read_parquet(folder/row['strategy']/'fills.parquet')
            records.append({**row,'window_start':str(nav.date.iloc[0]),'window_end':str(nav.date.iloc[-1]),'initial_cash':10_000_000,'slippage_bps':bps,'execution_version':'paper_routes_controlled_drop5_v2','complete':len(nav)==181,'annualized_one_way_turnover':float(fills.notional.sum()/2/nav.equity.mean()*252/len(nav))})
        fronts[str(bps)]=pareto_front(records)
    (OUT/'prior-experiments-pareto.json').write_text(json.dumps({'scope':'EVALUATOR_ACCEPTANCE_ONLY_NOT_NEW_MODEL','by_cost':fronts},ensure_ascii=False,indent=2),encoding='utf-8')
    print(coverage.tail(5).to_string(index=False))
    print({cost:{r['strategy']:r['pareto_status'] for r in rows} for cost,rows in fronts.items()})


if __name__=='__main__':main()
