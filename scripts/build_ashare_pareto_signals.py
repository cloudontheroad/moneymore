"""R2: causal family proxies, mature labels and non-tradable signal diagnostics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from moneymore.data.store import ParquetStore
from moneymore.data.fundamental_research import research_financial_frame,financial_features_as_of
from moneymore.research.baseline_v2_financials import common_period_features,required_period
from moneymore.research.baseline_v2_industry import industry_as_of
from moneymore.research.pareto_rebuild import residualize_cross_section
from audit_ashare_pareto_inputs import ROOT,OUT,load_basic

FAMILIES=['value_quality','improvement','momentum']
HORIZONS=[5,20,60]


def ranked(series):
    return series.replace([np.inf,-np.inf],np.nan).rank(pct=True)*2-1


def build():
    output=OUT/'signals';output.mkdir(parents=True,exist_ok=True)
    if (output/'panel.parquet').exists():return pd.read_parquet(output/'panel.parquet')
    source=ROOT/'state/mature-multifactor/p1-native-20261103-a';store=ParquetStore(ROOT/'data')
    member=pd.read_parquet(source/'membership.parquet');member.datetime=pd.to_datetime(member.datetime)
    member_groups={d.strftime('%Y%m%d'):sorted(g.instrument) for d,g in member.groupby('datetime')}
    bars=pd.read_parquet(source/'bars-input.parquet');bars.trade_date=bars.trade_date.astype(str)
    close=bars.assign(p=bars.close*bars.adj_factor).pivot(index='trade_date',columns='ts_code',values='p').sort_index()
    opened=bars.assign(p=bars.open*bars.adj_factor).pivot(index='trade_date',columns='ts_code',values='p').reindex_like(close)
    calendar=list(close.index)
    dates=[d for d in calendar if '20240131'<=d<='20260930' and d in member_groups]
    basic=load_basic(store,set(dates));basic_groups={d:g.set_index('ts_code') for d,g in basic.groupby('trade_date')}
    fin=research_financial_frame(store);industry=store.read('baseline_v2_industry_membership')
    momentum6=close.shift(20)/close.shift(126)-1
    momentum12=close.shift(20)/close.shift(252)-1
    # Endpoint returns remain missing if an endpoint is missing. No zero/forward filling.
    labels={h:close.shift(-h)/opened.shift(-1)-1 for h in HORIZONS}
    pieces=[]
    for i,day in enumerate(dates):
        symbols=member_groups[day];f=pd.DataFrame(index=pd.Index(symbols,name='symbol'))
        b=basic_groups.get(day,pd.DataFrame(columns=['pe_ttm','pb','total_mv'])).reindex(symbols)
        f['industry']=industry_as_of(industry,day).reindex(symbols).l1_code
        f['total_mv']=b.total_mv
        current=common_period_features(fin,day).set_index('ts_code').reindex(symbols)
        previous_period=str(int(required_period(day))-10000)
        previous=financial_features_as_of(fin.loc[fin.end_date.astype(str).eq(previous_period)],day).set_index('ts_code').reindex(symbols)
        f['ep']=1/b.pe_ttm.where(b.pe_ttm.ne(0));f['bp']=1/b.pb.where(b.pb.gt(0))
        f['roe']=current.roe;f['sales_yoy']=current.q_sales_yoy
        f['roe_change']=current.roe-previous.roe;f['sales_change']=current.q_sales_yoy-previous.q_sales_yoy
        f['mom6_1']=momentum6.loc[day].reindex(symbols);f['mom12_1']=momentum12.loc[day].reindex(symbols)
        family_inputs={'value_quality':['ep','bp','roe'],'improvement':['roe_change','sales_yoy','sales_change'],'momentum':['mom6_1','mom12_1']}
        for family,cols in family_inputs.items():
            # Fixed formula across stocks; missing input is not reweighted away.
            score=pd.concat([ranked(f[c]) for c in cols],axis=1).mean(axis=1,skipna=False)
            f[family]=score
            f[family+'_comparable']=ranked(residualize_cross_section(score,f.industry,f.total_mv))
        f['date']=day;f['required_fiscal_period']=required_period(day)
        for key in ['roe_ann_date','q_sales_yoy_ann_date']:
            f[key]=current[key] if key in current else None
            if key in previous:f['previous_'+key]=previous[key]
        offset=calendar.index(day)
        for h in HORIZONS:
            f['label_'+str(h)]=labels[h].loc[day].reindex(symbols)
            f['residual_label_'+str(h)]=residualize_cross_section(f['label_'+str(h)],f.industry,f.total_mv)
            f['label_end_'+str(h)]=calendar[offset+h] if offset+h<len(calendar) else None
        pieces.append(f.reset_index())
        if i%50==0:
            progress={'phase':'R2_PANEL','completed':i+1,'total':len(dates),'date':day}
            (output/'progress.json').write_text(json.dumps(progress),encoding='utf-8');print(progress,flush=True)
    panel=pd.concat(pieces,ignore_index=True);panel.to_parquet(output/'panel.parquet',index=False)
    return panel


def diagnose(panel,output=None,report_path=None):
    output=Path(output) if output else OUT/'signals';output.mkdir(parents=True,exist_ok=True);dates=sorted(panel.date.unique())
    # Fixed five-session grid; calendar blocks are the observational unit, not stocks.
    sample=panel.loc[panel.date.isin(dates[::5])];rows=[]
    for day,f in sample.groupby('date'):
        for family in FAMILIES:
            for variant in ['raw','comparable']:
                score=f[family if variant=='raw' else family+'_comparable']
                for h in HORIZONS:
                    if f['label_end_'+str(h)].isna().any() or str(f['label_end_'+str(h)].iloc[0])>'20260930':continue
                    for target in ['raw','residual']:
                        label=f[('label_' if target=='raw' else 'residual_label_')+str(h)]
                        valid=np.isfinite(score)&np.isfinite(label);s=score.loc[valid];y=label.loc[valid]
                        if len(s)<100 or s.nunique()<10 or y.std()<1e-12:continue
                        pct=s.rank(pct=True)
                        common_end=f.label_end_60.iloc[0]
                        rows.append({'date':day,'family':family,'variant':variant,'horizon':h,'target':target,'assets':len(s),'rank_ic':s.rank().corr(y.rank()),'top_decile_label':float(y.loc[pct.ge(.9)].mean()),'bottom_decile_label':float(y.loc[pct.le(.1)].mean()),'common_mature':bool(pd.notna(common_end) and str(common_end)<='20260930')})
    daily=pd.DataFrame(rows);daily.to_parquet(output/'diagnostic-dates.parquet',index=False)
    summaries=[]
    for common in [False,True]:
        selected=daily.loc[daily.common_mature] if common else daily
        for keys,g in selected.groupby(['family','variant','horizon','target',selected.date.str[:4]]):
            summaries.append(dict(zip(['family','variant','horizon','target','year'],keys))|{'common_mature_only':common,'observations':len(g),'min_assets':int(g.assets.min()),'mean_rank_ic':float(g.rank_ic.mean()),'positive_ic_fraction':float(g.rank_ic.gt(0).mean()),'mean_top_minus_bottom_label':float((g.top_decile_label-g.bottom_decile_label).mean())})
    summary=pd.DataFrame(summaries);summary.to_parquet(output/'summary.parquet',index=False)
    summary.to_json(output/'summary.json',orient='records',force_ascii=False,indent=2)
    covered={c:float(panel[c].notna().mean()) for c in FAMILIES+[f+'_comparable' for f in FAMILIES]}
    result={'status':'R2_DIAGNOSTICS_COMPLETE_NOT_EXECUTABLE_RETURN','rows':len(panel),'dates':len(dates),'coverage':covered,'maturity_end':'20260930','raw_and_residual_targets':True,'signal_date_controls_only':True,'original_vintage_pit_certified':False,'production_modified':False}
    (output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    table=summary.loc[summary.year.eq('2026')&summary.common_mature_only&summary.variant.eq('comparable')]
    report=['# R2 信号与期限诊断','', '三族固定代理、同日行业/市值可比化、每5交易日采样。下表用60日标签也已在2026-09-30前成熟的共同信号日期，避免不同期限悄悄比较不同市场阶段。**IC与分组标签收益不是可成交净收益，更不是新策略夏普。**','', '| 信号族 | 期限 | 目标 | 观察日 | 平均RankIC | IC正值比例 | 最少股票数 |','|---|---:|---|---:|---:|---:|---:|']
    for r in table.to_dict('records'):report.append(f"| {r['family']} | {r['horizon']} | {r['target']} | {r['observations']} | {r['mean_rank_ic']:.4f} | {r['positive_ic_fraction']:.1%} | {r['min_assets']} |")
    report += ['', '## 局限与后续','', '- ROE和营收变化只是质量/改善代理，不是完整QMJ或盈利超预期。','- 正RankIC不保证Top50赚钱，负值也不允许看过结果后直接翻符号。先保留全部结果，再由预设时间滚动训练学习权重。','- 动量用6—1、12—1月端点收益，缺端点不填；历史长度不足保持缺失，停牌和异常价格须在实际回放层继续核验。','- 标签为复权开盘至未来收盘收益，含重叠期限；未扣真实组合交易成本，不按股票行数计算显著性。','- 原始/可比信号、原始/残差标签、2024/2025/2026与全部成熟/共同成熟结果均保存在signals/summary.parquet；不只保留表现好的组。']
    report_path=Path(report_path) if report_path else ROOT/'docs/research/ASHARE_PARETO_SIGNAL_DIAGNOSTICS_20261111.md'
    report_path.write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(result);print(table.to_string(index=False))


if __name__=='__main__':
    with threadpool_limits(limits=2):diagnose(build())
