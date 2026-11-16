"""Pure, explicit adaptations of published rules; no production side effects."""
import numpy as np
import pandas as pd


def visible_reports(frame, asof):
    """Consolidated YTD statements, only after every supplied publication date."""
    f=frame.copy()
    for name in ['ann_date','f_ann_date','end_date']:
        if name in f:
            f[name]=pd.to_datetime(f[name].astype('string'),format='%Y%m%d',errors='coerce')
    if 'report_type' in f:
        f=f.loc[pd.to_numeric(f.report_type,errors='coerce').isin([1,4])]
    f['available']=f.ann_date
    if 'f_ann_date' in f:
        f['available']=f[['ann_date','f_ann_date']].max(axis=1)
    limit=pd.Timestamp(asof)
    f=f.loc[f.available.notna() & f.available.le(limit) & f.end_date.le(limit)]
    if 'update_flag' not in f:f['update_flag']='0'
    return f.sort_values(['available','update_flag']).drop_duplicates(['ts_code','end_date'],keep='last')


def ttm_at(frame, field, end):
    s=frame.set_index('end_date')[field]
    end=pd.Timestamp(end)
    if end not in s.index:return np.nan
    if end.month==12:return float(s[end])
    dates=[end,pd.Timestamp(end.year-1,12,31),end-pd.DateOffset(years=1)]
    if any(d not in s.index or not np.isfinite(s[d]) for d in dates):return np.nan
    return float(s[dates[0]]+s[dates[1]]-s[dates[2]])


def financial_snapshot(cashflow, indicators, income, asof, max_report_age_days=180):
    cf=visible_reports(cashflow,asof)
    fi=visible_reports(indicators,asof)
    inc=visible_reports(income,asof)
    rows=[]
    cg={s:g for s,g in cf.groupby('ts_code')}
    ig={s:g for s,g in inc.groupby('ts_code')}
    for symbol,f in fi.groupby('ts_code'):
        c=cg.get(symbol);i=ig.get(symbol)
        if c is None or i is None:continue
        end=f.end_date.max()
        if (pd.Timestamp(asof)-end).days>max_report_age_days:continue
        # No silently backdating to an older report when the latest is incomplete.
        if end not in set(c.end_date) or end not in set(i.end_date):continue
        latest=f.set_index('end_date').loc[end]
        quarters=pd.period_range(end=end.to_period('Q'),periods=12,freq='Q').to_timestamp(how='end').normalize()
        q=f.set_index('end_date').q_roe.reindex(quarters)
        stability=-float(q.std(ddof=0)) if q.notna().all() else np.nan
        last_year=end.year if end.month==12 else end.year-1
        required_annual={pd.Timestamp(y,12,31) for y in range(last_year-2,last_year+1)}
        annual=c.loc[c.end_date.isin(required_annual)].sort_values('end_date')
        complete=set(annual.end_date)==required_annual and annual.n_cashflow_act.notna().all()
        annual_positive=complete and annual.n_cashflow_act.gt(0).all()
        ocf=ttm_at(c,'n_cashflow_act',end)
        profit=ttm_at(i,'operate_profit',end)
        rows.append({'symbol':symbol,'report_end':end.strftime('%Y%m%d'),
                     'fcf':ttm_at(f,'fcff',end),'netdebt':latest.netdebt,
                     'roe_stability':stability,'annual_ocf_positive':bool(annual_positive),
                     'annual_ocf_complete':bool(complete),
                     'cash_conversion':ocf/profit if np.isfinite(profit) and profit!=0 else np.nan,
                     'last_available':max(f.available.max(),c.available.max(),i.available.max()).strftime('%Y%m%d')})
    columns=['symbol','report_end','fcf','netdebt','roe_stability','annual_ocf_positive','annual_ocf_complete','cash_conversion','last_available']
    return pd.DataFrame(rows,columns=columns).set_index('symbol')


def capped_weights(values, exposure=.995, cap=.05):
    values=values.astype(float)
    if not np.isfinite(values).all() or (values<=0).any():raise ValueError('INVALID_POSITIVE_WEIGHTS')
    if len(values)*cap<exposure-1e-12:raise ValueError('INSUFFICIENT_WEIGHT_CAPACITY')
    weights=pd.Series(0.,index=values.index)
    active=values.index
    remaining=exposure
    while len(active):
        proposal=values.loc[active]/values.loc[active].sum()*remaining
        over=proposal.gt(cap+1e-12)
        if not over.any():
            weights.loc[active]=proposal
            break
        fixed=proposal.index[over]
        weights.loc[fixed]=cap
        remaining-=cap*len(fixed)
        active=active.difference(fixed,sort=False)
    return weights


def conservative_targets(month_prices, dividend_yield, members, asof):
    # Require exactly 37 month-end prices for 36 monthly returns; no daily proxy.
    p=month_prices.loc[month_prices.index<=asof].reindex(columns=sorted(members)).tail(37)
    if len(p)!=37:raise ValueError('INSUFFICIENT_MONTHLY_CALENDAR')
    complete=p.notna().all() & p.gt(0).all()
    volatility=p.pct_change(fill_method=None).iloc[1:].std(ddof=1).where(complete)
    low=volatility.dropna().sort_values(kind='stable').head(500).index
    if len(low)<500:raise ValueError('INSUFFICIENT_LOW_VOL_UNIVERSE')
    momentum=p.iloc[-2]/p.iloc[-13]-1
    metrics=pd.DataFrame({'momentum':momentum.reindex(low),'dividend':dividend_yield.reindex(low)}).dropna()
    score=metrics.rank(pct=True).mean(axis=1)
    top=score.sort_index().sort_values(ascending=False,kind='stable').head(100)
    if len(top)!=100:raise ValueError('INSUFFICIENT_CONSERVATIVE_COVERAGE')
    return pd.Series(.995/100,index=top.index),{'complete_36_month_stocks':int(complete.sum()),'low_vol_stocks':len(low),'dividend_momentum_covered':len(metrics),'score':top.to_dict()}


def fcf_targets(attributes, finance, average_amount):
    a=attributes.join(finance,how='left').copy()
    for name in ['fcf','netdebt','roe_stability','cash_conversion']:
        a[name]=pd.to_numeric(a[name],errors='coerce')
    a['average_amount']=average_amount.reindex(a.index)
    a['enterprise_value']=a.total_mv*10000+a.netdebt
    audit={'initial':len(a)}
    a=a.loc[a.average_amount.ge(a.average_amount.quantile(.2))]
    audit['liquid']=len(a)
    a=a.loc[~a.industry.isin(['801780.SI','801790.SI','801180.SI']) & a.industry.notna()]
    audit['non_financial_property']=len(a)
    needed=['fcf','enterprise_value','roe_stability','cash_conversion']
    a=a.loc[np.isfinite(a[needed]).all(axis=1) & a.annual_ocf_positive.notna()]
    audit['financial_complete']=len(a)
    a=a.loc[a.roe_stability.ge(a.roe_stability.quantile(.1))]
    a=a.loc[a.fcf.gt(0)&a.enterprise_value.gt(0)&a.annual_ocf_positive.eq(True)]
    audit['positive_cashflow']=len(a)
    a=a.loc[a.cash_conversion.ge(a.cash_conversion.quantile(.3))]
    a['fcf_yield']=a.fcf/a.enterprise_value
    audit['conversion_pass']=len(a)
    top=a.sort_index().sort_values('fcf_yield',ascending=False,kind='stable').head(100)
    if len(top)!=100:raise ValueError('INSUFFICIENT_FCF_ELIGIBLE:'+str(audit))
    return capped_weights(top.fcf),{**audit,'selected':len(top),'fcf_yield':top.fcf_yield.to_dict()}


def cni_rebalance_days(sessions):
    result={sessions[0]}
    for year in sorted({pd.Timestamp(s).year for s in sessions}):
        for month in [3,6,9,12]:
            fridays=pd.date_range(f'{year}-{month:02d}-01',periods=31,freq='D')
            second=fridays[(fridays.month==month)&(fridays.dayofweek==4)][1].strftime('%Y%m%d')
            after=[s for s in sessions if s>second]
            if after:result.add(after[0])
    return result


def timing_filter(desired, held, residual_returns, age, mandatory_exits, announcements, max_delay=3):
    """No price lookup, no forecast refit, force release at the fixed deadline."""
    result=dict(desired);delayed=[]
    for symbol in sorted(set(desired)|set(held)):
        delta=desired.get(symbol,0)-held.get(symbol,0)
        residual=residual_returns.get(symbol,np.nan)
        if symbol in mandatory_exits or symbol in announcements or age>=max_delay or not np.isfinite(residual):continue
        if (delta>0 and residual>0) or (delta<0 and residual<0):
            result[symbol]=held.get(symbol,0);delayed.append(symbol)
    return result,delayed
