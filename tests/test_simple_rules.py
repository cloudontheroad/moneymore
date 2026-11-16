import numpy as np
import pandas as pd
import pytest
from moneymore.research.simple_rules import (visible_reports,ttm_at,capped_weights,
    conservative_targets,cni_rebalance_days,timing_filter,financial_snapshot,fcf_targets)


def test_actual_announcement_is_not_backdated():
    f=pd.DataFrame({'ts_code':['A']*3,'ann_date':['20250401','20250401','20250401'],
        'f_ann_date':['20250401','20250501','20250401'],'end_date':['20241231']*3,
        'report_type':['1','4','2'],'x':[1,999,777]})
    assert visible_reports(f,'20250415').x.tolist()==[1]
    assert visible_reports(f,'20250502').x.tolist()==[999]


def test_missing_announcement_is_not_visible():
    f=pd.DataFrame({'ts_code':['A'],'ann_date':[None],'end_date':['20241231']})
    assert visible_reports(f,'20250502').empty


def test_ytd_ttm_not_sum_of_cumulative_reports():
    f=pd.DataFrame({'end_date':pd.to_datetime(['20240930','20241231','20250930']),'x':[60.,100.,90.]})
    assert ttm_at(f,'x','20250930')==130
    assert ttm_at(f,'x','20241231')==100
    assert np.isnan(ttm_at(f.iloc[1:],'x','20250930'))


def test_capped_weights_and_capacity():
    w=capped_weights(pd.Series([1000.]+[1.]*99))
    assert w.sum()==pytest.approx(.995)
    assert w.max()==pytest.approx(.05)
    with pytest.raises(ValueError,match='CAPACITY'):capped_weights(pd.Series([1.]*10))


def test_cni_calendar_not_calendar_quarter():
    days=pd.bdate_range('20260105','20260930').strftime('%Y%m%d').tolist()
    assert cni_rebalance_days(days)=={'20260105','20260316','20260615','20260914'}


@pytest.mark.parametrize('age,delay',[(0,True),(1,True),(2,True),(3,False),(4,False)])
def test_timing_bounded_and_mandatory_sell_never_delayed(age,delay):
    q,d=timing_filter({'A':100,'B':0,'C':0},{'B':100,'C':100},pd.Series({'A':.1,'B':-.1,'C':-.1}),age,{'C'},set())
    assert q['C']==0
    assert ('A' in d)==delay
    assert ('B' in d)==delay


def test_unknown_or_earnings_does_not_invent_timing():
    q,d=timing_filter({'A':100,'B':100},{},pd.Series({'A':np.nan,'B':.1}),0,set(),{'B'})
    assert q=={'A':100,'B':100} and d==[]


def test_conservative_requires_36_months_and_ignores_future():
    rng=np.random.default_rng(17)
    p=pd.DataFrame(np.exp(np.cumsum(rng.normal(.001,.03,(38,600)),axis=0)),index=pd.date_range('20221231',periods=38,freq='ME').strftime('%Y%m%d'),columns=[str(i) for i in range(600)])
    div=pd.Series(1.,index=p.columns)
    w,a=conservative_targets(p,div,set(p.columns),'20251231')
    p.iloc[-1]=1e9
    w2,_=conservative_targets(p,div,set(p.columns),'20251231')
    pd.testing.assert_series_equal(w,w2)
    assert len(w)==100 and a['complete_36_month_stocks']==600
    with pytest.raises(ValueError,match='MONTHLY_CALENDAR'):conservative_targets(p,div,set(p.columns),'20251130')


def test_fcf_finance_exclusion_and_currency_units():
    ix=[str(i) for i in range(200)]
    attrs=pd.DataFrame({'total_mv':100.,'industry':'801110.SI'},index=ix)
    attrs.loc['0','industry']='801780.SI'
    f=pd.DataFrame({'fcf':100_000.,'netdebt':1_000_000.,'roe_stability':-1.,'cash_conversion':1.,'annual_ocf_positive':True},index=ix)
    w,a=fcf_targets(attrs,f,pd.Series(1.,index=ix))
    assert '0' not in w.index and len(w)==100
    assert next(iter(a['fcf_yield'].values()))==pytest.approx(.05)
    f['netdebt']=np.nan
    with pytest.raises(ValueError,match='INSUFFICIENT_FCF'):fcf_targets(attrs,f,pd.Series(1.,index=ix))


def test_financials_do_not_substitute_latest_year_for_three_years():
    ends=pd.period_range('2022Q4','2025Q3',freq='Q').to_timestamp(how='end').normalize()
    f=pd.DataFrame({'ts_code':'A','ann_date':'20251101','end_date':ends.strftime('%Y%m%d'),'q_roe':1.,'fcff':100.,'netdebt':50.})
    c=f[['ts_code','ann_date','end_date']].assign(report_type='1',n_cashflow_act=200.)
    inc=c.assign(operate_profit=100.)
    result=financial_snapshot(c,f,inc,'20251231')
    assert result.loc['A','annual_ocf_positive']
    assert result.loc['A','roe_stability']==0
    assert result.loc['A','fcf']==100.
    result=financial_snapshot(c.loc[c.end_date.ne('20221231')],f,inc,'20251231')
    assert not result.loc['A','annual_ocf_positive']
    # Three old annual reports cannot substitute for a missing required latest year.
    old=c.iloc[[0]].copy();old['end_date']='20211231'
    result=financial_snapshot(pd.concat([old,c.loc[c.end_date.ne('20241231')]]),f,inc,'20251231')
    assert not result.loc['A','annual_ocf_complete']


def test_old_financials_cannot_compete_as_current():
    f=pd.DataFrame({'ts_code':['A'],'ann_date':['20241101'],'end_date':['20240930'],'q_roe':[1.],'fcff':[100.],'netdebt':[50.]})
    c=f.assign(report_type='1',n_cashflow_act=200.,operate_profit=100.)
    # The entire available history is stale: no current score is manufactured.
    result=financial_snapshot(c,f,c,'20251231')
    assert result.empty


def test_future_annual_result_cannot_repair_ttm_history():
    ends=pd.period_range('2022Q4','2025Q3',freq='Q').to_timestamp(how='end').normalize()
    f=pd.DataFrame({'ts_code':'A','ann_date':'20251101','end_date':ends.strftime('%Y%m%d'),'q_roe':1.,'fcff':100.,'netdebt':50.})
    c=f[['ts_code','ann_date','end_date']].assign(report_type='1',n_cashflow_act=200.)
    inc=c.assign(operate_profit=100.)
    future=f.iloc[[-1]].assign(ann_date='20260301',end_date='20251231',fcff=9999999.)
    a=financial_snapshot(c,f,inc,'20251231')
    b=financial_snapshot(c,pd.concat([f,future]),inc,'20251231')
    pd.testing.assert_frame_equal(a,b)


def test_missing_netdebt_not_interpreted_as_zero():
    ix=[str(i) for i in range(200)]
    attrs=pd.DataFrame({'total_mv':100.,'industry':'801110.SI'},index=ix)
    f=pd.DataFrame({'fcf':100_000.,'netdebt':1_000_000.,'roe_stability':-1.,'cash_conversion':1.,'annual_ocf_positive':True},index=ix)
    f.loc['0',['fcf','netdebt']]=[1e20,np.nan]
    w,a=fcf_targets(attrs,f,pd.Series(1.,index=ix))
    assert '0' not in w.index and a['financial_complete']==199


def test_empty_financial_universe_is_explicitly_unavailable():
    f=pd.DataFrame(columns=['fcf','netdebt','roe_stability','cash_conversion','annual_ocf_positive'])
    attrs=pd.DataFrame({'total_mv':[100.],'industry':['801110.SI']},index=['A'])
    with pytest.raises(ValueError,match='INSUFFICIENT_FCF'):
        fcf_targets(attrs,f,pd.Series({'A':100.}))
