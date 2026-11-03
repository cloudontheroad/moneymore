import pandas as pd
from moneymore.data.global_financials import latest_global_financials


def test_global_financials_retains_valid_field_and_does_not_use_future_disclosure():
    frame=pd.DataFrame([
        {'ts_code':'688009.SH','ann_date':'20260820','end_date':'20260630','roe':5.,'q_sales_yoy':10.},
        {'ts_code':'688009.SH','ann_date':'20261029','end_date':'20260930','roe':8.,'q_sales_yoy':None},
        {'ts_code':'688009.SH','ann_date':'20261104','end_date':'20260930','roe':99.,'q_sales_yoy':99.}])
    dates=pd.DataFrame({'symbol':['688009.SH'],'latest_date':[pd.Timestamp('2026-11-03')]})
    row=latest_global_financials(frame,dates).loc['688009.SH']
    assert row.roe==8. and row.q_sales_yoy==10.
    assert row.q_sales_yoy_end_date=='20260630'


def test_global_financials_does_not_reuse_years_old_growth():
    frame=pd.DataFrame([
        {'ts_code':'600816.SH','ann_date':'20190831','end_date':'20190630','roe':2.,'q_sales_yoy':-1756.},
        {'ts_code':'600816.SH','ann_date':'20261028','end_date':'20260930','roe':7.,'q_sales_yoy':None}])
    dates=pd.DataFrame({'symbol':['600816.SH'],'latest_date':[pd.Timestamp('2026-11-03')]})
    row=latest_global_financials(frame,dates).loc['600816.SH']
    assert row.roe==7. and pd.isna(row.q_sales_yoy) and row.q_sales_yoy_stale


def test_global_financials_empty_symbol_is_missing_not_zero():
    frame=pd.DataFrame(columns=['ts_code','ann_date','end_date','roe','q_sales_yoy'])
    dates=pd.DataFrame({'symbol':['000001.SZ'],'latest_date':[pd.Timestamp('2026-11-03')]})
    row=latest_global_financials(frame,dates).loc['000001.SZ']
    assert pd.isna(row.roe) and pd.isna(row.q_sales_yoy)
