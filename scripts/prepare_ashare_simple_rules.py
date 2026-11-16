"""Bounded official endpoint audit/cache for simple-rule research; no live writes."""
import argparse
import json
import time
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from moneymore.data.tushare_provider import TushareProvider
from prepare_four_strategy_2026 import save

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'state/ashare-simple-rules-20261113/data-v1'
FIN_FIELDS={
 'cashflow_vip':'ts_code,ann_date,f_ann_date,end_date,report_type,update_flag,n_cashflow_act,c_pay_acq_const_fiolta,c_pay_dist_dpcp_int_exp,c_recp_cap_contrib,free_cashflow',
 'balancesheet_vip':'ts_code,ann_date,f_ann_date,end_date,report_type,update_flag,money_cap,total_liab,total_share,treasury_share,total_hldr_eqy_exc_min_int',
 'income_vip':'ts_code,ann_date,f_ann_date,end_date,report_type,update_flag,operate_profit,n_income_attr_p',
 'fina_indicator_vip':'ts_code,ann_date,end_date,update_flag,roe,q_roe,interestdebt,netdebt,fcff',
}


def fetch(client,endpoint,tag,**params):
    path=OUT/f'{endpoint}-{tag}.parquet'
    if path.exists():return pd.read_parquet(path)
    for attempt in range(2):
        try:
            frame=getattr(client,endpoint)(**params)
            if frame is None:raise ValueError('NONE_RESPONSE')
            frame.to_parquet(path,index=False)
            return frame
        except Exception as exc:
            if attempt:
                msg=str(exc)
                category='PERMISSION' if any(s in msg for s in ['权限','积分','permission']) else 'RATE_LIMIT' if any(s in msg for s in ['频次','每分钟','rate limit']) else 'OTHER'
                save(OUT/f'{endpoint}-{tag}.failure.json',{'endpoint':endpoint,'params':params,'error_type':type(exc).__name__,'category':category,'error_body_suppressed':True})
                return None
            time.sleep(2)
        finally:time.sleep(.55)


def bounded_data():
    """One serial stream; permission failures stop that endpoint, no VIP retries."""
    OUT.mkdir(parents=True,exist_ok=True)
    load_dotenv(ROOT/'.env')
    client=TushareProvider()._client
    cal=fetch(client,'trade_cal','20221201-20260930',exchange='SSE',start_date='20221201',end_date='20260930',is_open='1')
    if cal is None:raise ValueError('CALENDAR_UNAVAILABLE')
    days=cal.cal_date.astype(str).sort_values()
    for day in days.groupby(days.str[:6]).max():
        for endpoint in ['monthly','adj_factor']:
            if fetch(client,endpoint,day,trade_date=day) is None:raise ValueError('MONTHLY_UNAVAILABLE')
        print('MONTH_COMPLETE',day,flush=True)
    panel=pd.read_parquet(ROOT/'state/ashare-mainline-20261112/panel-v2/panel.parquet',columns=['date','symbol','industry'])
    # Fixed review dates, selected before observing any strategy returns.
    reviews=['20251231','20260313','20260612','20260911']
    pool=panel.loc[panel.date.isin(reviews)]
    pool=pool.loc[~pool.industry.isin(['801780.SI','801790.SI','801180.SI'])]
    symbols=sorted(pool.symbol.unique())
    save(OUT/'fetch-contract.json',{'symbols':symbols,'review_dates':reviews,'endpoints':['cashflow','fina_indicator','income'],'max_requests':len(symbols)*3+len(days.groupby(days.str[:6]))*2,'serial':True,'sleep_seconds':.55,'production_modified':False})
    completed=0
    consecutive=0
    for symbol in symbols:
        for endpoint in ['cashflow','fina_indicator','income']:
            fields=FIN_FIELDS[endpoint+'_vip']
            f=fetch(client,endpoint,symbol,ts_code=symbol,start_date='20220101',end_date='20260930',fields=fields)
            completed+=1
            if f is None:
                consecutive+=1
                failure=json.loads((OUT/f'{endpoint}-{symbol}.failure.json').read_bytes())
                if failure.get('category')=='PERMISSION' or consecutive>=3:
                    raise ValueError('BOUNDED_FETCH_CIRCUIT_OPEN:'+endpoint)
            else:consecutive=0
        save(OUT/'financial-progress.json',{'status':'RUNNING','completed_requests':completed,'total_requests':len(symbols)*3,'last_symbol':symbol})
        if completed%30==0:print('FINANCIAL_PROGRESS',completed,len(symbols)*3,flush=True)
    save(OUT/'financial-progress.json',{'status':'COMPLETE','completed_requests':completed,'total_requests':len(symbols)*3,'production_modified':False})


def main(mode):
    OUT.mkdir(parents=True,exist_ok=True)
    load_dotenv(ROOT/'.env')
    client=TushareProvider()._client
    reports=[]
    quarters=['20250930'] if mode=='probe' else [p.end_time.strftime('%Y%m%d') for p in pd.period_range('2022Q4','2026Q2',freq='Q')]
    for period in quarters:
        for endpoint,fields in FIN_FIELDS.items():
            f=fetch(client,endpoint,period,period=period,fields=fields,limit=10000)
            row={'endpoint':endpoint,'period':period,'rows':None if f is None else len(f),'columns':[] if f is None else f.columns.tolist(),'nonnull':{} if f is None else f.notna().sum().to_dict()}
            reports.append(row)
            print(json.dumps({'endpoint':endpoint,'period':period,'rows':row['rows']}),flush=True)
            save(OUT/'progress.json',{'status':'RUNNING','completed':len(reports),'total':len(quarters)*4})
    cal=fetch(client,'trade_cal','20221201-20260930',exchange='SSE',start_date='20221201',end_date='20260930',is_open='1')
    if cal is not None:
        days=cal.cal_date.astype(str).sort_values()
        ends=days.groupby(days.str[:6]).max().tolist()
        if mode=='probe':ends=ends[:1]
        for day in ends:
            for endpoint in ['monthly','adj_factor','daily_basic']:
                f=fetch(client,endpoint,day,trade_date=day)
                reports.append({'endpoint':endpoint,'day':day,'rows':None if f is None else len(f)})
                print(json.dumps(reports[-1]),flush=True)
    save(OUT/f'{mode}-results.json',{'status':'FETCH_FINISHED_REQUIRES_COVERAGE_AUDIT','results':reports,'production_modified':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['probe','bounded'],default='probe');a=p.parse_args()
    if a.mode=='bounded':bounded_data()
    else:main(a.mode)
