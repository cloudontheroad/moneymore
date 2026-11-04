"""Bounded same-window official price-index comparison; no production writes."""
from pathlib import Path
import json,time,hashlib
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from moneymore.data.tushare_provider import TushareProvider
from moneymore.research.period_statistics import period_statistics

root=Path(__file__).resolve().parents[1];study=root/'state/four-strategy-2026-jan-sep-v1'
out=study/'index-comparison';out.mkdir(exist_ok=True)
load_dotenv(root/'.env');client=TushareProvider()
result=[]
for code,name in [('000001.SH','上证指数'),('399001.SZ','深证成指'),('399006.SZ','创业板指'),('000688.SH','科创50')]:
    path=out/(code+'.parquet')
    if path.exists():frame=pd.read_parquet(path)
    else:
        frame=client.index_daily(code,'20251231','20260930');frame.to_parquet(path,index=False);time.sleep(1.1)
    frame=frame.sort_values('trade_date');frame.trade_date=frame.trade_date.astype(str)
    assert not frame.trade_date.duplicated().any() and frame.trade_date.iloc[0]=='20251231' and frame.trade_date.iloc[-1]=='20260930'
    assert len(frame)==182 and np.isfinite(frame.close).all()
    base=float(frame.close.iloc[0]);values=frame.close.iloc[1:]
    month=frame.assign(month=frame.trade_date.str[:6]).groupby('month').tail(1)
    monthly=month.close.pct_change().iloc[1:]
    result.append({'name':name,'code':code,'start_close':base,'end_close':float(frame.close.iloc[-1]),
        **period_statistics(values,base),'monthly_returns':dict(zip(month.month.iloc[1:],monthly)),
        'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
strategies=json.loads((study/'results.json').read_bytes())['cases']
report={'status':'COMPLETE_SAME_WINDOW_PRICE_INDEX_COMPARISON','window':['20251231','20260930'],
    'indexes':result,'strategies':[{**r,'arithmetic_relative_returns_pp':{b['code']:(r['net_return']-b['net_return'])*100 for b in result}} for r in strategies],
    'method':'indexes are price returns without dividends or trading costs; strategy NAV includes cash rights and execution costs; not benchmark-adjusted alpha',
    'production_modified':False}
(out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=True))
