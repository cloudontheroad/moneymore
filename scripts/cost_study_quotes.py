"""Bounded held-security quote repair, isolated cache and serialized requests."""
from pathlib import Path
import time
import pandas as pd
from filelock import FileLock

def ensure_quotes(root,bars,symbols,day):
    folder=root/'state/cost-aware-ab-v2/quote-cache';folder.mkdir(parents=True,exist_ok=True)
    missing=set(symbols)-set(bars.loc[bars.trade_date.astype(str).eq(day),'ts_code'])
    if not missing:return bars
    from dotenv import load_dotenv
    from moneymore.data.tushare_provider import TushareProvider
    load_dotenv(root/'.env')
    for symbol in sorted(missing):
        path=folder/(symbol+'-2025-2026.parquet')
        with FileLock(str(folder/'request.lock'),timeout=90):
            if path.exists():extra=pd.read_parquet(path)
            else:
                if len(list(folder.glob('*.parquet')))>=60:raise ValueError('BOUNDED_MONITOR_QUOTE_BUDGET_REACHED')
                client=TushareProvider()._client
                try:
                    data=client.daily(ts_code=symbol,start_date='20250101',end_date='20260930');time.sleep(1.1)
                    adj=client.adj_factor(ts_code=symbol,start_date='20250101',end_date='20260930');time.sleep(1.1)
                except Exception as error:raise ValueError('OFFICIAL_MONITOR_QUOTE_REQUEST_FAILED:'+type(error).__name__) from None
                extra=data.merge(adj[['ts_code','trade_date','adj_factor']],on=['ts_code','trade_date'],validate='one_to_one')
                if extra.empty or not extra.ts_code.eq(symbol).all():raise ValueError('OFFICIAL_MONITOR_QUOTE_EMPTY')
                extra.to_parquet(path,index=False)
        bars=pd.concat([bars,extra],ignore_index=True).drop_duplicates(['trade_date','ts_code'],keep='first')
    return bars
