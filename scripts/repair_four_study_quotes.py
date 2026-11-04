"""Fetch only the identified isolated-study held exit's missing bars."""
from pathlib import Path
import json,hashlib
from dotenv import load_dotenv
from moneymore.data.tushare_provider import TushareProvider
root=Path(__file__).resolve().parents[1];out=root/'state/four-strategy-2026-jan-sep-v1/official'
load_dotenv(root/'.env');client=TushareProvider()._client
rows=client.daily(ts_code='002405.SZ',start_date='20260907',end_date='20260930')
assert rows is not None and not rows.empty and rows.ts_code.eq('002405.SZ').all()
adjustment=client.adj_factor(ts_code='002405.SZ',start_date='20260907',end_date='20260930')
rows=rows.merge(adjustment[['ts_code','trade_date','adj_factor']],on=['ts_code','trade_date'],validate='one_to_one')
assert len(rows)==17 and rows.adj_factor.notna().all()
path=out/'held-exit-002405-202609.parquet';rows.to_parquet(path,index=False)
(out/'held-exit-002405-202609.json').write_text(json.dumps({'status':'OFFICIAL_EXIT_QUOTES','rows':len(rows),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'production_modified':False}))
print(json.dumps({'rows':len(rows),'first':rows.trade_date.min(),'last':rows.trade_date.max()}))
