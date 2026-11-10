"""Import sparse web-search discoveries into the point-in-time warehouse.

Input is JSONL produced by a web-search pass. Required fields are source_url,
title, published_at_utc, symbols, and event_type. No article body is copied.
"""
from __future__ import annotations
import argparse, json
from datetime import UTC, datetime
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
WAREHOUSE=ROOT/'data/us_overnight_event_warehouse/web_events'
UNIVERSE=max((ROOT/'data/us_overnight_paper_only/universe').glob('top1000-*.parquet'))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('jsonl',type=Path); args=ap.parse_args()
    allowed=set(pd.read_parquet(UNIVERSE).code.astype(str)); now=datetime.now(UTC).isoformat(); rows=[]
    for line in args.jsonl.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        item=json.loads(line); missing={'source_url','title','published_at_utc','symbols','event_type'}-item.keys()
        if missing: raise ValueError(f'missing fields: {sorted(missing)}')
        symbols=['US.'+s.upper().removeprefix('US.') for s in item['symbols']]
        matched=sorted(set(symbols)&allowed)
        if not matched: continue
        rows.append({**item,'symbols':matched,'published_at_utc':pd.to_datetime(item['published_at_utc'],utc=True),
                     'discovered_at_utc':item.get('discovered_at_utc',now)})
    frame=pd.DataFrame(rows); WAREHOUSE.mkdir(parents=True,exist_ok=True)
    if frame.empty: print(json.dumps({'imported':0})); return 0
    stamp=now.replace(':','').replace('-',''); frame.to_parquet(WAREHOUSE/f'{stamp}.parquet',index=False)
    canonical=WAREHOUSE/'canonical.parquet'
    if canonical.exists(): frame=pd.concat([pd.read_parquet(canonical),frame],ignore_index=True,sort=False)
    frame=frame.sort_values('discovered_at_utc').drop_duplicates(['source_url','published_at_utc'],keep='first')
    frame.to_parquet(canonical,index=False); print(json.dumps({'imported':len(rows),'canonical_rows':len(frame)})); return 0
if __name__=='__main__': raise SystemExit(main())
