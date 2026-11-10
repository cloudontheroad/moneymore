"""Incremental point-in-time US event warehouse backed by Futu OpenD.

Every table retains source timestamps and first-observed timestamps.  Date-only
sources are conservatively eligible on the following US trading session.
Research-only: this script never invokes a trade API.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from futu import MacroRegion, Market, OpenQuoteContext, RET_OK

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE = max((ROOT / "data/us_overnight_paper_only/universe").glob("top1000-*.parquet"))
OUT = ROOT / "data/us_overnight_event_warehouse"


def save_incremental(name: str, frame: pd.DataFrame, keys: list[str], captured: str) -> dict:
    folder = OUT / name; (folder / "snapshots").mkdir(parents=True, exist_ok=True)
    canonical = folder / "canonical.parquet"
    if frame.empty and canonical.exists():
        old = pd.read_parquet(canonical)
        return {"rows": len(old), "new_snapshot_rows": 0, "path": str(canonical)}
    frame = frame.copy(); frame["first_observed_at_utc"] = captured
    # OpenD occasionally mixes numeric values and literal "N/A" in object
    # columns. Preserve the raw representation deterministically for Parquet.
    for column in frame.select_dtypes(include="object"):
        nonnull = frame[column].dropna()
        if nonnull.map(type).nunique() > 1:
            frame[column] = frame[column].map(lambda value: None if pd.isna(value) else str(value))
    stamp = captured.replace(":", "").replace("-", "")
    frame.to_parquet(folder / "snapshots" / f"{stamp}.parquet", index=False)
    if canonical.exists():
        old = pd.read_parquet(canonical)
        frame = pd.concat([old, frame], ignore_index=True, sort=False)
        # Preserve the genuinely first observation when the source is seen again.
        frame = frame.sort_values("first_observed_at_utc").drop_duplicates(keys, keep="first")
    frame.to_parquet(canonical, index=False)
    return {"rows": len(frame), "new_snapshot_rows": len(pd.read_parquet(folder / "snapshots" / f"{stamp}.parquet")),
            "path": str(canonical)}


def checked(result, source):
    if result[0] != RET_OK: raise RuntimeError(f"{source}: {result[1]}")
    return result


def chunks(start: pd.Timestamp, end: pd.Timestamp, days: int):
    cur=start
    while cur <= end:
        right=min(cur+pd.Timedelta(days=int(days)-1),end); yield cur,right; cur=right+pd.Timedelta(days=1)


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--start',default='2023-01-01'); ap.add_argument('--host',default='127.0.0.1'); ap.add_argument('--port',type=int,default=11111)
    args=ap.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    captured=datetime.now(UTC).isoformat(); start=pd.Timestamp(args.start); end=pd.Timestamp.now(tz='UTC').tz_localize(None).normalize()
    universe=pd.read_parquet(UNIVERSE); symbols=universe.code.astype(str).tolist(); failures=[]; results={}
    q=OpenQuoteContext(host=args.host,port=args.port)
    try:
        # Earnings API permits at most seven calendar days per request.
        frames=[]
        for left,right in chunks(start,end,7):
            try:
                data=checked(q.get_earnings_calendar(Market.US,begin_date=str(left.date()),end_date=str(right.date())),'earnings')[1]
                if len(data): frames.append(data)
            except Exception as exc: failures.append({'source':'earnings','range':f'{left.date()}:{right.date()}','error':str(exc)})
            time.sleep(.06)
        earnings=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=['security','earnings_date','period_text'])
        if len(earnings):
            earnings=earnings[earnings.security.isin(symbols)].copy()
            earnings['event_at_utc']=pd.to_datetime(earnings.earnings_timestamp,unit='s',utc=True,errors='coerce')
            for c in ['eps_actual','eps_predict','revenue_actual','revenue_predict','ebit_actual','ebit_predict']:
                earnings[c]=pd.to_numeric(earnings[c],errors='coerce')
            earnings['eps_surprise_pct']=(earnings.eps_actual-earnings.eps_predict)/earnings.eps_predict.abs().replace(0,np.nan)
            earnings['revenue_surprise_pct']=(earnings.revenue_actual-earnings.revenue_predict)/earnings.revenue_predict.abs().replace(0,np.nan)
            earnings['source']='FUTU_OPEND_EARNINGS_CALENDAR'
        results['earnings']=save_incremental('earnings',earnings,['security','earnings_date','period_text'],captured)

        # Economic calendar is paginated and carries actual/consensus/previous.
        econ=[]
        for left,right in chunks(start,end,30):
            cursor=None
            while True:
                try:
                    ret,data,cursor,more=checked(q.get_economic_calendar(str(left.date()),str(right.date()),market_list=[Market.US],count=100,next_page=cursor),'economic')
                    if len(data): econ.append(data)
                    if not more: break
                except Exception as exc: failures.append({'source':'economic','range':f'{left.date()}:{right.date()}','error':str(exc)}); break
        economic=pd.concat(econ,ignore_index=True) if econ else pd.DataFrame(columns=['title','timestamp'])
        if len(economic):
            economic['event_at_utc']=pd.to_datetime(economic.timestamp,unit='s',utc=True,errors='coerce')
            economic['source']='FUTU_OPEND_ECONOMIC_CALENDAR'
        results['economic']=save_incremental('economic',economic,['title','timestamp'],captured)

        # Macro histories include release timestamps and actual/forecast values.
        macro_list=checked(q.get_macro_indicator_list(MacroRegion.US),'macro_list')[1]
        histories=[]
        for row in macro_list.itertuples(index=False):
            try:
                data=checked(q.get_macro_indicator_history(int(row.indicator_id),time=str(end.date()),max_count=1000),f'macro:{row.indicator_id}')[1]
                if len(data):
                    data=data.copy(); data['indicator_id']=int(row.indicator_id); data['indicator_name']=row.name; data['category_name']=row.category_name; histories.append(data)
            except Exception as exc: failures.append({'source':'macro','indicator_id':int(row.indicator_id),'error':str(exc)})
            time.sleep(.04)
        macro=pd.concat(histories,ignore_index=True) if histories else pd.DataFrame(columns=['indicator_id'])
        for column in ('actual_value','predict_value','previous_value','revised_value'):
            if column in macro: macro[column+'_numeric']=pd.to_numeric(macro[column],errors='coerce')
        macro['source']='FUTU_OPEND_MACRO_HISTORY'
        macro_keys=['indicator_id']+[c for c in ('data_time','release_time','time','timestamp','period') if c in macro.columns]
        results['macro']=save_incremental('macro',macro,macro_keys,captured)

        # Current ratings endpoint is cursor-paginated. Date-only events are not
        # eligible intraday; downstream must shift them to the next session.
        ratings=[]; cursor=None
        for _ in range(500):
            ret,data,cursor,total=checked(q.get_rating_change(Market.US,count=20,page=cursor),'ratings')
            if len(data): ratings.append(data)
            if not cursor or sum(len(x) for x in ratings)>=total: break
        rating=pd.concat(ratings,ignore_index=True) if ratings else pd.DataFrame(columns=['security','recommendation_date','institution_name'])
        if len(rating):
            rating=rating[rating.security.isin(symbols)].copy(); rating['event_date']=pd.to_datetime(rating.recommendation_date,errors='coerce')
            rating['availability_policy']='NEXT_US_SESSION_DATE_ONLY'; rating['source']='FUTU_OPEND_RATING_CHANGE'
        results['ratings']=save_incremental('ratings',rating,['security','recommendation_date','institution_name','rating','target_price'],captured)

        # Point-in-time snapshots of provider plate membership. Never backfill this
        # current mapping into dates before first_observed_at_utc.
        plates=[]
        def fetch_plate_batch(batch):
            try:
                data=checked(q.get_owner_plate(batch),'owner_plate')[1]
                if len(data): plates.append(data)
            except Exception as exc:
                if len(batch) == 1:
                    failures.append({'source':'plates','symbol':batch[0],'error':str(exc)})
                else:
                    middle=len(batch)//2; fetch_plate_batch(batch[:middle]); fetch_plate_batch(batch[middle:])
            time.sleep(3.1)
        for i in range(0,len(symbols),100):
            fetch_plate_batch(symbols[i:i+100])
        plate=pd.concat(plates,ignore_index=True) if plates else pd.DataFrame(columns=['code','plate_code'])
        plate['source']='FUTU_OPEND_OWNER_PLATE_SNAPSHOT'
        results['plates']=save_incremental('plates',plate,['code','plate_code'],captured)

        # News and material corporate events are intentionally absent here.
        # They are sparse, so they are discovered market-wide with WebSearch and
        # imported by import_us_web_event_discovery.py after entity mapping.
    finally: q.close()

    manifest={'captured_at_utc':captured,'universe':str(UNIVERSE),'universe_symbols':len(symbols),'range':[str(start.date()),str(end.date())],
              'tables':results,'failures':failures,'causal_contract':{
               'timestamped':'usable only when event_at_utc < decision timestamp',
               'date_only':'usable from next US regular session',
               'snapshot_membership':'never valid before first_observed_at_utc'}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str),encoding='utf-8')
    print(json.dumps(manifest,indent=2,default=str)); return 0 if not failures else 2

if __name__=='__main__': raise SystemExit(main())
