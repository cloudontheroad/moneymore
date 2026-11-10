"""Coverage and point-in-time quality audit for the US event warehouse."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; W=ROOT/'data/us_overnight_event_warehouse'
UNIVERSE=max((ROOT/'data/us_overnight_paper_only/universe').glob('top1000-*.parquet'))

def main():
    symbols=set(pd.read_parquet(UNIVERSE).code.astype(str)); report={}
    symbol_cols={'earnings':'security','ratings':'security','plates':'code','news':'query_symbol'}
    time_cols={'earnings':'event_at_utc','economic':'event_at_utc','macro':'release_time','ratings':'event_date'}
    for name in ('earnings','economic','macro','ratings','plates','news'):
        path=W/name/'canonical.parquet'; frame=pd.read_parquet(path)
        comparable=frame.map(lambda value: repr(value) if not isinstance(value,(str,int,float,bool,type(None),pd.Timestamp)) else value)
        item={'rows':len(frame),'columns':list(frame.columns),'duplicate_rows':int(comparable.duplicated().sum())}
        if name in symbol_cols:
            present=set(frame[symbol_cols[name]].dropna().astype(str)); item['universe_symbols_covered']=len(present&symbols); item['coverage_pct']=len(present&symbols)/len(symbols)
        if name in time_cols and time_cols[name] in frame:
            ts=pd.to_datetime(frame[time_cols[name]],errors='coerce',utc=True); item['null_time_pct']=float(ts.isna().mean()); item['min_time']=str(ts.min()); item['max_time']=str(ts.max())
        item['first_observed_null_pct']=float(frame.first_observed_at_utc.isna().mean())
        report[name]=item
    (W/'quality_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
