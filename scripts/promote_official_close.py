"""Promote a hash-verified official close bundle; never runs strategies or orders."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd
from moneymore.data.store import ParquetStore
from moneymore.data.quality import validate_daily_bars, validate_adjustment_factors


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];store=ParquetStore(root/'data')
    manifest=json.loads((args.bundle/'results.json').read_bytes());day=manifest['date']
    if manifest['status']!='OFFICIAL_CLOSE_READY':raise ValueError('BUNDLE_NOT_READY')
    frames={}
    for name,digest in manifest['hashes'].items():
        path=args.bundle/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('BUNDLE_HASH_MISMATCH:'+name)
        frame=pd.read_parquet(path)
        if not frame.empty and not frame.trade_date.astype(str).eq(day).all():raise ValueError('DATE_MISMATCH')
        frames[path.stem]=frame
    validate_daily_bars(frames['daily'],day);validate_adjustment_factors(frames['adj_factor'],day)
    mapping={'daily':'daily','adj_factor':'adj_factor','daily_basic':'daily_basic','stk_limit':'stock_limits','suspend_d':'suspensions'}
    backup=root/'state/official-close-promotion'/(day+'-'+datetime.now().strftime('%H%M%S'))
    backup.mkdir(parents=True,exist_ok=False);results=[]
    for endpoint,table in mapping.items():
        frame=frames[endpoint];keys=['ts_code','trade_date']+(['suspend_type'] if endpoint=='suspend_d' else [])
        if frame.empty:continue
        if frame.duplicated(keys).any():raise ValueError('DUPLICATE_KEYS:'+endpoint)
        path=store.curated/(table+'.parquet');before=0
        if path.exists():
            shutil.copy2(path,backup/path.name)
            old=store.read(table,filters=[('trade_date','==',day)])
            before=len(old)
            if not old.empty:
                common=[c for c in old.columns if c in frame.columns]
                joined=old[common].merge(frame[common],on=keys,suffixes=('_old','_new'),validate='one_to_one')
                for c in set(common)-set(keys):
                    a,b=joined[c+'_old'],joined[c+'_new']
                    if not (a.eq(b)|(a.isna()&b.isna())).all():raise ValueError('EXISTING_CLOSE_CONFLICT:'+table+':'+c)
        store.save_snapshot(table,frame,'tushare',keys,day if endpoint in {'daily','adj_factor'} else 'official_close_'+day)
        # Also repairs an interrupted raw->curated promotion on retry.
        store.merge_curated(table,[frame],keys)
        after=store.read(table,filters=[('trade_date','==',day)])
        if len(after)!=len(frame) or after.duplicated(keys).any():raise ValueError('PROMOTION_COUNT_MISMATCH:'+table)
        results.append({'table':table,'before_rows':before,'after_rows':len(after)})
        print(json.dumps(results[-1]),flush=True)
    result={'status':'OFFICIAL_CLOSE_PROMOTED','date':day,'tables':results,'backup':str(backup),'strategy_or_order_execution':False}
    (backup/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
