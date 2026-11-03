"""Hash-bound isolated held-risk supplements, never a production data fallback."""
import hashlib
import json
import pandas as pd


def read_held_basic(folder):
    manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['status']!='READY':raise ValueError('HELD_EVIDENCE_NOT_READY')
    frames=[]
    for row in manifest['requests']:
        file=folder/row['file']
        if hashlib.sha256(file.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('HELD_EVIDENCE_CHANGED')
        if row['endpoint']!='daily_basic':continue
        frame=pd.read_parquet(file);params=row['params']
        if frame.empty or not frame.trade_date.astype(str).eq(params['trade_date']).all() or ('ts_code' in params and (len(frame)!=1 or not frame.ts_code.eq(params['ts_code']).all())):
            raise ValueError('HELD_EVIDENCE_IDENTITY_MISMATCH')
        if not pd.to_numeric(frame.total_mv,errors='coerce').gt(0).all():raise ValueError('HELD_EVIDENCE_INVALID_SIZE')
        frames.append(frame)
    if not frames:raise ValueError('HELD_BASIC_EMPTY')
    result=pd.concat(frames,ignore_index=True)
    if result.duplicated(['ts_code','trade_date']).any():raise ValueError('HELD_BASIC_DUPLICATE')
    return result
