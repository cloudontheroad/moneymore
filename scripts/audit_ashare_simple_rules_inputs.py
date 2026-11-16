"""Freeze start-hash-verified source copies and audit the downloaded data contract."""
import argparse
import json
from pathlib import Path
import pandas as pd
from moneymore.research.ashare_mainline import sha256
from prepare_four_strategy_2026 import save

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'state/ashare-simple-rules-20261113'


def freeze(run):
    runtime=json.loads((run/'runtime.json').read_bytes())
    for name,digest in runtime['source_hashes'].items():
        path=Path(name);dest=run/'sources'/path.relative_to(ROOT)
        if dest.exists():
            if sha256(dest)!=digest:raise ValueError('FROZEN_SOURCE_CORRUPT')
            continue
        if sha256(path)!=digest:raise ValueError('SOURCE_CHANGED_BEFORE_FREEZE:'+name)
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(path.read_bytes())
    save(run/'source-freeze.json',{'matched_start_hashes':True,'files':runtime['source_hashes']})


def data_audit():
    data=BASE/'data-v1'
    contract=json.loads((data/'fetch-contract.json').read_bytes())
    status=json.loads((data/'financial-progress.json').read_bytes())
    if status['status']!='COMPLETE':raise ValueError('DOWNLOAD_NOT_COMPLETE')
    results=[]
    for endpoint in contract['endpoints']:
        for symbol in contract['symbols']:
            path=data/f'{endpoint}-{symbol}.parquet'
            if not path.exists():raise ValueError('MISSING_RESPONSE:'+path.name)
            f=pd.read_parquet(path)
            if endpoint=='fina_indicator' and len(f)>=100:raise ValueError('POSSIBLE_TRUNCATION:'+path.name)
            if not f.empty and set(f.ts_code)!={symbol}:raise ValueError('WRONG_SYMBOL:'+path.name)
            results.append({'file':path.name,'rows':len(f),'empty_response_excluded_not_imputed':f.empty,'sha256':sha256(path),'min_ann':str(f.ann_date.min()) if not f.empty else None,'max_ann':str(f.ann_date.max()) if not f.empty else None})
    native=pd.read_parquet(ROOT/'state/mature-multifactor/p1-native-20261103-a/bars-input.parquet',columns=['ts_code','trade_date','close','adj_factor'])
    native.trade_date=native.trade_date.astype(str)
    overlap=[];factor_rows=[]
    for path in sorted(data.glob('monthly-*.parquet')):
        day=path.stem.split('-')[1]
        m=pd.read_parquet(path)
        if m.ts_code.duplicated().any() or len(m)<4000 or set(m.trade_date)!={day}:raise ValueError('INCOMPLETE_MONTH:'+day)
        adj=pd.read_parquet(data/f'adj_factor-{day}.parquet')
        check=m.merge(adj[['ts_code','adj_factor']],on='ts_code').merge(native,on=['ts_code','trade_date'],suffixes=('_new','_native'))
        if len(check):
            price_error=float((check.close_new/check.close_native-1).abs().max())
            adjusted_error=float((check.close_new*check.adj_factor_new/(check.close_native*check.adj_factor_native)-1).abs().max())
            if price_error>1e-9:raise ValueError('MONTHLY_NATIVE_RAW_CLOSE_MISMATCH:'+day)
            overlap.append({'date':day,'symbols':len(check),'max_close_relative_error':price_error,'max_adjusted_relative_error':adjusted_error})
            factor_rows.append(check.assign(ratio=check.adj_factor_new/check.adj_factor_native)[['ts_code','trade_date','ratio']])
        for p in [path,data/f'adj_factor-{day}.parquet']:
            results.append({'file':p.name,'rows':len(pd.read_parquet(p)),'sha256':sha256(p)})
    factors=pd.concat(factor_rows).groupby('ts_code').ratio.agg(['min','max'])
    factors['relative_scale_variation']=factors['max']/factors['min']-1
    factors.to_parquet(BASE/'monthly-factor-version-differences.parquet')
    adjustment_audit={'exact_adjustment_snapshot_match':bool(factors['relative_scale_variation'].max()<=1e-9),'nonconstant_difference_symbols':int(factors.relative_scale_variation.gt(1e-9).sum()),'maximum_relative_scale_variation':float(factors.relative_scale_variation.max()),'action':'NEW_OFFICIAL_MONTHLY_CACHE_USED_AS_REGISTERED_NO_SILENT_OVERWRITE_OF_FROZEN_MODEL_INPUTS','limitation':'Provider adjustment-factor versions differ; raw execution closes agree. This is not proof of identical historical information sets.'}
    save(BASE/'data-audit-v1.json',{'status':'COMPLETE','financial_symbols':len(contract['symbols']),'required_responses':len(contract['symbols'])*3,'files':results,'monthly_native_overlap':overlap,'adjustment_factor_versions':adjustment_audit,'original_unrevised_vintage_proven':False,'production_modified':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path,nargs='*');p.add_argument('--data',action='store_true');a=p.parse_args()
    for run in a.freeze or []:freeze(run.resolve())
    if a.data:data_audit()
