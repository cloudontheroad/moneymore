"""Verify whether official dated size fills resolve each frozen held-risk gap."""
import argparse
import json
from pathlib import Path
import pandas as pd
from moneymore.research.held_evidence import read_held_basic
from moneymore.research.style_price_features import named_price_risk_exposures
from moneymore.challenger_diagnostics import trading_dates
from moneymore.data.store import ParquetStore


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_HELD_AUDIT')
    root=Path(__file__).resolve().parents[1];store=ParquetStore(root/'data');basic=read_held_basic(args.evidence)
    manifest=json.loads((args.evidence/'manifest.json').read_text())
    foundation=json.loads((root/'state/style-balance/s1-feature-paths-20261102-a/audit.json').read_text())
    pool=pd.read_parquet(foundation['historical_pool']['path'])
    bars=pd.read_parquet(root/'state/style-balance/s3-cost-fold0-20261103-c/market-history.parquet')
    prices=bars.pivot(index='trade_date',columns='ts_code',values='total_price');calendar=trading_dates(store,'20240730');rows=[]
    for record in basic.to_dict('records'):
        day=str(record['trade_date']);symbol=record['ts_code'];days=[d for d in calendar if d<=day][-121:]
        attrs=pool.loc[pool.signal_date.astype(str).eq(day)].set_index('ts_code')
        ix=attrs.index.union(pd.Index([symbol]));size=attrs.total_mv.reindex(ix)*10000;size.loc[symbol]=record['total_mv']*10000
        history=prices.reindex(index=days,columns=ix);raw,_=named_price_risk_exposures(history,size)
        missing=history[symbol].index[history[symbol].isna()].astype(str).tolist()
        request=next(r for r in manifest['requests'] if r['endpoint']=='suspend_d' and r['params']['ts_code']==symbol)
        suspended=pd.read_parquet(args.evidence/request['file'])
        whole=set(suspended.loc[suspended.suspend_type.eq('S')&(suspended.suspend_timing.isna()|suspended.suspend_timing.eq('')),'trade_date'].astype(str)) if len(suspended) else set()
        rows.append({'symbol':symbol,'date':day,'dated_size_valid':True,'all_core_risk_finite':bool(raw.loc[symbol].notna().all()),
            'missing_price_dates':missing,'official_whole_day_halts':sorted(set(missing)&whole),'unexplained_price_gaps':sorted(set(missing)-whole),
            'price_or_score_imputed':False})
    args.output.write_text(json.dumps({'status':'DATA_GAP_DIAGNOSIS_NOT_EXECUTION_PASS','cases':rows},indent=2));print(json.dumps(rows))


if __name__=='__main__':main()
