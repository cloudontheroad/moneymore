"""P1 only: native Qlib expressions on versioned local daily data, no accounts."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from moneymore.data.store import ParquetStore
from moneymore.challenger_diagnostics import trading_dates
from moneymore.research.mature_multifactor import native_bar_fields, dated_membership


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke-symbols', type=int, default=0)
    parser.add_argument('--prediction-only', action='store_true')
    parser.add_argument('--official-overlay', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('IMMUTABLE_ALPHA158_INPUTS')
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    protocol_file = root / 'configs/mature_multifactor_v1.yaml'
    protocol = yaml.safe_load(protocol_file.read_text(encoding='utf-8'))
    (args.output / 'protocol.yaml').write_bytes(protocol_file.read_bytes())
    def progress(stage, **extra):
        (args.output / 'progress.json').write_text(json.dumps({'stage':stage, **extra}), encoding='utf-8')
        print(json.dumps({'stage':stage, **extra}), flush=True)
    progress('INPUT_SNAPSHOT')
    store = ParquetStore(root / 'data')
    snapshots = store.read('strategy_universe')
    foundation = json.loads((root / 'state/style-balance/s1-feature-paths-20261102-a/audit.json').read_bytes())
    pool_path = Path(foundation['historical_pool']['path'])
    if hashlib.sha256(pool_path.read_bytes()).hexdigest() != foundation['historical_pool']['sha256']:
        raise ValueError('HISTORICAL_POOL_CHANGED')
    historical = pd.read_parquet(pool_path)
    latest_names = sorted(snapshots.loc[snapshots.effective_date.astype(str).eq(snapshots.effective_date.astype(str).max()), 'symbol'])
    symbols = sorted(set(historical.ts_code) | set(snapshots.symbol))
    if args.smoke_symbols:
        symbols = latest_names[:args.smoke_symbols]
    now = datetime.now(ZoneInfo('Asia/Shanghai'))
    # A running session is never treated as an official completed close.
    ceiling = (pd.Timestamp(now).normalize() - pd.Timedelta(days=1)).strftime('%Y%m%d') if now.hour < 18 else now.strftime('%Y%m%d')
    filters = [('ts_code','in',symbols), ('trade_date','>=',protocol['input_start']), ('trade_date','<=',ceiling)]
    bars = store.read('daily', columns=['ts_code','trade_date','open','high','low','close','vol','amount'], filters=filters)
    adj = store.read('adj_factor', columns=['ts_code','trade_date','adj_factor'], filters=filters)
    if args.official_overlay:
        overlay_manifest=json.loads((args.official_overlay/'results.json').read_bytes())
        if overlay_manifest['status']!='OFFICIAL_CLOSE_READY':raise ValueError('OVERLAY_NOT_VALIDATED')
        for name,digest in overlay_manifest['hashes'].items():
            if hashlib.sha256((args.official_overlay/name).read_bytes()).hexdigest()!=digest:raise ValueError('OVERLAY_CHANGED')
        extra_bars=pd.read_parquet(args.official_overlay/'daily.parquet')
        extra_adj=pd.read_parquet(args.official_overlay/'adj_factor.parquet')
        bars=pd.concat([bars,extra_bars.loc[extra_bars.ts_code.isin(symbols),bars.columns]],ignore_index=True).drop_duplicates(['ts_code','trade_date'],keep='last')
        adj=pd.concat([adj,extra_adj.loc[extra_adj.ts_code.isin(symbols),adj.columns]],ignore_index=True).drop_duplicates(['ts_code','trade_date'],keep='last')
    bars = bars.merge(adj,on=['ts_code','trade_date'],how='left',validate='one_to_one')
    coverage = bars.loc[bars.ts_code.isin(latest_names) & bars.close.gt(0) & bars.adj_factor.gt(0)].groupby('trade_date').ts_code.nunique()
    required = len(symbols) if args.smoke_symbols else len(latest_names)
    eligible_dates = coverage.loc[coverage >= protocol['minimum_latest_coverage'] * required]
    if eligible_dates.empty:
        raise ValueError('NO_COVERED_COMPLETED_DATE')
    cutoff = str(eligible_dates.index.max())
    bars = bars.loc[bars.trade_date.astype(str).le(cutoff)]
    calendar = [d for d in trading_dates(store, cutoff) if protocol['input_start'] <= d <= cutoff]
    membership, carried = dated_membership(calendar, historical, snapshots, protocol['train_start'])
    if args.prediction_only:
        membership=pd.MultiIndex.from_product([[pd.Timestamp(cutoff)],symbols],names=['datetime','instrument'])
    if args.smoke_symbols:
        membership = membership[membership.get_level_values('instrument').isin(symbols)]
    bars.to_parquet(args.output / 'bars-input.parquet', index=False)
    snapshots.to_parquet(args.output / 'universe-input.parquet', index=False)
    pd.DataFrame(index=membership).reset_index().to_parquet(args.output / 'membership.parquet', index=False)
    provider = args.output / 'provider'
    (provider / 'calendars').mkdir(parents=True)
    (provider / 'instruments').mkdir()
    (provider / 'features').mkdir()
    (provider / 'calendars/day.txt').write_text('\n'.join(str(pd.Timestamp(d).date()) for d in calendar) + '\n')
    mapping, spans, checks = {}, [], []
    price_panels = []
    for offset, (symbol, group) in enumerate(bars.groupby('ts_code', sort=True)):
        code = symbol.split('.')[1] + symbol[:6]
        mapping[code] = symbol
        fields, check = native_bar_fields(group)
        checks.append({'symbol':symbol, **check})
        first, last = str(fields.index.min()), str(fields.index.max())
        spans.append(f'{code}\t{pd.Timestamp(first).date()}\t{pd.Timestamp(last).date()}')
        directory = provider / 'features' / code.lower()
        directory.mkdir()
        for field in fields:
            values = fields[field].reindex(calendar).to_numpy(dtype='<f4')
            np.concatenate((np.array([0], dtype='<f4'), values)).tofile(directory / (field + '.day.bin'))
        price_panels.append(pd.DataFrame({'date':fields.index,'symbol':symbol,'adjusted_close':fields.close.to_numpy()}))
        if offset % 250 == 0:
            progress('PROVIDER_EXPORT', symbols=offset+1, total=len(symbols))
    (provider / 'instruments/all.txt').write_text('\n'.join(spans) + '\n')
    import qlib
    from qlib.contrib.data.loader import Alpha158DL
    from qlib.data import D
    qlib.init(provider_uri=str(provider.resolve()), region='cn', kernels=1, expression_cache=None, dataset_cache=None, logging_level=30)
    expressions, names = Alpha158DL.get_feature_config()
    if len(names) != 158 or len(set(names)) != 158:
        raise ValueError('NATIVE_ALPHA158_SCHEMA_CHANGED')
    progress('NATIVE_ALPHA158', symbols=len(mapping), fields=len(names), cutoff=cutoff)
    features = D.features(list(mapping), expressions, start_time=cutoff if args.prediction_only else protocol['train_start'], end_time=cutoff)
    features.columns = names
    features = features.reset_index()
    features['instrument'] = features.instrument.map(mapping)
    features = features.set_index(['datetime','instrument']).sort_index().reindex(membership)
    features = features.replace([np.inf,-np.inf],np.nan).astype('float32')
    feature_coverage = features.notna().mean(axis=1)
    features = features.loc[feature_coverage >= protocol['minimum_feature_fraction']]
    price = pd.concat(price_panels).pivot(index='date',columns='symbol',values='adjusted_close').reindex(calendar)
    label = (price.shift(-protocol['label_horizon']) / price - 1).stack(future_stack=True).rename('label')
    label.index = pd.MultiIndex.from_arrays([pd.to_datetime(label.index.get_level_values(0)),label.index.get_level_values(1)],names=['datetime','instrument'])
    features['economic_label'] = label.reindex(features.index).astype('float32')
    features.to_parquet(args.output / 'alpha158-panel.parquet')
    source = Path(inspect.getsourcefile(Alpha158DL))
    latest_rows = int(((features.index.get_level_values('datetime') == pd.Timestamp(cutoff)) & features.index.get_level_values('instrument').isin(latest_names)).sum())
    result = {'status':'P1_NATIVE_ALPHA158_READY', 'cutoff':cutoff, 'native_features':names, 'feature_rows':len(features),
        'latest_feature_rows':latest_rows,'required_latest_names':required,'latest_coverage':latest_rows / required,
        'native_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'qlib_version':qlib.__version__,
        'label_horizon':protocol['label_horizon'],'immature_labels_retained_for_inference':True,
        'membership_carried_dates':carried, 'membership_policy':'only_known_snapshots_last_known_carry_not_fresh_monthly_or_original_vintage',
        'invalid_vwap_rows':sum(c['invalid_vwap'] for c in checks),'bar_units':'hundred_shares_to_shares_thousand_CNY_to_CNY',
        'smoke_only':bool(args.smoke_symbols),'real_orders_enabled':False,
        'prediction_only':args.prediction_only,
        'input_hashes':{file.name:hashlib.sha256(file.read_bytes()).hexdigest() for file in [args.output / n for n in ['bars-input.parquet','universe-input.parquet','membership.parquet','alpha158-panel.parquet','protocol.yaml']]}}
    if result['latest_coverage'] < protocol['minimum_latest_coverage']:
        raise ValueError('LATEST_ALPHA158_COVERAGE_INSUFFICIENT')
    (args.output / 'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    progress('P1_COMPLETE', latest_feature_rows=latest_rows, cutoff=cutoff)


if __name__ == '__main__':
    main()
