"""Repair label endpoints from local official prices without changing signal features.

Original evidence is immutable. Missing observations are never forward/zero filled.
"""
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from audit_ashare_pareto_inputs import ROOT, OUT
from moneymore.research.pareto_rebuild import residualize_cross_section


def main(include_downloads=False):
    dest = OUT / ('label-repair-v2' if include_downloads else 'label-repair-v1')
    dest.mkdir(exist_ok=True)
    original = OUT / 'signals/panel.parquet'
    panel = pd.read_parquet(original)
    source = ROOT / 'state/mature-multifactor/p1-native-20261103-a'
    bars = pd.read_parquet(source / 'bars-input.parquet')
    bars.trade_date = bars.trade_date.astype(str)
    calendar = sorted(bars.trade_date.unique())
    successor = dict(zip(calendar[:-1], calendar[1:]))
    panel['entry_date'] = panel.date.map(successor)
    required = pd.concat([
        panel[['symbol', 'entry_date']].rename(columns={'entry_date': 'trade_date'}),
        *[panel[['symbol', f'label_end_{h}']].rename(columns={f'label_end_{h}': 'trade_date'}) for h in (5, 20, 60)]
    ]).dropna().drop_duplicates().rename(columns={'symbol': 'ts_code'})
    keys = ['ts_code', 'trade_date']
    required_index = pd.MultiIndex.from_frame(required[keys])
    sources = []
    overlap_audits=[]
    def select(frame):
        frame = frame.copy()
        frame.trade_date = frame.trade_date.astype(str)
        return frame.loc[pd.MultiIndex.from_frame(frame[keys]).isin(required_index)]
    parts = [select(bars)]
    anchor=parts[0][keys+['open','close','adj_factor']].drop_duplicates(keys)
    def compatible(frame, name):
        if frame.empty:return frame
        overlap=frame[keys+['open','close','adj_factor']].merge(anchor,on=keys,suffixes=('_new','_base'),validate='one_to_one')
        groups={symbol:g for symbol,g in overlap.groupby('ts_code',sort=False)}
        allowed=[]
        for symbol in frame.ts_code.unique():
            g=groups.get(symbol,overlap.iloc[:0])
            ok=not g.empty
            for field in ('open','close','adj_factor'):
                good=np.isfinite(g[field+'_new'])&np.isfinite(g[field+'_base'])
                ok=ok and bool(good.any()) and bool(np.allclose(g.loc[good,field+'_new'],g.loc[good,field+'_base'],atol=1e-6,rtol=1e-6))
            overlap_audits.append({'source':name,'symbol':symbol,'overlap_rows':len(g),'accepted':ok})
            if ok:allowed.append(symbol)
        return frame.loc[frame.ts_code.isin(allowed)]
    # Local official stores first; only missing source rows are supplemented.
    daily = pd.read_parquet(ROOT / 'data/processed/daily.parquet', columns=keys + ['open', 'close'])
    daily = select(daily)
    adj = pd.read_parquet(ROOT / 'data/processed/adj_factor.parquet', columns=keys + ['adj_factor'])
    adj = select(adj)
    if daily.duplicated(keys).any() or adj.duplicated(keys).any():
        raise ValueError('LOCAL_OFFICIAL_DUPLICATE_KEYS')
    parts.append(compatible(daily.merge(adj, on=keys, how='inner', validate='one_to_one'),'processed'))
    sources.append({'source': 'processed/daily+adj_factor', 'matched_rows': len(parts[-1])})
    folders=[('state/cost-aware-ab-v2/quote-cache', '*.parquet'), ('state/four-strategy-2026-jan-sep-v1/official', 'held-exit-*.parquet')]
    if include_downloads:
        proof=json.loads((OUT/'endpoint-cache/result.json').read_bytes())
        if not proof['complete']:raise ValueError('ENDPOINT_SUPPLEMENT_NOT_COMPLETE')
        folders.append((str((OUT/'endpoint-cache').relative_to(ROOT)), '*.parquet'))
    for folder, pattern in folders:
        for path in sorted((ROOT / folder).glob(pattern)):
            frame = select(pd.read_parquet(path))
            if not frame.empty:
                parts.append(compatible(frame,str(path.relative_to(ROOT))))
                sources.append({'source': str(path.relative_to(ROOT)), 'matched_rows': len(frame)})
    prices = pd.concat(parts, ignore_index=True)
    prices = prices.loc[np.isfinite(prices.open) & prices.open.gt(0) & np.isfinite(prices.close) & prices.close.gt(0) & np.isfinite(prices.adj_factor) & prices.adj_factor.gt(0)]
    prices = prices.drop_duplicates(keys, keep='first').set_index(keys)
    entry_idx = pd.MultiIndex.from_arrays([panel.symbol, panel.entry_date])
    entry = (prices.open * prices.adj_factor).reindex(entry_idx).to_numpy()
    members = pd.read_parquet(source / 'membership.parquet')
    members['date'] = pd.to_datetime(members.datetime).dt.strftime('%Y%m%d')
    member_idx = pd.MultiIndex.from_frame(members[['instrument', 'date']])
    stats = []; unresolved = []
    for h in (5, 20, 60):
        label = f'label_{h}'; end = f'label_end_{h}'
        idx = pd.MultiIndex.from_arrays([panel.symbol, panel[end]])
        exit_price = (prices.close * prices.adj_factor).reindex(idx).to_numpy()
        computed = exit_price / entry - 1
        old = panel[label].copy()
        shared = old.notna() & np.isfinite(computed)
        if not np.allclose(old.loc[shared], computed[shared], rtol=1e-10, atol=1e-10):
            raise ValueError('ORIGINAL_LABEL_ENDPOINT_MISMATCH')
        fill = old.isna() & np.isfinite(computed) & panel[end].notna()
        panel.loc[fill, label] = computed[fill]
        for _, indices in panel.groupby('date').groups.items():
            f = panel.loc[indices]
            panel.loc[indices, f'residual_label_{h}'] = residualize_cross_section(f[label], f.industry, f.total_mv)
        missing = panel[label].isna() & panel[end].notna()
        detail = panel.loc[missing, ['symbol', 'date', 'entry_date', end]].copy().rename(columns={end: 'label_end'})
        detail['horizon'] = h
        detail['outside_pool_at_end'] = ~idx[missing].isin(member_idx)
        detail['entry_missing'] = ~np.isfinite(entry[missing])
        detail['exit_missing'] = ~np.isfinite(exit_price[missing])
        unresolved.append(detail)
        for year in ('2024', '2025', '2026'):
            scope = panel.date.str.startswith(year) & panel[end].notna()
            stats.append({'year': year, 'horizon': h, 'mature_endpoint_rows': int(scope.sum()), 'missing_before': int((scope & old.isna()).sum()), 'repaired': int((scope & fill).sum()), 'missing_after': int((scope & missing).sum())})
        print({'horizon': h, 'repaired': int(fill.sum()), 'remaining': int(missing.sum())}, flush=True)
    panel.drop(columns='entry_date').to_parquet(dest / 'panel.parquet', index=False)
    untouched=[c for c in panel.columns if not c.startswith(('label_','residual_label_')) and c!='entry_date']
    pd.testing.assert_frame_equal(panel[untouched],pd.read_parquet(original,columns=untouched))
    pd.concat(unresolved, ignore_index=True).to_parquet(dest / 'unresolved.parquet', index=False)
    pd.DataFrame(overlap_audits).to_parquet(dest/'source-overlap-audit.parquet',index=False)
    report = {'status': 'LOCAL_ENDPOINT_REPAIR_COMPLETE_UNRESOLVED_RETAINED', 'original_sha256': hashlib.sha256(original.read_bytes()).hexdigest(), 'feature_columns_unchanged': True, 'production_modified': False, 'sources': sources, 'counts': stats,'rejected_source_symbol_pairs':sum(not r['accepted'] for r in overlap_audits)}
    (dest / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--include-downloads',action='store_true');args=parser.parse_args()
    with threadpool_limits(limits=2):
        main(args.include_downloads)
