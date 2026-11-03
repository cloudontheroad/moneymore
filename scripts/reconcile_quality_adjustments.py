"""Bounded version diagnosis and immutable official overlay, not production repair."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from moneymore.research.adjustment_audit import audit_adjustment_versions


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_VERSION_AUDIT')
    root=Path(__file__).resolve().parents[1];evidence=root/'state/style-balance/quality-repair-evidence-20261103-a'
    manifest=json.loads((evidence/'manifest.json').read_text())
    for item in manifest['requests']:
        if hashlib.sha256((evidence/item['file']).read_bytes()).hexdigest()!=item['sha256']:raise ValueError('OFFICIAL_EVIDENCE_CHANGED')
    source=root/'state/style-balance/real-archive-group-decomposition-20261102-b/source-price-history.parquet'
    old=pd.read_parquet(source);rows=[];provenance=[]
    for day in ['20260928','20260929','20260930']:
        official=pd.read_parquet(evidence/f'adj_factor-{day}.parquet')[['ts_code','trade_date','adj_factor']]
        current=old.loc[old.trade_date.eq(day),['ts_code','adj_factor']].rename(columns={'ts_code':'symbol','adj_factor':'cached'})
        current=current.merge(official.rename(columns={'ts_code':'symbol','trade_date':'date','adj_factor':'official'}),on='symbol',how='left',validate='one_to_one')
        rows.append(current)
        meta=root/f'data/raw/qmt_tushare/adj_factor/{day}.json'
        original=json.loads(meta.read_text(encoding='utf-8'))
        raw=root/f'data/raw/qmt_tushare/adj_factor/{day}.parquet'
        provenance.append({'day':day,'provider':original['provider'],'captured_at_utc':original['captured_at_utc'],
            'raw_rows':original['rows'],'raw_hash_matches_metadata':hashlib.sha256(raw.read_bytes()).hexdigest()==original['sha256']})
    audit=audit_adjustment_versions(pd.concat(rows,ignore_index=True))
    material=audit.loc[~audit.same_reported_precision]
    path_symbols=material.loc[material.ratio_path_change,'symbol'].unique().tolist()
    args.output.mkdir(parents=True);audit.to_parquet(args.output/'version-comparison.parquet',index=False)
    # Three-date official view only. Do not splice it into a long mixed-unit
    # history or call it fully corrected risk data.
    pd.concat([pd.read_parquet(evidence/f'adj_factor-{d}.parquet') for d in ['20260928','20260929','20260930']]).to_parquet(args.output/'official-three-day-overlay.parquet',index=False)
    result={'status':'VERSION_DIAGNOSIS_NOT_FULL_RISK_REPAIR','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'literal_mismatches':int((abs(audit.cached-audit.official)>1e-8).sum()),
        'precision_only_rows':int(((abs(audit.cached-audit.official)>1e-8)&audit.same_reported_precision).sum()),
        'material_rows':len(material),'constant_scale_rows':int(material.constant_scale_candidate.sum()),
        'changed_path_symbols':path_symbols,'provenance':provenance,
        'historical_method':'QmtTushareProvider.adjustment_factors estimates previous_factor*previous_close/current_pre_close',
        'limitations':['constant scaling classified over three dates only_not_full_history','rounding_not_equal_to_zero_effect_at_infinite_precision',
            'latest_official_version_not_original_vintage_proof','three_day_overlay_cannot_be_spliced_without_boundary_validation'],
        'next_step':'fetch_complete_risk_lookback_for_changed_path_symbols_then_rebuild_affected_exposures',
        'production_data_updated':False,'orders_enabled':False}
    (args.output/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()
