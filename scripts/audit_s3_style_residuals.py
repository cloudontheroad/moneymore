"""Residual signal audit on fixed OOS predictions; no model refit or trading."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from moneymore.data.store import ParquetStore
from moneymore.research.baseline_v2_industry import industry_as_of
from moneymore.research.score_neutralization import neutralize_scores
from moneymore.research.walkforward_protocol import block_interval


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_RESIDUAL_AUDIT')
    root=Path(__file__).resolve().parents[1];study=root/'state/style-balance/s3-csrank-20261103-a'
    protocol_path=root/'configs/style_s3_neutralization_audit_v1.yaml';protocol=yaml.safe_load(protocol_path.read_text())
    panel_path=study/'research-panel.parquet';panel=pd.read_parquet(panel_path)
    source_report=json.loads((study/'results.json').read_text())
    if hashlib.sha256(panel_path.read_bytes()).hexdigest()!=source_report['frozen_panel_sha256']:raise ValueError('FROZEN_FEATURE_PANEL_CHANGED')
    store=ParquetStore(root/'data');membership=store.read('baseline_v2_industry_membership')
    args.output.mkdir(parents=True);(args.output/'protocol.yaml').write_bytes(protocol_path.read_bytes());rows=[];details=[];hashes={}
    source_hash=hashlib.sha256((store.curated/'baseline_v2_industry_membership.parquet').read_bytes()).hexdigest()
    for model in protocol['models']:
        for file in sorted(study.glob(f'fold*-price_only-{model}-predictions.parquet')):
            hashes[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
            predictions=pd.read_parquet(file)
            for day,g in predictions.groupby(level='datetime'):
                codes=g.index.get_level_values('instrument');controls=panel.loc[g.index,protocol['continuous_controls']].copy()
                for field in protocol['continuous_controls']:
                    missing=controls[field].isna()
                    if controls[field].notna().sum()==0:raise ValueError('NO_OBSERVED_STYLE_REFERENCE')
                    controls['unknown_'+field]=missing.astype(float)
                    controls[field]=controls[field].fillna(controls[field].median())
                industry=codes.map(industry_as_of(membership,str(day.date())).l1_code)
                known=float(pd.notna(industry).mean());categories=pd.Series(industry,index=g.index).fillna('UNKNOWN')
                controls=pd.concat([controls,pd.get_dummies(categories,prefix='industry',dtype=float)],axis=1)
                score,meta=neutralize_scores(g.score,controls)
                # This second projection is retrospective attribution only.
                # It is never used to construct the neutral prediction score.
                outcome,_=neutralize_scores(g.economic_label,controls)
                def corr(a,b):return float(a.corr(b,method='spearman')) if a.std()>1e-12 and b.std()>1e-12 else None
                row={'date':str(day.date()),'fold':int(file.name.split('-')[0][4:]),'model':model,'assets':len(g),'industry_known_fraction':known,
                    'raw_ic':corr(g.score,g.economic_label),'score_residual_vs_raw_return':corr(score,g.economic_label),
                    'score_residual_vs_return_residual':corr(score,outcome),**meta}
                for label,vector in [('raw',g.score),('residual',score)]:
                    top=vector.nlargest(20);cutoff=top.min();strict=vector.gt(cutoff);ties=vector.eq(cutoff)
                    inclusion=strict.astype(float)+ties.astype(float)*(len(top)-int(strict.sum()))/int(ties.sum())
                    row[label+'_top20_double_growth_board_fraction']=float(inclusion.loc[codes.str.startswith(('30','68'))].sum()/len(top))
                rows.append(row)
                details.append(pd.DataFrame({'raw_score':g.score,'neutral_score':score,'raw_economic_label':g.economic_label,'evaluation_only_residual_label':outcome,'model':model}))
    daily=pd.DataFrame(rows);daily.to_parquet(args.output/'daily-residual-ic.parquet',index=False)
    pd.concat(details).to_parquet(args.output/'residual-predictions.parquet')
    summary=[]
    for model,g in daily.groupby('model'):
        item={'model':model,'days':len(g),'industry_min_coverage':float(g.industry_known_fraction.min()),
            'max_orthogonality_error':float(g.maximum_normalized_orthogonality_error.max()),
            'raw_board_top20':float(g.raw_top20_double_growth_board_fraction.mean()),'residual_board_top20':float(g.residual_top20_double_growth_board_fraction.mean()),'metrics':{}}
        for metric in protocol['evaluation']['views']:
            x=g[metric].dropna()
            item['metrics'][metric]={'undefined_days':int(g[metric].isna().sum()),**block_interval(x,comparisons=protocol['evaluation']['comparisons'])}
        summary.append(item)
    result={'status':'STYLE_RESIDUAL_AUDIT_NOT_RELEASED','summary':summary,'source_prediction_hashes':hashes,'industry_source_sha256':source_hash,
        'panel_sha256':hashlib.sha256(panel_path.read_bytes()).hexdigest(),'orders_enabled':False,'models_retrained':False,
        'limitations':['OLS_proxy_controls_not_full_Barra','unknown_industry_is_explicit_not_known_sector','evaluation_label_projection_not_tradable_prediction',
            'historical_observed_period_not_fresh_forward','rank_correlation_of_OLS_residuals_not_exact_partial_Spearman','no_cost_or_return_alpha_claim']}
    (args.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(summary))


if __name__=='__main__':main()
