"""Read-only paired and style diagnostics; no model selection or retraining."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from moneymore.research.walkforward_protocol import block_interval


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_S3_AUDIT')
    report=json.loads((args.study/'results.json').read_text());panel=pd.read_parquet(args.study/'research-panel.parquet')
    protocol=yaml.safe_load((args.study/'protocol.yaml').read_text());family=protocol['comparisons']
    daily=pd.read_parquet(args.study/'daily-ic.parquet');paired=[]
    pivot=daily.pivot(index='date',columns=['variant','model'],values='rank_ic')
    for name in ['ridge','lightgbm']:
        delta=pivot[('price_plus_market_factors',name)]-pivot[('price_only',name)]
        if not np.isfinite(delta).all():raise ValueError('PAIRED_DAYS_NOT_COMPLETE')
        paired.append({'comparison':'market_factors_minus_price_only','model':name,'interval':block_interval(delta,comparisons=family)})
    for variant in ['price_only','price_plus_market_factors']:
        delta=pivot[(variant,'ridge')]-pivot[('price_only','shuffled_ridge')]
        paired.append({'comparison':'ridge_minus_price_only_shuffled_control','variant':variant,'interval':block_interval(delta,comparisons=family)})
    styles=[];style_daily=[]
    for variant in ['price_only','price_plus_market_factors']:
        for model in ['ridge','lightgbm']:
            for file in sorted(args.study.glob(f'fold*-{variant}-{model}-predictions.parquet')):
                data=pd.read_parquet(file).join(panel[['log_size','book_to_price','dividend_yield','return_20','volatility_20']])
                for day,g in data.groupby(level='datetime'):
                    top=g.nlargest(20,'score');codes=top.index.get_level_values('instrument').astype(str)
                    all_codes=g.index.get_level_values('instrument').astype(str)
                    cutoff=top.score.min();strict=g.score.gt(cutoff);ties=g.score.eq(cutoff)
                    inclusion=strict.astype(float)+ties.astype(float)*(len(top)-int(strict.sum()))/int(ties.sum())
                    row={'date':str(day.date()),'variant':variant,'model':model,'eligible_assets':len(g),
                        'top20_star':float(codes.str.startswith('68').mean()),'top20_chinext':float(codes.str.startswith('30').mean()),
                        'pool_star':float(all_codes.str.startswith('68').mean()),'pool_chinext':float(all_codes.str.startswith('30').mean()),
                        'unique_scores':int(g.score.nunique())}
                    row.update(top20_cutoff_tie_size=int(ties.sum()),
                        top20_tie_aware_star=float(inclusion.loc[all_codes.str.startswith('68')].sum()/len(top)),
                        top20_tie_aware_chinext=float(inclusion.loc[all_codes.str.startswith('30')].sum()/len(top)))
                    for field in ['log_size','book_to_price','dividend_yield','return_20','volatility_20']:
                        observed=g[['score',field]].dropna()
                        row['score_rank_correlation_'+field]=float(observed.score.corr(observed[field],method='spearman')) if observed[field].nunique()>1 and observed.score.nunique()>1 else None
                    style_daily.append(row)
            x=pd.DataFrame([r for r in style_daily if r['variant']==variant and r['model']==model])
            styles.append({'variant':variant,'model':model,'daily_mean':x.select_dtypes('number').mean().to_dict(),
                'scope':'unweighted_top20_rank_diagnostic_not_risk_optimized_positions','board_not_technology_industry':True})
    fold_results=[{k:r[k] for k in ['fold','variant','model','rank_ic','best_round','test_rows','test_days']} for r in report['fold_results']]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    details=args.output.with_suffix('.style-daily.parquet');pd.DataFrame(style_daily).to_parquet(details,index=False)
    result={'status':'EXPLORATORY_DIAGNOSTICS_NOT_RELEASED','paired_comparisons':paired,'style_diagnostics':styles,'fold_results':fold_results,
        'audit_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'artifacts_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.study.iterdir() if p.is_file()},
        'publication_allowed':False,'models_changed':False,'orders_enabled':False,
        'limitations':['already_seen_historical_period','close_to_close_labels_not_executable_returns',
            'top20_ties_use_existing_deterministic_row_order_not_trade_selection','four_fold_variation_not_stationary_alpha_proof']}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({'paired':paired,'styles':styles}))


if __name__=='__main__':main()
