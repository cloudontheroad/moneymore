"""Necessary lot/industry conditions; never changes support or submits trades."""
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from moneymore.research.style_missing_controls import uncertain_controls
from moneymore.research.frozen_risk_scale import CORE
from moneymore.research.quality_contract import quality_upper_only_bounds


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--capital',type=float);args=parser.parse_args()
    if args.capital is None:
        config=Path(__file__).resolve().parents[1]/'configs/style_quality_execution_v1.yaml'
        args.capital=float(yaml.safe_load(config.read_text(encoding='utf-8'))['research_initial_capital'])
    if not np.isfinite(args.capital) or args.capital<=0:raise ValueError('INVALID_RESEARCH_CAPITAL')
    if args.output.exists():raise FileExistsError('IMMUTABLE_AUDIT')
    root=Path(__file__).resolve().parents[1];parent=root/'state/style-balance/real-archive-group-decomposition-20261102-b'
    raw=pd.read_parquet(root/'state/style-balance/preperiod-drift-202609-b/observations-20260922.parquet')
    liq=pd.read_parquet(parent/'registered-liquidity.parquet').set_index(['date','symbol']).turnover_mean_20
    raw=raw.drop(columns=['turnover_rate']).assign(turnover_mean_20=liq.loc['20260922'].reindex(raw.index));c=uncertain_controls(raw,CORE)[0]
    for field in raw.select_dtypes(include='number').columns:
        if field not in CORE:c['unknown_'+field]=raw[field].replace([np.inf,-np.inf],np.nan).isna().astype(float)
    wb=np.exp(raw.log_size);wb/=wb.sum();reference=wb @ c
    limits=pd.Series({f:.01 if f.startswith('unknown_') else .03 if f.startswith(('industry_','board_')) else .1 for f in c.columns})
    envelope=json.loads((parent/'observed-error-envelope.json').read_text())
    prices=pd.read_parquet(parent/'source-price-history.parquet');prices=prices.loc[prices.trade_date.eq('20260923')].set_index('ts_code').open
    weights=pd.read_parquet(root/'state/style-balance/quality-score-heldout-20261103-c/weights-20260922-archived_main_gru.parquet')
    rows=[]
    for variant in ['quality_unbuffered','quality_buffered']:
        margins={f:{'upper':0.,'lower':0.} for f in c.columns} if variant=='quality_unbuffered' else envelope
        lo,hi=quality_upper_only_bounds(c,wb,limits,margins);support=weights.index[weights[variant]>1e-6]
        for factor in c.columns[c.columns.str.startswith('industry_')]:
            members=c.index[c[factor].eq(1.)];held=members.intersection(support)
            floor=float(reference[factor]+lo[factor]);cap=float(reference[factor]+hi[factor])
            lots=pd.Series({s:(200 if s.startswith(('688','689')) else 100)*prices.get(s,np.nan)/args.capital for s in held})
            if floor>1e-8 and (not len(held) or (lots>cap+1e-8).all()):
                rows.append({'variant':variant,'factor':factor,'required_minimum_weight':floor,'maximum_weight':cap,
                    'target_support':held.tolist(),'minimum_lot_weights':lots.to_dict(),
                    'certificate':'required_industry_has_no_target_stock_with_affordable_lot',
                    'basis':'stock_sleeve_invested_value_cannot_exceed_capital_so_lot_weight_lower_bound_is_valid'})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'capital':args.capital,'certificates':rows,'orders_enabled':False,
        'empty_certificate_is_not_full_integer_feasibility_proof':True},indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(rows,ensure_ascii=True))


if __name__=='__main__':main()
