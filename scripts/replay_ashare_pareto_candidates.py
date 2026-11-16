"""R4 pilot: signal changes under one existing, independently reconciled paper engine."""
import argparse,json
import pandas as pd
from audit_ashare_pareto_inputs import ROOT,OUT
import research_ashare_paper_routes as engine


def main(bps,repaired=False):
    panel=pd.read_parquet(OUT/('label-repair-v2/panel.parquet' if repaired else 'signals/panel.parquet'))
    forecasts=pd.read_parquet(OUT/('walkforward-repaired/predictions.parquet' if repaired else 'walkforward/predictions.parquet'))
    features=['value_quality_comparable','improvement_comparable','momentum_comparable']
    overrides={}
    for name,column in [('r_value','value_quality_comparable'),('r_improvement','improvement_comparable'),('r_momentum','momentum_comparable')]:
        overrides[name]={day:g.set_index('symbol')[column] for day,g in panel.groupby('date')}
    # Equal signal weights require all three normalized inputs; no per-stock formula renormalization.
    panel['equal_signal']=panel[features].mean(axis=1,skipna=False)
    overrides['r_equal']={day:g.set_index('symbol').equal_signal for day,g in panel.groupby('date')}
    for name,g in forecasts.groupby('strategy'):
        overrides['r_'+name]={day:part.set_index('symbol').score for day,part in g.groupby('date')}
    old=pd.read_parquet(ROOT/'state/four-strategy-2026-jan-sep-v1/model/predictions.parquet').score
    overrides['r_old_ridge20']={d.strftime('%Y%m%d'):g.droplevel('datetime') for d,g in old.groupby(level='datetime')}
    engine.OUT=OUT/('repaired-replays' if repaired else 'pilot-replays')
    engine.KINDS=['ridge_reference','ep_buffer',*overrides]
    intervals={k:20 for k in overrides}
    path=engine.OUT/f'bps{bps}';path.mkdir(parents=True,exist_ok=True)
    (path/'research-contract.json').write_text(json.dumps({'scope':'R4_SIGNAL_PILOT_NOT_FULL_BASELINE_V2_OR_QLIB_RISK_COMPARISON','rules':{'new_groups_rebalance_every_sessions':20,'max_ordinary_replacements':5,'topk':50,'mandatory_pool_exits':'daily','old_daily_ridge_and_ep_buffer':'unchanged daily cadence'},'intervals':intervals,'not_shares_of_backtest_nav':True,'real_orders_enabled':False},indent=2),encoding='utf-8')
    engine.run(bps,score_overrides=overrides,selection_intervals=intervals)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bps',type=int,choices=[10,20],default=10);parser.add_argument('--repaired',action='store_true');args=parser.parse_args();main(args.bps,args.repaired)
