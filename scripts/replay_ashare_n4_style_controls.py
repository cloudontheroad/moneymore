"""Transparent sign-locked style controls; isolate learned direction from style premise."""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from threadpoolctl import threadpool_limits
import replay_ashare_n4_sources as source
from prepare_four_strategy_2026 import save
from moneymore.research.ashare_mainline import sha256
from report_ashare_mainline_replay import audit_ledger
from report_ashare_n4 import attribution

ROOT,BASE=source.ROOT,source.BASE
CONFIG=ROOT/'configs/ashare_n4_style_controls_v1.json'


def fixed_scores(panel):
    result={}
    for day,g in panel.groupby('date'):
        g=g.set_index('symbol')
        # Both horizons get the same fixed weight regardless of missingness.
        momentum=g[['mom6_1','mom12_1']].replace([np.inf,-np.inf],np.nan).rank(pct=True).fillna(.5).mean(axis=1)
        result[('factor_momentum',day)]=momentum
        # A missing five-session feature is not a fabricated gain/loss or a
        # reason to drop a held security. Use the explicitly neutral ratio 1.
        result[('factor_reversal',day)]=g.ROC5.replace([np.inf,-np.inf],np.nan).fillna(1.)
    return result


class FixedStylePlanner(source.SleevePlanner):
    def __init__(self,output,loaded):
        super().__init__(output,'evaluation',loaded)
        panel=pd.read_parquet(BASE/'panel-v2/panel.parquet',columns=['date','symbol','mom6_1','mom12_1','ROC5'])
        self.scores.update(fixed_scores(panel))
        self.n4['sources'].update(json.loads(CONFIG.read_bytes())['cases'])


def run(bps):
    output=BASE/f'n4-matrix-v2/style-controls-v2-cost{bps}'
    output.mkdir(parents=True,exist_ok=False)
    loaded=source.load_inputs('evaluation')
    planner=FixedStylePlanner(output,loaded)
    specs=json.loads(CONFIG.read_bytes())
    save(output/'contract.json',specs)
    engine=source.old.engine
    engine.load_inputs=lambda:loaded
    engine.OUT=output
    engine.KINDS=[c+'--rank_cap' for c in specs['cases']]
    engine.run(bps,execution_planner=planner,config_overrides={'initial_cash':10_000_000,'max_position_weight':.05})
    folder=output/f'bps{bps}'
    result=json.loads((folder/'results.json').read_bytes())
    h=pd.read_parquet(folder/'holdings.parquet')
    panel=pd.read_parquet(BASE/'panel-v2/panel.parquet',columns=['date','symbol','industry'])
    for row in result['cases']:
        case=row['strategy']
        e=pd.read_parquet(folder/case/'equity.parquet')
        f=pd.read_parquet(folder/case/'fills.parquet')
        row['ledger']=audit_ledger(folder/case/'paper.sqlite3',f)
        if len(e)!=181:raise ValueError('INCOMPLETE_STYLE_CONTROL')
        row['attribution']=attribution(folder,case,e,f,h.loc[h.kind.eq(case)],panel)
    save(output/'results.json',{'status':'SIGN_LOCKED_STYLE_CONTROLS_COMPLETE','cases':result['cases'],'config_sha256':sha256(CONFIG),'runner_sha256':sha256(Path(__file__)),'production_modified':False})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--bps',type=int,choices=[10,20],required=True)
    a=p.parse_args()
    with threadpool_limits(limits=2):run(a.bps)
