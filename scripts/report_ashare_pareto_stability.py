"""Report paired date uncertainty for completed runs, without selecting a winner."""
import argparse
import json
from pathlib import Path
import pandas as pd
from audit_ashare_pareto_inputs import OUT
from moneymore.research.paired_bootstrap import paired_block_intervals
from moneymore.research.period_statistics import period_statistics
from moneymore.research.pareto_rebuild import pareto_front
from moneymore.execution.paper import PaperBroker


def collect(folder,bps,suffix=''):
    report=json.loads((folder/'results.json').read_bytes());result=[];returns={}
    for row in report['cases']:
        name=row['strategy'];nav=pd.read_parquet(folder/name/'equity.parquet').set_index('date')
        if not nav.index.is_unique or not nav.index.is_monotonic_increasing or len(nav)!=181:raise ValueError('INCOMPLETE_ACCOUNT')
        fills=pd.read_parquet(folder/name/'fills.parquet')
        broker=PaperBroker(folder/name/'paper.sqlite3')
        if not broker.reconcile('isolated').matched or len(broker.fills('isolated'))!=len(fills):raise ValueError('FINAL_LEDGER_RECONCILIATION_FAILED')
        change=nav.equity.pct_change();change.iloc[0]=nav.equity.iloc[0]/10_000_000-1
        returns[name+suffix]=change
        independently=period_statistics(nav.equity,10_000_000)
        for key in ('net_return','sharpe','maximum_drawdown'):
            if row[key] is None:
                if independently[key] is not None:raise ValueError('METRIC_MISMATCH')
            elif abs(row[key]-independently[key])>1e-10:raise ValueError('METRIC_MISMATCH')
        turnover=float(fills.notional.sum()/2/nav.equity.mean()*252/len(nav)) if len(fills) else 0.
        result.append({**row,'strategy':name+suffix,'window_start':str(nav.index[0]),'window_end':str(nav.index[-1]),'sessions':len(nav),'initial_cash':10_000_000,'slippage_bps':bps,'execution_version':'uniform_prev_close_shares_paperbroker_v1','complete':True,'annualized_one_way_turnover':turnover})
    return result,returns


def main(root):
    evidence={}
    for bps in (10,20):
        rows,returns=collect(root/f'bps{bps}',bps)
        baseline=OUT/f'baseline-replays/bps{bps}'
        if (baseline/'results.json').exists():
            more,series=collect(baseline,bps);rows+=more;returns.update(series)
        missing_risk=[]
        for interval in (15,20,25):
            risk=OUT/f'qlib-replays/interval{interval}/full-proof-v4/bps{bps}'
            if (risk/'results.json').exists():
                more,series=collect(risk,bps,suffix=f'_{interval}');rows+=more;returns.update(series)
            else:missing_risk.append(interval)
        front=pareto_front(rows);pairs=[]
        for name in returns:
            if name in ('baseline','v2','r_old_ridge20','ridge_reference','ep_buffer'):continue
            for control in ('r_old_ridge20','baseline','v2'):
                if control not in returns:continue
                for block in (10,20,40):
                    pairs.append({'candidate':name,'reference':control,**paired_block_intervals(returns[name],returns[control],block=block,draws=1000)})
        evidence[str(bps)]={'pareto':front,'paired_intervals':pairs,'return_correlation':pd.DataFrame(returns).corr().to_dict(),'missing_baseline_comparators':[n for n in ('baseline','v2') if n not in returns],'missing_risk_intervals':missing_risk}
    (root/'stability.json').write_text(json.dumps({'scope':'RETROSPECTIVE_UNCERTAINTY_NOT_RELEASE','costs':evidence,'selection_used_2026_returns':False,'new_candidate_released':False},indent=2),encoding='utf-8')
    print(root/'stability.json',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=OUT/'pilot-replays');args=parser.parse_args();main(args.root)
