"""Publish explicit executed/no-trade/solver-failure coverage, never select a winner."""
import argparse
from collections import Counter
import json
from pathlib import Path
from moneymore.research.rebalance_outcomes import classify_rebalance


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_EXECUTION_SUMMARY')
    results=json.loads((args.study/'results.json').read_text());rows=[]
    for case in results['cases']:
        events=[e for e in case['events'] if e['status']!='TERMINAL_EXIT_ATTEMPT']
        if len(events)!=case['planned_rebalances'] or len({e['date'] for e in events})!=len(events):raise ValueError('SCHEDULE_ACCOUNTING_MISMATCH')
        rows.extend(dict(account=case['account'],classification=classify_rebalance(e),**e) for e in events)
    out={'status':'EXECUTION_ACCOUNTING_NOT_STRATEGY_ACCEPTANCE','counts':dict(Counter(r['classification'] for r in rows)),
         'planned_rebalances':len(rows),'fills':sum(c['fills'] for c in results['cases']),'events':rows,
         'winner_selection_allowed':False,'orders_enabled':False,
         'note':'Known no-trade outcomes are recorded, not fabricated fills; solver failures remain unresolved. This does not waive independent alpha or risk gates.'}
    args.output.write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='events'}))


if __name__=='__main__':main()
