"""Prove frozen no-trade constraint conflicts, without relaxing or executing them."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import cvxpy as cp
import numpy as np
import pandas as pd
from moneymore.research.integer_case import decode_case
from moneymore.research.integer_execution import plan_integer_execution
from moneymore.research.quality_contract import quality_upper_only_bounds


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_NO_TRADE_PROOF')
    summary=json.loads((args.study/'execution-summary.json').read_text());events=summary['events'];proofs=[]
    source=json.loads((args.study/'results.json').read_text())
    for case in source['cases']:
        original=pd.read_parquet(args.study/case['account']/'equity.parquet');recreated=pd.read_parquet(args.capture/case['account']/'equity.parquet')
        if not original.equals(recreated):raise ValueError('CAPTURE_ACCOUNT_PATH_CHANGED')
    for event in events:
        reason=event.get('reason');folder=args.capture/event['account'];day=event['date']
        if reason=='CONTINUOUS_SOLVER_INFEASIBLE':
            file=folder/f'continuous-case-{day}.json';case=json.loads(file.read_text());c=pd.DataFrame(**case['controls']);wb=pd.Series(case['wb']).reindex(c.index)
            lo,hi=quality_upper_only_bounds(c,wb,pd.Series(case['limits']),case['quality_envelope'])
            held=pd.Series(case['held_values'],dtype=float);w0=(held/held.sum()).reindex(c.index,fill_value=0.)
            if held.sum()<=0:raise ValueError('CASH_ENTRY_CONFLICT_NEEDS_SEPARATE_PROOF')
            w=cp.Variable(len(c));active=c.to_numpy().T@(w-wb.to_numpy());exits=~c.index.isin(case['candidates'])
            constraints=[w>=0,w<=.15,cp.sum(w)==1,active>=lo.to_numpy(),active<=hi.to_numpy()]
            if exits.any():constraints.append(w[exits]==0)
            objective=cp.norm1(w-w0.to_numpy());problem=cp.Problem(cp.Minimize(objective),constraints)
            problem.solve(solver=cp.CLARABEL,max_iter=300,tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10)
            if problem.status!='optimal' or objective.value<=.20+1e-6:raise ValueError('CONTINUOUS_CONFLICT_NOT_PROVEN')
            proof={'minimum_required_l1':float(objective.value),'normal_l1_budget':.20,'method':'independent_minimum_turnover_LP'}
            event['classification']='verified_constraint_no_trade'
        elif reason=='INTEGER_FEASIBILITY_INFEASIBLE':
            file=folder/f'integer-case-{day}.json';case=json.loads(file.read_text());inputs,kw=decode_case(case)
            result=plan_integer_execution(*inputs,**kw,time_limit=60,solver_threads=2,mip_relative_gap=0.,diagnostic_feasibility_only=True)
            if result.get('reason')!='INTEGER_FEASIBILITY_INFEASIBLE':raise ValueError('INTEGER_INFEASIBILITY_NOT_REPRODUCED')
            proof={'method':'same_constraints_zero_objective_integer_feasibility_proof','result':result}
            event['classification']='verified_constraint_no_trade'
        else:continue
        proof.update(account=event['account'],date=day,reason=reason,input_sha256=hashlib.sha256(file.read_bytes()).hexdigest());proofs.append(proof)
    if not proofs:raise ValueError('NO_CONTRACT_CONFLICTS_TO_VERIFY')
    summary.update(status='CONTRACT_NO_TRADE_PROVEN_NOT_STRATEGY_ACCEPTANCE',counts=dict(Counter(e['classification'] for e in events)),proofs=proofs,
        source_summary_sha256=hashlib.sha256((args.study/'execution-summary.json').read_bytes()).hexdigest(),constraints_relaxed=False)
    args.output.write_text(json.dumps(summary,indent=2,allow_nan=False));print(json.dumps({'counts':summary['counts'],'proofs':proofs}))


if __name__=='__main__':main()
