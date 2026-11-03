"""Resolve the four frozen integer cases; no broker and no constraint changes."""
import argparse
import hashlib
import json
from pathlib import Path
from moneymore.research.integer_case import decode_case
from moneymore.research.integer_execution import plan_integer_execution


def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=60);p.add_argument('--retry-unresolved-from',type=Path);p.add_argument('--currency-objective',action='store_true');args=p.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_INTEGER_VALIDATION')
    if not 0<args.seconds<=300:raise ValueError('BOUNDED_SOLVER_BUDGET_REQUIRED')
    files=sorted(args.capture.glob('*/integer-case-*.json'))
    if len(files)!=4:raise ValueError('EXPECTED_EXACTLY_FOUR_FIXED_CASES')
    args.output.mkdir(parents=True);rows=[]
    previous={}
    if args.retry_unresolved_from:
        previous={r['case']:r for r in json.loads((args.retry_unresolved_from/'results.json').read_text())['cases']}
    for file in files:
        inputs,kwargs=decode_case(json.loads(file.read_text()))
        name=file.parent.name+'-'+file.stem
        digest=hashlib.sha256(file.read_bytes()).hexdigest();inherited=previous.get(name)
        if inherited and inherited['input_sha256']!=digest:raise ValueError('RETRY_INPUT_CHANGED')
        reused=bool(inherited and inherited['status']=='PASS_REFERENCE_QUOTES_ONLY')
        if reused:
            plan=json.loads((args.retry_unresolved_from/f'{name}.json').read_text())
        else:
            plan=plan_integer_execution(*inputs,**kwargs,time_limit=args.seconds,research_secondary_time_limit=min(10,args.seconds),
                research_objective_scale=inputs[5] if args.currency_objective else None,solver_threads=2,mip_relative_gap=.01)
        (args.output/f'{name}.json').write_text(json.dumps(plan,indent=2,allow_nan=False))
        row={'case':name,'input_sha256':digest,'status':plan['status'],'reused_certified_same_input':reused,
             'reason':plan.get('reason'),'solver_diagnostics':plan.get('solver_diagnostics'),'tracking_l1':plan.get('tracking_l1'),
             'orders':len(plan['orders'])}
        rows.append(row);(args.output/'progress.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
    result={'status':'FOUR_FIXED_CASES_PASS' if all(r['status']=='PASS_REFERENCE_QUOTES_ONLY' for r in rows) else 'UNRESOLVED',
            'cases':rows,'time_limit_each_phase':args.seconds,'mip_relative_gap':.01,'constraints_changed':False,
            'orders_enabled':False,'broker_called':False,'method':'direct_same_objective_no_zero_objective_seed'}
    (args.output/'results.json').write_text(json.dumps(result,indent=2));print(result['status'],flush=True)


if __name__=='__main__':main()
