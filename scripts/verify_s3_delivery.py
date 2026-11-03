"""Persist actual backend/frontend verification outcomes, without starting jobs."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_DELIVERY_VERIFICATION')
    root=Path(__file__).resolve().parents[1];args.output.mkdir(parents=True);out=args.output.resolve();checks=[]
    def run(name,command,cwd=root,shell=False):
        log=out/f'{name}.log'
        with log.open('w',encoding='utf-8') as handle:
            result=subprocess.run(command,cwd=cwd,shell=shell,stdout=handle,stderr=subprocess.STDOUT,timeout=1800,
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        checks.append({'name':name,'exit_code':result.returncode,'status':'PASS' if result.returncode==0 else 'FAIL','log':str(log)})
        (out/'progress.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks[-1]),flush=True)
        if result.returncode:raise RuntimeError('DELIVERY_CHECK_FAILED:'+name)
    names=['test_integer_case','test_rebalance_outcomes','test_held_evidence','test_holding_eligibility','test_integer_trade_caps','test_s3_terminal_exit',
           'test_action_versions','test_inference_coverage','test_score_neutralization','test_rank_learning_target','test_walkforward_protocol','test_input_versions',
           'test_adjustment_audit','test_quality_contract','test_missing_group_drift','test_style_bounds','test_portfolio_error_envelope','test_frozen_risk_scale',
           'test_style_drift_audit','test_style_repair_blockers','test_style_replay_inputs','test_style_price_features','test_integer_execution','test_execution_replay',
           'test_named_factor_history','test_quote_permissions','test_fixed_order_cost','test_s3_industry_matching','test_replay_attribution','test_s3_review','test_s3_workflow_close','test_order_chunks',
           'test_baseline_v2_release','test_baseline_v2_release_stress','test_paper_release_jobs','test_release_evidence','test_paper_execution','test_paper_publication']
    junit=out/'pytest.xml'
    run('backend-tests',[sys.executable,'-m','pytest',*[f'tests/{n}.py' for n in names],'-q','-p','no:cacheprovider',
        '--basetemp',str(out/'tmp'),'--junitxml',str(junit)])
    run('api-import',[sys.executable,'-c','import moneymore.web; print("API_IMPORT_OK_NO_LIFESPAN_STARTED")'])
    dashboard=root/'apps/dashboard'
    # Verify the deployed local application, not unused Worker/D1 cloud templates
    # whose runtime declarations are supplied only by Cloudflare deployment.
    run('frontend-local-types',[shutil.which('node'),str(dashboard/'node_modules/typescript/bin/tsc'),'--noEmit','--project','tsconfig.local.json'],cwd=dashboard)
    run('frontend-build-and-tests',[shutil.which('npm'),'test'],cwd=dashboard,shell=os.name=='nt')
    suites=ET.parse(junit).getroot().findall('.//testsuite');counts={k:sum(int(s.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
    (out/'results.json').write_text(json.dumps({'status':'DELIVERY_MECHANISMS_VERIFIED_NOT_MODEL_PUBLICATION','checks':checks,'pytest':counts,
        'real_orders_enabled':False,'new_model_account_created':False},indent=2));print(json.dumps(counts))


if __name__=='__main__':main()
