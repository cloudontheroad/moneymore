"""One-shot local sequential research workflow, no scheduler or broker access."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import yaml


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--first-study',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--complete-analysis',action='store_true');args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    if args.resume:
        previous=json.loads((args.output/'progress.json').read_text());steps=previous['steps'];studies=previous['studies']
        for step in steps:
            if hashlib.sha256(Path(step['artifact']).read_bytes()).hexdigest()!=step['sha256']:raise ValueError('COMPLETED_STEP_ARTIFACT_CHANGED')
        if previous.get('status') == 'RESEARCH_REVIEW_COMPLETE_NO_ELIGIBLE_NEW_MODEL':
            reviewed = Path(previous['final_review_path'])
            if hashlib.sha256(reviewed.read_bytes()).hexdigest() != previous['final_review_sha256']:
                raise ValueError('FINAL_REVIEW_CHANGED')
            for name, digest in json.loads(reviewed.read_bytes())['input_hashes'].items():
                if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                    raise ValueError('FINAL_REVIEW_EVIDENCE_CHANGED')
            print(json.dumps({'status':previous['status'],'resume':'ALREADY_CLOSED_NO_RECOMPUTE','orders_enabled':False}))
            return
        failure=args.output/'failure.json'
        if failure.exists():
            archive=args.output/f'failure-history-{hashlib.sha256(failure.read_bytes()).hexdigest()[:12]}.json'
            if not archive.exists():failure.replace(archive)
    else:
        args.output.mkdir(parents=True,exist_ok=False);steps=[];studies=[]
    current=None
    def save(status):
        (args.output/'progress.json').write_text(json.dumps({'status':status,'current':current,'steps':steps,'studies':studies,
            'orders_enabled':False,'automatic_promotion':False},indent=2))
    def run(name,script,params,artifact):
        nonlocal current
        existing=next((s for s in steps if s['step']==name),None)
        if existing:
            if str(artifact.resolve())!=existing['artifact'] or hashlib.sha256(artifact.read_bytes()).hexdigest()!=existing['sha256']:raise ValueError('RESUME_STEP_MISMATCH')
            return
        current=name;save('RUNNING');log=args.output/f'{len(steps):02d}-{name}.log'
        with log.open('w',encoding='utf-8') as output:
            result=subprocess.run([sys.executable,str(root/'scripts'/script),*map(str,params)],cwd=root,stdout=output,stderr=subprocess.STDOUT,timeout=21600)
        if result.returncode or not artifact.exists():raise RuntimeError('LOCAL_WORKFLOW_STEP_FAILED:'+name)
        steps.append({'step':name,'artifact':str(artifact.resolve()),'sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),'log':str(log.resolve())})
        print(json.dumps({'completed':name}),flush=True);save('RUNNING')
    def audit(study):
        run(study.name+'-ledger','audit_s3_cost_ledgers.py',['--study',study,'--output',study/'independent-audit.json'],study/'independent-audit.json')
        run(study.name+'-summary','summarize_s3_execution.py',['--study',study,'--output',study/'execution-summary.json'],study/'execution-summary.json')
        summary=json.loads((study/'execution-summary.json').read_text())
        if any(e.get('reason') in ['CONTINUOUS_SOLVER_INFEASIBLE','INTEGER_FEASIBILITY_INFEASIBLE'] for e in summary['events']):
            reviewed=study/'execution-summary-reviewed.json'
            if not reviewed.exists():
                binding=json.loads((study/'episode-binding.json').read_text());held=json.loads((study/'held-evidence-binding.json').read_text())
                protocol=yaml.safe_load((study/'protocol.yaml').read_text());capture=study/'contract-capture'
                run(study.name+'-contract-capture','replay_s3_cost_episode.py',['--fold',binding['fold'],'--output',capture,'--held-evidence',held['path'],
                    '--capture-from-study',study,'--slippage-bps',protocol['slippage_bps_each_side']],capture/'results.json')
                run(study.name+'-contract-proof','verify_s3_contract_blocks.py',['--study',study,'--capture',capture,'--output',reviewed],reviewed)
            proof=json.loads(reviewed.read_text())
            if proof['source_summary_sha256']!=hashlib.sha256((study/'execution-summary.json').read_bytes()).hexdigest():raise ValueError('CONTRACT_PROOF_SOURCE_CHANGED')
            summary=proof
            if not any(s['step']==study.name+'-verified-no-trade' for s in steps):steps.append({'step':study.name+'-verified-no-trade','artifact':str(reviewed.resolve()),'sha256':hashlib.sha256(reviewed.read_bytes()).hexdigest()})
        studies[:]=[s for s in studies if s['path']!=str(study.resolve())]
        studies.append({'path':str(study.resolve()),'counts':summary['counts'],'fills':summary['fills']});save('RUNNING')
        if summary['counts'].get('solver_incomplete',0) or summary['counts'].get('unclassified_failure',0):
            raise RuntimeError('REQUIRES_SAME_INPUT_SOLVER_REPAIR:'+study.name)
    try:
        current='wait-for-first-fixed-window';save('RUNNING');deadline=time.monotonic()+21600
        while not (args.first_study/'results.json').exists():
            if time.monotonic()>deadline:raise TimeoutError('FIRST_STUDY_NOT_COMPLETE')
            time.sleep(10)
        audit(args.first_study)
        # First finish all four 10bps episodes, then the four 2bps episodes.
        for bps in [10,2]:
          for fold in range(4):
            if bps==10 and fold==0:continue
            evidence=root/f'state/style-balance/s3-cost-evidence-fold{fold}-20261103-a'
            checked=root/f'state/style-balance/s3-cost-evidence-audit-fold{fold}-20261103-a'
            if not evidence.exists():run(f'fold{fold}-official','prepare_s3_cost_evidence.py',['--fold',fold,'--output',evidence],evidence/'manifest.json')
            if not checked.exists():run(f'fold{fold}-evidence-audit','audit_s3_cost_evidence.py',['--evidence',evidence,'--output',checked],checked/'results.json')
            if json.loads((checked/'results.json').read_text())['status']!='GATES_ACTIONS_AND_QUOTES_AUDITED':raise ValueError('EPISODE_EVIDENCE_NOT_READY')
            basic=root/f'state/style-balance/s3-dated-basic-fold{fold}-20261103-a'
            if not basic.exists():run(f'fold{fold}-basic','prepare_s3_dated_basic.py',['--fold',fold,'--output',basic],basic/'manifest.json')
            if json.loads((basic/'manifest.json').read_text())['status']!='READY':raise ValueError('DATED_BASIC_NOT_READY')
            study=args.output/f'fold{fold}-{bps}bps'
            run(f'fold{fold}-{bps}bps','replay_s3_cost_episode.py',['--fold',fold,'--output',study,'--held-evidence',basic,'--integer-direct-seconds',300,'--slippage-bps',bps],study/'results.json')
            audit(study)
        current=None;save('COST_MATRIX_EXECUTED_REQUIRES_ATTRIBUTION_AND_RELEASE_REVIEW')
        if args.complete_analysis:
            frozen=args.output/'cost-matrix-complete.json'
            if not frozen.exists():frozen.write_text((args.output/'progress.json').read_text())
            elif json.loads(frozen.read_text())['studies']!=studies:raise ValueError('COST_MATRIX_COMPLETION_CHANGED')
            for fold in range(4):
                basic=root/f'state/style-balance/s3-attribution-basic-fold{fold}-20261103-a'
                reused=root/f'state/style-balance/s3-dated-basic-fold{fold}-20261103-a'
                run(f'fold{fold}-attribution-inputs','prepare_s3_dated_basic.py',['--fold',fold,'--output',basic,'--all-signal-days','--reuse',reused],basic/'manifest.json')
                if json.loads((basic/'manifest.json').read_text())['status']!='READY':raise ValueError('ATTRIBUTION_INPUTS_NOT_READY')
            attribution=root/'state/style-balance/s3-cost-attribution-20261103-a'
            run('cost-and-style-attribution','attribute_s3_cost_matrix.py',['--matrix',args.output,'--output',attribution],attribution/'results.json')
            named=root/'state/style-balance/s3-named-history-20261103-a'
            run('named-risk-history','build_s3_named_history.py',['--output',named],named/'results.json')
            current=None;save('ANALYSIS_COMPLETE_REQUIRES_RELEASE_REVIEW_AND_UI_ACCEPTANCE')
    except Exception as error:
        save('STOPPED_REQUIRES_REPAIR');(args.output/'failure.json').write_text(json.dumps({'error_type':type(error).__name__,'reason':str(error),'step':current},indent=2))
        raise


if __name__=='__main__':main()
