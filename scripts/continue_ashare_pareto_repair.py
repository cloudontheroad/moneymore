"""Isolated, fail-stop continuation; never promotes candidates or touches live NAV."""
import json
import subprocess
import sys
import time
from audit_ashare_pareto_inputs import ROOT, OUT


def main():
    state=OUT/'continuation-v1';state.mkdir(exist_ok=True)
    def status(phase,**kw):
        (state/'progress.json').write_text(json.dumps({'phase':phase,'production_modified':False,**kw},indent=2),encoding='utf-8')
        print(phase,kw,flush=True)
    def run(name,args,proof):
        if proof.exists():
            status(name+'_EXISTING_EVIDENCE');return
        status(name+'_RUNNING')
        with (state/(name+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode or not proof.exists():
            status(name+'_FAILED',exit_code=result.returncode);raise RuntimeError(name)
        status(name+'_COMPLETE')
    status('WAITING_FOR_BOUNDED_ENDPOINT_DOWNLOAD')
    deadline=time.monotonic()+3600
    while not (OUT/'endpoint-cache/result.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('ENDPOINT_DOWNLOAD_NOT_FINISHED')
        time.sleep(5)
    if not json.loads((OUT/'endpoint-cache/result.json').read_bytes())['complete']:raise ValueError('ENDPOINT_FETCH_INCOMPLETE')
    run('repair',['scripts/repair_ashare_pareto_labels.py','--include-downloads'],OUT/'label-repair-v2/results.json')
    panel=str(OUT/'label-repair-v2/panel.parquet')
    run('validation2025',['scripts/train_ashare_pareto_walkforward.py','--panel',panel,'--output',str(OUT/'validation2025'),'--year','2025'],OUT/'validation2025/results.json')
    run('blendvalidation',['scripts/select_ashare_pareto_blend.py'],OUT/'blend-selection/results.json')
    run('refit2026',['scripts/train_ashare_pareto_walkforward.py','--panel',panel,'--output',str(OUT/'walkforward-repaired')],OUT/'walkforward-repaired/results.json')
    for bps in (10,20):
        run('replay'+str(bps),['scripts/replay_ashare_pareto_candidates.py','--repaired','--bps',str(bps)],OUT/f'repaired-replays/bps{bps}/results.json')
    # Wait for the separately running first baseline replay; no duplicate writer.
    status('WAITING_FOR_BASELINE_10BPS')
    deadline=time.monotonic()+7200
    while not (OUT/'baseline-replays/bps10/results.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('BASELINE10_NOT_FINISHED')
        time.sleep(5)
    run('baseline20',['scripts/replay_ashare_pareto_baselines.py','--bps','20'],OUT/'baseline-replays/bps20/results.json')
    run('uncertainty',['scripts/report_ashare_pareto_stability.py','--root',str(OUT/'repaired-replays')],OUT/'repaired-replays/stability.json')
    status('R4_REPLAY_COMPLETE_R5_QLIB_AND_R6_PENDING')


if __name__=='__main__':
    main()
