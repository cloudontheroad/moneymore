"""One-shot bounded completion of the already registered experiment, not a service."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from prepare_four_strategy_2026 import save

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'state/ashare-simple-rules-20261113'


def command(args,log):
    with log.open('w',encoding='utf-8') as f:
        subprocess.run([sys.executable,'-W','ignore::FutureWarning',*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)


def main():
    started=time.monotonic();last=None;changed=started
    status={'stage':'WAITING_BOUNDED_DATA','pid':os.getpid()}
    save(BASE/'completion-job.json',status)
    try:
        while True:
            p=BASE/'data-v1/financial-progress.json'
            try:r=json.loads(p.read_bytes())
            except json.JSONDecodeError:time.sleep(2);continue
            if r['status']=='COMPLETE':break
            if r['completed_requests']!=last:last=r['completed_requests'];changed=time.monotonic()
            if time.monotonic()-changed>180 or time.monotonic()-started>3600:raise RuntimeError('DATA_PROGRESS_STALLED_OR_TIMEOUT')
            print(json.dumps({'stage':status['stage'],'data_complete':last,'data_total':r['total_requests']}),flush=True)
            time.sleep(45)
        status['stage']='AUDIT_DATA';save(BASE/'completion-job.json',status)
        command(['scripts/audit_ashare_simple_rules_inputs.py','--data'],BASE/'data-audit.log')
        jobs=[];logs=[]
        status['stage']='FCF_REPLAYS';save(BASE/'completion-job.json',status)
        for bps in [10,20]:
            output=BASE/f'fcf-cost{bps}-v1'
            if (output/'completed.json').exists():continue
            if output.exists():raise RuntimeError('PARTIAL_FCF_OUTPUT_REQUIRES_INSPECTION')
            log=(BASE/f'fcf-cost{bps}.log').open('w',encoding='utf-8');logs.append(log)
            jobs.append(subprocess.Popen([sys.executable,'-W','ignore::FutureWarning','scripts/replay_ashare_simple_rules.py','--output',str(output),'--bps',str(bps),'--group','fcf'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0))
        for job in jobs:
            if job.wait()!=0:raise RuntimeError('FCF_REPLAY_FAILED')
        for log in logs:log.close()
        status['stage']='FINAL_AUDIT';save(BASE/'completion-job.json',status)
        command(['scripts/report_ashare_simple_rules.py'],BASE/'final-report.log')
        status.update(stage='COMPLETE',production_modified=False)
        print(json.dumps(status),flush=True)
    except BaseException as error:
        status.update(stage='FAILED',error_type=type(error).__name__,error=str(error))
        raise
    finally:save(BASE/'completion-job.json',status)


if __name__=='__main__':main()
