"""Bounded local orchestration: two cost workers, hidden windows, no production writes."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from prepare_four_strategy_2026 import save

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'state/ashare-mainline-20261112'


def main(stage, output):
    output.mkdir(parents=True,exist_ok=True)
    workers=[]
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONHASHSEED='42')
    try:
        for bps in [10,20]:
            if stage=='allocation':
                executable=BASE/'n4-riskfolio-runtime/Scripts/python.exe'
                command=[str(executable),str(ROOT/'scripts/allocate_ashare_n4_hrp.py'),'--runs',str(output),'--bps',str(bps),'--output',str(output/f'allocation-cost{bps}.json')]
            elif stage=='blends':
                command=[sys.executable,str(ROOT/'scripts/replay_ashare_n4_blends.py'),'--runs',str(output),'--bps',str(bps),'--output',str(output/f'blends-cost{bps}')]
            else:
                command=[sys.executable,str(ROOT/'scripts/replay_ashare_n4_sources.py'),'--period',stage,'--bps',str(bps),'--output',str(output/f'{stage}-cost{bps}')]
            stream=(output/f'{stage}-cost{bps}.log').open('x',encoding='utf-8')
            child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env,cwd=ROOT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            workers.append((bps,child,stream))
        while any(child.poll() is None for _,child,_ in workers):
            save(output/f'{stage}-progress.json',{'status':'RUNNING','workers':[{'bps':bps,'pid':p.pid,'returncode':p.poll()} for bps,p,_ in workers]})
            time.sleep(5)
        result={'status':'COMPLETE' if all(p.returncode==0 for _,p,_ in workers) else 'FAILED','workers':[{'bps':bps,'pid':p.pid,'returncode':p.returncode} for bps,p,_ in workers]}
        save(output/f'{stage}-progress.json',result)
        print(json.dumps(result),flush=True)
        if result['status']!='COMPLETE': raise RuntimeError('N4_STAGE_FAILED_CHECK_LOGS')
    finally:
        for _,p,stream in workers:
            if p.poll() is None: p.terminate()
            stream.close()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['history','evaluation','allocation','blends'],required=True)
    p.add_argument('--output',type=Path,default=BASE/'n4-matrix-v1')
    a=p.parse_args()
    main(a.stage,a.output)
