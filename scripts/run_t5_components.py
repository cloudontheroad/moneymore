"""Bounded six-cell component study, at most four concurrent local workers."""
from pathlib import Path
import json,subprocess,sys,time,os
import psutil
from filelock import FileLock
from prepare_four_strategy_2026 import ROOT,save

OUT=ROOT/'state/cost-aware-ab-v2';CASES=['a_reference','b_reference','a_cost','b_cost_exec','b_exec','b_cost']

if __name__=='__main__':
    with FileLock(str(OUT/'t5-coordinator.lock'),timeout=1):
        logs=OUT/'logs';logs.mkdir(exist_ok=True);jobs={};finished={};handles=[];pending=list(CASES)
        retry_path=OUT/'t5-transient-retries.json'
        retries=json.loads(retry_path.read_bytes()) if retry_path.exists() else {}
        # Reattach independent workers after a coordinator restart; never launch duplicates.
        attached={}
        for process in psutil.process_iter(['pid','cmdline','create_time']):
            cmd=process.info['cmdline'] or []
            if any(Path(arg).name=='replay_cost_ab_v2.py' for arg in cmd) and '--strategy' in cmd and '--smoke' not in cmd:
                kind=cmd[cmd.index('--strategy')+1]
                if kind in CASES:
                    attached.setdefault(kind,[]).append(process)
        for kind,processes in attached.items():
            # Windows venv launchers have a child interpreter: monitor the interpreter.
            jobs[kind]=max(processes,key=lambda p:p.create_time())
            pending.remove(kind)
        def exit_code(process):
            if isinstance(process,subprocess.Popen):return process.poll()
            if process.is_running():return None
            return 0 if (OUT/'t5'/next(k for k,p in jobs.items() if p is process)/'results.json').exists() else 1
        while pending or jobs:
            while pending and len(jobs)<4:
                kind=pending.pop(0);result=OUT/'t5'/kind/'results.json'
                if result.exists():finished[kind]=json.loads(result.read_bytes());continue
                log=(logs/(kind+'.log')).open('a',encoding='utf-8');handles.append(log)
                jobs[kind]=subprocess.Popen([sys.executable,str(ROOT/'scripts/replay_cost_ab_v2.py'),'--strategy',kind],cwd=ROOT,
                    stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            progress={}
            for kind,process in list(jobs.items()):
                file=OUT/'t5'/kind/'progress.json'
                code=exit_code(process)
                progress[kind]={'pid':process.pid,'exit_code':code,**(json.loads(file.read_bytes()) if file.exists() else {'completed_sessions':0})}
                if code is not None:
                    if code:
                        tail=(logs/(kind+'.log')).read_text(encoding='utf-8',errors='replace')[-5000:]
                        if 'PermissionError: [WinError 5]' in tail and retries.get(kind,0)<2:
                            retries[kind]=retries.get(kind,0)+1;save(retry_path,retries)
                            del jobs[kind];pending.insert(0,kind)
                            continue
                        save(OUT/'t5-job.json',{'status':'COMPONENT_FAILED','strategy':kind,'logs':str(logs/(kind+'.log'))})
                        # Other already-started independent components continue safely.
                        raise SystemExit(1)
                    finished[kind]=json.loads((OUT/'t5'/kind/'results.json').read_bytes());del jobs[kind]
            save(OUT/'t5-job.json',{'status':'RUNNING','finished':list(finished),'pending':pending,'running':progress})
            if jobs:time.sleep(10)
        for h in handles:h.close()
        cases=[]
        for name in CASES:
            result=finished[name];assert result['cases'][0]['sessions']==181 and not result['smoke_only']
            cases.extend(result['cases'])
        save(OUT/'t5-results.json',{'status':'COMPLETE_SIX_FIXED_COMPONENTS','cases':cases,'timeline':'close_plan_next_open_execution',
            'legacy_replay_timeline_not_identical':'old summaries retained; compare only within this unified six-cell study',
            'calibration':json.loads((OUT/'t2/historical.json').read_bytes()),'real_orders_enabled':False,'production_modified':False})
        save(OUT/'t5-job.json',{'status':'COMPLETED','results':str(OUT/'t5-results.json')})
        print(json.dumps(cases),flush=True)
