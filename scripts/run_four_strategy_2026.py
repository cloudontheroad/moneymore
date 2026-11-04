"""One bounded local research job: parallel replays and a final common-window table."""
from pathlib import Path
import subprocess,sys,time,json,os
from prepare_four_strategy_2026 import ROOT,OUT,save

if __name__=='__main__':
    import argparse,psutil
    parser=argparse.ArgumentParser();parser.add_argument('--resume-monitor',action='store_true');args=parser.parse_args()
    logs=OUT/'logs';logs.mkdir(parents=True,exist_ok=True);jobs={};handles=[]
    existing=json.loads((OUT/'job.json').read_bytes()).get('pids',{}) if args.resume_monitor else {}
    class Attached:
        def __init__(self,pid,kind):self.pid=pid;self.kind=kind
        def poll(self):
            try:
                process=psutil.Process(self.pid)
                if any('replay_four_strategy_2026.py' in value for value in process.cmdline()) and process.is_running():return None
            except psutil.Error:pass
            return 0 if (OUT/'parallel'/self.kind/'results.json').exists() else 1
        @property
        def returncode(self):return self.poll()
    for kind in ['baseline','v2','raw','balanced']:
        if kind in existing:
            attached=Attached(existing[kind],kind)
            if attached.poll() is None or attached.poll()==0:
                jobs[kind]=attached;continue
        log=(logs/(kind+'.log')).open('a',encoding='utf-8');handles.append(log)
        jobs[kind]=subprocess.Popen([sys.executable,str(ROOT/'scripts/replay_four_strategy_2026.py'),'--strategy',kind],cwd=ROOT,
            stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    save(OUT/'job.json',{'status':'RUNNING','pids':{k:p.pid for k,p in jobs.items()},'real_orders_enabled':False})
    while any(p.poll() is None for p in jobs.values()):
        progress={}
        for kind,process in jobs.items():
            file=OUT/'parallel'/kind/'progress.json'
            progress[kind]={'exit_code':process.poll(),**(json.loads(file.read_bytes()) if file.exists() else {'completed_sessions':0})}
        save(OUT/'job-progress.json',{'status':'RUNNING','accounts':progress});time.sleep(10)
    failures={k:p.returncode for k,p in jobs.items() if p.returncode}
    for handle in handles:handle.close()
    if failures:
        save(OUT/'job.json',{'status':'FAILED_REQUIRES_RESUME','failures':failures,'real_orders_enabled':False})
        raise SystemExit(1)
    rows=[]
    for kind in jobs:
        result=json.loads((OUT/'parallel'/kind/'results.json').read_bytes())
        assert result['window']==['20260101','20260930'] and result['cases'][0]['sessions']==181
        rows.extend(result['cases'])
    save(OUT/'results.json',{'status':'COMPLETE_FOUR_FROZEN_STRATEGIES','window':['20260101','20260930'],
        'cases':rows,'annual_sessions':252,'annual_risk_free':0,'slippage_bps_each_side':10,
        'production_accounts_modified':False,'real_orders_enabled':False,
        'caveat':'historical snapshot membership includes carried intervals; retrospective framework comparison, not original-vintage/live evidence'})
    save(OUT/'job.json',{'status':'COMPLETED','results':str(OUT/'results.json'),'real_orders_enabled':False})
    print(json.dumps(rows),flush=True)
