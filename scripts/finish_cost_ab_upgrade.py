"""One-off local completion chain. Not a Codex automation or a real trader."""
from datetime import datetime,UTC
from pathlib import Path
from zoneinfo import ZoneInfo
import json,subprocess,sys,time,os,hashlib,urllib.request
from filelock import FileLock
from prepare_four_strategy_2026 import ROOT,save
OUT=ROOT/'state/cost-aware-ab-v2'

def run(script,*args):
    log=OUT/'logs'/('completion-'+script+'.log')
    with log.open('a',encoding='utf-8') as handle:
        result=subprocess.run([sys.executable,str(ROOT/'scripts'/script),*args],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,
            timeout=3600,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    if result.returncode:raise ValueError('COMPLETION_STEP_FAILED:'+script)

def finish(day):
    deadline=time.monotonic()+6*3600
    while not (OUT/'t5-results.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('FIXED_STUDY_NOT_COMPLETE_WITHIN_SIX_HOURS')
        job=json.loads((OUT/'t5-job.json').read_bytes())
        if job['status']=='COMPONENT_FAILED':raise ValueError('COMPONENT_FAILED:'+job.get('strategy','unknown'))
        save(OUT/'completion-status.json',{'status':'WAITING_FIXED_STUDY','day':day,'updated_at':datetime.now(UTC).isoformat(),
            'finished':job.get('finished',[]),'running':{k:v.get('completed_sessions') for k,v in job.get('running',{}).items()},'real_orders_enabled':False})
        time.sleep(15)
    run('audit_t5_components.py')
    # The existing official-close contract does not permit same-day plans before 18:00.
    while True:
        local=datetime.now(ZoneInfo('Asia/Shanghai'))
        if local.strftime('%Y%m%d')!=day:raise ValueError('RELEASE_DAY_CHANGED_DO_NOT_BACKDATE_BRANCH')
        if (local.hour,local.minute)>=(18,35):break
        save(OUT/'completion-status.json',{'status':'WAITING_FORMAL_CLOSE_WINDOW','day':day,'real_orders_enabled':False})
        time.sleep(15)
    run('release_cost_ab.py','--date',day,'--publish')
    registry=json.loads((OUT/'registry.json').read_bytes());expected={r['candidate_tag'] for r in registry['accounts']};public=None
    for attempt in range(3):
        with urllib.request.urlopen('http://127.0.0.1:8788/api/qlib-challenger',timeout=60) as response:payload=json.load(response)
        public=[r for r in payload['candidate_observation']['candidates'] if r['candidate_tag'] in expected]
        if len(public)==2:break
        time.sleep(15)
    assert len(public)==2,'PUBLISHED_PAIR_NOT_VISIBLE_IN_API'
    assert all(r['research_gate_passed'] and r['reconciliation']['matched'] and r['latest']['signal_date']==day and r.get('order_preview') for r in public)
    with urllib.request.urlopen('http://127.0.0.1:3000',timeout=30) as response:assert response.status==200
    save(OUT/'release-api-acceptance.json',{'status':'API_AND_DASHBOARD_ACCEPTED','accounts':[{'tag':r['candidate_tag'],'account_id':r['account_id'],
        'signal_date':r['latest']['signal_date'],'planned_trade_date':r['latest']['planned_trade_date'],'preview_status':r['order_preview']['status'],
        'reconciled':r['reconciliation']['matched']} for r in public],'interactive_browser_acceptance':'NOT_AUTOMATED_BY_THIS_SCRIPT','real_orders_enabled':False})
    plan=ROOT/'docs/PLAN.md';lines=plan.read_text(encoding='utf-8').splitlines();updated=[]
    statuses={'T3':'完成：成本增量/精确费用/同NAV单位/保留实际股数与强制退出，测试及六组终验通过',
        'T5':'完成：六组各181日，独立逐日/逐股会计桥接通过；最终报告 research/COST_AWARE_AB_FINAL_20261104.md',
        'T6':f"已发布独立模拟分支 {', '.join(sorted(expected))}；同父账户现金/持仓/权益起点，API/页面可达；不代表已发生下一交易日成交"}
    for line in lines:
        for task,status in statuses.items():
            if line.startswith('| '+task+' |'):
                parts=line.split('|');parts[-2]=' '+status+' ';line='|'.join(parts)
        updated.append(line)
    plan.write_text('\n'.join(updated)+'\n',encoding='utf-8')
    save(OUT/'completion-status.json',{'status':'PUBLISHED_AND_API_ACCEPTED','day':day,'updated_at':datetime.now(UTC).isoformat(),
        'registry':str(OUT/'registry.json'),'interactive_browser_acceptance_pending':True,'real_orders_enabled':False})

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--date',required=True);args=parser.parse_args()
    with FileLock(str(OUT/'completion.lock'),timeout=1):
        try:finish(args.date)
        except Exception as error:
            save(OUT/'completion-status.json',{'status':'FAILED_NOT_PUBLISHED_OR_NOT_ACCEPTED','day':args.date,'error':type(error).__name__+':'+str(error),
                'updated_at':datetime.now(UTC).isoformat(),'real_orders_enabled':False})
            raise
