"""P4 register the explicitly authorized paper-observation pair after P1/P2/P3."""
import argparse
from datetime import datetime, UTC
import hashlib
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from moneymore.execution.paper import PaperBroker


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--study',type=Path,required=True);p.add_argument('--prepare',action='store_true');args=p.parse_args()
    root=Path(__file__).resolve().parents[1];path=root/'state/mature-multifactor/registry.json'
    if path.exists():raise FileExistsError('PAIR_ALREADY_REGISTERED')
    model=json.loads((args.model/'results.json').read_bytes());ledger=json.loads((args.study/'independent-audit.json').read_bytes());style=json.loads((args.study/'style-independent-audit.json').read_bytes())
    if model['status']!='P2_NATIVE_MODEL_AND_LATEST_REFIT_READY' or model['reload_max_abs_error']>1e-12:raise ValueError('MODEL_NOT_VERIFIED')
    import pandas as pd
    from moneymore.research.prediction_quality import prediction_quality
    quality=prediction_quality(pd.read_parquet(args.model/'latest-predictions.parquet')['score'])
    if not quality['passed']:raise ValueError('MODEL_TOPK_DISCRIMINATION_NOT_VERIFIED')
    if ledger['status']!='PASS_LEDGER_AUDIT_NOT_STRATEGY_ACCEPTANCE' or not all(c['planned_schedule_fully_executed'] for c in ledger['cases']):raise ValueError('FULL_TWO_ACCOUNT_EXECUTION_AUDIT_REQUIRED')
    if style['status']!='P3_ACTUAL_FILL_RISK_AND_COUNTEREXAMPLES_AUDITED':raise ValueError('INDEPENDENT_STYLE_AUDIT_REQUIRED')
    today=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d');accounts=[];broker=PaperBroker(root/'state/paper_orders.sqlite3')
    from moneymore.challenger_diagnostics import trading_dates
    from moneymore.data.store import ParquetStore
    calendar=trading_dates(ParquetStore(root/'data'),today);first_execution=calendar[calendar.index(today)+1]
    for letter,kind,name in [('A','raw','多因子原选股'),('B','balanced','多因子风格均衡')]:
        tag=today+letter;account='qlib_candidate_'+tag
        broker.initialize_account(10_000_000,account)
        if broker.account_snapshot({},account)['positions']:raise ValueError('NEW_ACCOUNT_ALREADY_HAS_HOLDINGS')
        accounts.append({'candidate_tag':tag,'account_id':account,'display_name':name,'portfolio_variant':'mature_alpha158_'+kind,
            'created_at':datetime.now(UTC).isoformat(),'activation_date':today,'first_execution_date':first_execution,'initial_cash':10_000_000,
            'strategy_id':'mature_alpha158_v1_'+kind,'model_id':model['model_id'],
            'comparison_reference_account':'qlib_candidate_'+today+'A' if kind=='balanced' else 'qlib_candidate_'+today+'B',
            'execution_policy':{'version':'mature_alpha158_paper_10bps_v1','slippage_bps':10,'real_orders_enabled':False},
            'promotion_eligible':False,'real_orders_enabled':False})
    files=[args.model/'results.json',args.model/'model.pkl',args.model/'processor.pkl',args.study/'independent-audit.json',args.study/'style-independent-audit.json',root/'configs/mature_multifactor_v1.yaml']
    files += [root/p for p in ['src/moneymore/mature_multifactor_runtime.py','src/moneymore/research/mature_multifactor.py','src/moneymore/research/enhanced_indexing.py','scripts/build_mature_alpha158.py']]
    registry={'schema':'MATURE_ALPHA158_PAIR_V1','version':'mature_alpha158_pair_v1','accounts':accounts,
        'model_dir':str(args.model.resolve()),'model_manifest':str((args.model/'results.json').resolve()),'data_cutoff':model['data_cutoff'],
        'historical_rank_ic':model['test_rank_ic'],'historical_replay':str((args.study/'results.json').resolve()),
        'evidence_hashes':{str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        'observation_authorized':True,'promotion_authorized':False,'real_orders_enabled':False,
        'admission_scope':'data_time_model_execution_style_integrity_for_independent_paper_not_residual_alpha_significance'}
    temporary=path.with_suffix('.json.tmp');temporary.write_text(json.dumps(registry,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temporary,path)
    if args.prepare:
        from moneymore.mature_multifactor_runtime import mature_candidates,prepare_mature_daily
        from moneymore.data.store import ParquetStore
        from moneymore.config import BacktestConfig
        for candidate in reversed(mature_candidates(root)):
            result=prepare_mature_daily(root=root,store=ParquetStore(root/'data'),broker=broker,config=BacktestConfig.from_yaml(root/'configs/default.yaml'),candidate=candidate,trade_date=today)
            print(json.dumps({'candidate':candidate['candidate_tag'],'status':result.status,'orders_today':len(result.orders),'next_plan':result.report_path}))
    print('MATURE_PAIR_REGISTERED_PAPER_ONLY_NO_REAL_ORDER')


if __name__=='__main__':main()
