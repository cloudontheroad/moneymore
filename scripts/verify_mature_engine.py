"""Deterministic P4 wiring test in a separate root with fixture-only quotes."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd
from moneymore.config import BacktestConfig
from moneymore.data.store import ParquetStore
from moneymore.execution.paper import PaperBroker
from moneymore.mature_multifactor_runtime import prepare_mature_daily
from moneymore.portfolio_candidate import execute_portfolio_candidate_open
from moneymore.challenger_diagnostics import trading_dates


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--date',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_MATURE_ENGINE_ACCEPTANCE')
    args.output.mkdir(parents=True);root=Path(__file__).resolve().parents[1];store=ParquetStore(root/'data');broker=PaperBroker(args.output/'state/paper_orders.sqlite3')
    official=root/f'state/mature-multifactor/runtime/official/{args.date}-a';features=root/f'state/mature-multifactor/runtime/features/{args.date}-a'
    quotes=pd.read_parquet(official/'daily.parquet').set_index('ts_code').close.to_dict()
    calendar=trading_dates(store,args.date);following=calendar[calendar.index(args.date)+1];now=datetime.strptime(following+' 09:31','%Y%m%d %H:%M').replace(tzinfo=ZoneInfo('Asia/Shanghai'))
    class FixtureProvider:
        name='DETERMINISTIC_FIXTURE_NOT_REAL_MARKET'
        def intraday_quotes(self,symbols):
            return pd.DataFrame([{'symbol':s,'last':quotes[s],'up_limit':quotes[s]*1.3,'down_limit':quotes[s]*.7,
                'execution_ready':True,'timestamp':int(now.timestamp()*1000)} for s in symbols])
    results=[]
    for letter,kind in [('A','raw'),('B','balanced')]:
        candidate={'candidate_tag':args.date+letter,'account_id':'fixture_mature_'+kind,'display_name':'fixture_'+kind,
            'portfolio_variant':'mature_alpha158_'+kind,'initial_cash':10_000_000,'activation_date':args.date,
            'model_id':'mature_alpha158_lgbm_v1','strategy_id':'fixture_mature_'+kind,'model_dir':str(args.model.resolve()),
            'model_manifest':str((args.model/'results.json').resolve()),'execution_policy':{'version':'mature_10bps_v1','slippage_bps':10,'real_orders_enabled':False}}
        with patch('moneymore.mature_multifactor_runtime.runtime_inputs',return_value=(official,features)),patch('moneymore.qlib_challenger_daily._record_account_history'):
            prepared=prepare_mature_daily(root=args.output,store=store,broker=broker,config=BacktestConfig.from_yaml(root/'configs/default.yaml'),candidate=candidate,trade_date=args.date)
            first=execute_portfolio_candidate_open(root=args.output,store=store,broker=broker,config=BacktestConfig.from_yaml(root/'configs/default.yaml'),candidate=candidate,provider=FixtureProvider(),trade_date=following,now=now)
            second=execute_portfolio_candidate_open(root=args.output,store=store,broker=broker,config=BacktestConfig.from_yaml(root/'configs/default.yaml'),candidate=candidate,provider=FixtureProvider(),trade_date=following,now=now)
        if first['status']!='COMPLETED' or second['status']!='COMPLETED':raise ValueError('MATURE_FIXTURE_EXECUTION_FAILED:'+json.dumps(first))
        if len(broker.fills(candidate['account_id']))!=len(first['plan']['orders']):raise ValueError('DUPLICATE_FIXTURE_FILL')
        results.append({'kind':kind,'status':first['status'],'fills':len(broker.fills(candidate['account_id'])),
            'fresh_quote_revalidated':first['plan']['fresh_quote_revalidated'],'idempotent_retry':True,'reconciliation':first['reconciliation']})
    result={'status':'P4_ENGINE_FIXTURE_LOOP_VERIFIED','results':results,'fixture_prices_not_future_market_data':True,
        'production_accounts_or_history_modified':False,'real_orders_enabled':False}
    (args.output/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):main()
