"""Same-day before/after financial-only comparison; prepare forward orders, never fills."""
from pathlib import Path
import hashlib,json
from unittest.mock import patch
import pandas as pd,yaml
from moneymore.data.store import ParquetStore
from moneymore.research.sector_model import build_global_factor_snapshot
from moneymore.strategy_universe import active_strategy_universe
from moneymore.execution.paper import PaperBroker
from moneymore.config import BacktestConfig
from moneymore.multi_sector_daily import run_multi_sector_daily

root=Path(__file__).resolve().parents[1];store=ParquetStore(root/'data');day='20261103'
out=root/'state/global-financial-repair/20261103-comparison';out.mkdir(exist_ok=False)
old_path=root/'state/global-financial-repair/20261103-215148/fina_indicator-before.parquet'
policy=yaml.safe_load((root/'configs/sector_models.yaml').read_text(encoding='utf-8'))['global_selection']
symbols=active_strategy_universe(store,day).symbol.tolist()
class OldStore:
    def read(self,table,**kwargs):
        return pd.read_parquet(old_path) if table=='fina_indicator' else store.read(table,**kwargs)
def whole_row(frame,dates):
    data=frame.rename(columns={'ts_code':'symbol'}).copy()
    data['available_date']=pd.to_datetime(data.ann_date)+pd.offsets.Day(1)
    data=data.merge(dates,on='symbol');data=data.loc[data.available_date<=data.latest_date]
    return data.sort_values(['symbol','available_date','end_date']).groupby('symbol').tail(1).set_index('symbol')
capture_before={};capture_after={}
with patch('moneymore.data.global_financials.latest_global_financials',whole_row):
    before=build_global_factor_snapshot(OldStore(),symbols,policy['factors'],audit_capture=capture_before)
after=build_global_factor_snapshot(store,symbols,policy['factors'],audit_capture=capture_after)
for label,frame,capture in [('before',before,capture_before),('after',after,capture_after)]:
    frame['raw_rank']=frame.score.rank(ascending=False,method='min');frame.to_parquet(out/(label+'.parquet'),index=False)
    capture['raw_factors'].to_parquet(out/(label+'-raw.parquet'),index=False)
def summary(frame):
    return {'roe_covered':int(frame.roe.notna().sum()),'growth_covered':int(frame.quarterly_revenue_growth.notna().sum()),
        'star_best_rank':float(frame.loc[frame.symbol.str.startswith('68'),'raw_rank'].min()),
        'raw_top10':frame.sort_values('score',ascending=False).head(10)[['symbol','score']].to_dict('records')}
result={'date':day,'weights_unchanged':True,'same_market_inputs':True,'before':summary(before),'after':summary(after)}
broker=PaperBroker(root/'state/paper_orders.sqlite3');fills_before=broker.fills('multi_sector_shadow')
histories={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in (store.curated).glob('*account_daily.parquet')}
snapshot_before=broker.account_snapshot({},'multi_sector_shadow')
planned=run_multi_sector_daily(store=store,broker=broker,config=BacktestConfig.from_yaml(root/'configs/default.yaml'),trade_date=day,
    signal_dir=root/'state/multi-sector-signals',report_dir=root/'state/multi-sector-shadow',planning_only=True)
if broker.fills('multi_sector_shadow')!=fills_before or broker.account_snapshot({},'multi_sector_shadow')!=snapshot_before:raise ValueError('PLANNING_MUTATED_POSITIONS_OR_CASH')
if any(hashlib.sha256(p.read_bytes()).hexdigest()!=digest for p,digest in histories.items()):raise ValueError('PLANNING_REWROTE_HISTORY')
result.update(plan_status=planned.status,report_path=planned.report_path,holdings_cash_fills_and_histories_unchanged=True)
(out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
