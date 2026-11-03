"""Publish verified financial observations globally without changing strategy weights or ledgers."""
from pathlib import Path
from datetime import datetime,UTC
import json
import shutil
import pandas as pd
from moneymore.data.store import ParquetStore
from moneymore.data.fundamental_research import research_financial_frame,financial_features_as_of,financial_quality
from moneymore.strategy_universe import active_strategy_universe

root=Path(__file__).resolve().parents[1];store=ParquetStore(root/'data');day='20261103'
directory=root/'state/global-financial-repair'/(day+'-'+datetime.now().strftime('%H%M%S'));directory.mkdir(parents=True)
staged=research_financial_frame(store);old=store.read('fina_indicator');keys=['ts_code','ann_date','end_date']
# Consolidate core fields per disclosure key. Conflicting observations stay null.
core=staged.drop_duplicates(keys,keep='last').set_index(keys)
for field in ['roe','q_sales_yoy']:
    v=staged.groupby(keys)[field];conflict=v.nunique()>1
    core[field]=v.first().where(~conflict)
    flag='conflict__'+field
    if flag in staged:conflict=conflict|staged.groupby(keys)[flag].any()
    core[flag]=conflict;core.loc[conflict,field]=float('nan')
existing=old.drop_duplicates(keys,keep='last').set_index(keys)
combined=core.combine_first(existing)
for field in ['roe','q_sales_yoy']:
    conflicts=core.index[core['conflict__'+field].fillna(False)]
    combined.loc[conflicts,field]=float('nan')
published=combined.reset_index();shutil.copy2(store.curated/'fina_indicator.parquet',directory/'fina_indicator-before.parquet')
store.save_snapshot('fina_indicator',published,'global_verified_financials',keys,day+'_'+datetime.now().strftime('%H%M%S'))
symbols=active_strategy_universe(store,day).symbol.tolist();features=financial_features_as_of(research_financial_frame(store),day)
features.to_parquet(directory/'per-field-provenance.parquet',index=False)
result={'published_at':datetime.now(UTC).isoformat(),'date':day,'rows_before':len(old),'rows_after':len(published),
    'quality':financial_quality(staged,symbols,day,per_field=True),'orders_or_history_modified':False,'backup':str(directory)}
(directory/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
