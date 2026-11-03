"""Field-local, announcement-safe financial values shared by live scoring."""
import pandas as pd
from .fundamental_research import financial_features_as_of


def latest_global_financials(frame, latest_dates):
    rows=[]
    for cutoff,group in latest_dates.groupby('latest_date'):
        day=pd.Timestamp(cutoff).strftime('%Y%m%d');symbols=set(group.symbol)
        values=financial_features_as_of(frame.loc[frame.ts_code.isin(symbols)].copy(),day)
        if values.empty:
            values=pd.DataFrame({'ts_code':sorted(symbols),'roe':float('nan'),'q_sales_yoy':float('nan'),
                'roe_end_date':None,'q_sales_yoy_end_date':None})
        for field in ['roe','q_sales_yoy']:
            period=pd.to_datetime(values[field+'_end_date'],format='%Y%m%d',errors='coerce')
            age=(pd.Timestamp(cutoff)-period).dt.days
            stale=age.isna()|~age.between(0,200)
            values[field+'_stale']=stale
            values.loc[stale,field]=float('nan')
        rows.append(values)
    return pd.concat(rows,ignore_index=True).rename(columns={'ts_code':'symbol'}).set_index('symbol')


def _publish_global_financials(root,store,day):
    from datetime import datetime,UTC
    import json,shutil
    from .fundamental_research import research_financial_frame,financial_quality
    from ..strategy_universe import active_strategy_universe
    directory=root/'state/global-financial-repair'/(day+'-'+datetime.now().strftime('%H%M%S%f'));directory.mkdir(parents=True)
    staged=research_financial_frame(store);old=store.read('fina_indicator');keys=['ts_code','ann_date','end_date']
    core=staged.drop_duplicates(keys,keep='last').set_index(keys)
    for field in ['roe','q_sales_yoy']:
        group=staged.groupby(keys)[field];conflict=group.nunique()>1
        flag='conflict__'+field
        if flag in staged:conflict=conflict|staged.groupby(keys)[flag].any()
        core[field]=group.first().where(~conflict);core[flag]=conflict
    full=store.read('fina_indicator_research_v2').drop_duplicates(keys,keep='last').set_index(keys)
    combined=core.combine_first(full).combine_first(old.drop_duplicates(keys,keep='last').set_index(keys))
    for field in ['roe','q_sales_yoy']:
        combined.loc[core.index[core['conflict__'+field].fillna(False)],field]=float('nan')
    published=combined.reset_index();shutil.copy2(store.curated/'fina_indicator.parquet',directory/'fina_indicator-before.parquet')
    store.save_snapshot('fina_indicator',published,'global_verified_financials',keys,day+'_'+datetime.now().strftime('%H%M%S%f'))
    symbols=active_strategy_universe(store,day).symbol.tolist()
    result={'date':day,'published_at':datetime.now(UTC).isoformat(),'rows_before':len(old),'rows_after':len(published),
        'quality':financial_quality(staged,symbols,day,per_field=True),'orders_or_history_modified':False}
    (directory/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def publish_global_financials(root,store,day):
    import sqlite3
    directory=root/'state/global-financial-repair';directory.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(directory/'publication-lock.sqlite3',timeout=0) as lock:
        try:lock.execute('BEGIN IMMEDIATE')
        except sqlite3.OperationalError:return {'status':'PUBLICATION_ALREADY_RUNNING','orders_or_history_modified':False}
        return _publish_global_financials(root,store,day)
