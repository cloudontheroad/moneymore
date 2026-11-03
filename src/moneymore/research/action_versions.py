"""Research-only latest-known proposal selection; retain true same-time conflicts."""
import pandas as pd


def latest_known_action_versions(raw):
    if raw.empty:return raw.copy(),[]
    rows=[];audit=[]
    data=raw.loc[raw.div_proc.eq('实施')].copy()
    data['_ann']=pd.to_datetime(data.ann_date,errors='coerce')
    data['_imp']=pd.to_datetime(data.imp_ann_date,errors='coerce')
    data['_record']=pd.to_datetime(data.record_date,errors='coerce')
    data['_known']=data[['_ann','_imp']].max(axis=1)
    for key,group in data.groupby(['ts_code','end_date','record_date'],dropna=False):
        visible=group.loc[group._known.notna()&group._record.notna()&group._known.le(group._record)]
        if visible.empty:raise ValueError('NO_ACTION_VERSION_KNOWN_BY_RECORD_DATE')
        ordered=visible.sort_values(['_known','_ann','_imp'],na_position='first')
        last=ordered.iloc[-1]
        same_ann=ordered._ann.eq(last._ann)|(ordered._ann.isna()&pd.isna(last._ann))
        latest=ordered.loc[ordered._known.eq(last._known)&same_ann]
        # Compare normalized economic terms, not announcement text or NaN vs 0.
        from .replay_actions import official_replay_actions
        normalized=pd.concat([official_replay_actions(latest.iloc[[i]]) for i in range(len(latest))],ignore_index=True)
        terms=['symbol','record_date','ex_date','cash_per_share','stock_ratio','cash_pay_date','stock_list_date']
        if len(normalized[terms].drop_duplicates())>1:raise ValueError('SAME_TIME_ACTION_TERMS_CONFLICT')
        rows.append(last[raw.columns])
        if len(group)>1:audit.append({'key':list(key),'source_versions':len(group),'known_by_record':len(visible),
            'selected_ann_date':str(last.ann_date),'selected_implementation_date':str(last.imp_ann_date),
            'policy':'latest_known_announcement_not_best_return;equal_time_conflicts_block'})
    return pd.DataFrame(rows,columns=raw.columns),audit
