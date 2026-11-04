"""Isolated 2026 Jan-Sep study inputs; no production accounts, plans or models."""
from pathlib import Path
import argparse,json,pickle,hashlib,time
from uuid import uuid4
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'state/four-strategy-2026-jan-sep-v1'

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    for attempt in range(10):
        try:
            temporary.replace(path);return
        except PermissionError:
            if attempt==9:raise
            # Windows readers/AV can briefly hold a destination without delete sharing.
            time.sleep(.05*(attempt+1))

def train():
    from qlib.contrib.model.linear import LinearModel
    from qlib.data.dataset.processor import CSZScoreNorm
    from moneymore.qlib_challenger import QlibPanelDataset
    from moneymore.research.qlib_feature_preprocessing import fit_training_processor,transform_features
    output=OUT/'model';output.mkdir(parents=True,exist_ok=True)
    if (output/'results.json').exists():return
    source=ROOT/'state/mature-multifactor/p1-native-20261103-a'
    info=json.loads((source/'results.json').read_bytes());panel=pd.read_parquet(source/'alpha158-panel.parquet')
    calendar=pd.DatetimeIndex(pd.to_datetime((source/'provider/calendars/day.txt').read_text().splitlines()))
    pre=calendar[calendar<=pd.Timestamp('2025-12-31')];end=pre[-6]
    dates=panel.index.get_level_values('datetime');mask=(dates>=pd.Timestamp('2024-01-31'))&(dates<=end)&np.isfinite(panel.economic_label)
    names=info['native_features'];median=panel.loc[mask,names].median();assert median.notna().all()
    frame=panel[names].fillna(median);frame.columns=pd.MultiIndex.from_product([['feature'],names])
    frame[('label','TOTAL_RETURN')]=panel.economic_label
    processor=fit_training_processor(frame,'2024-01-31',str(end.date()))
    frame=transform_features(processor,frame).astype('float64')
    training=CSZScoreNorm(fields_group='label')(frame.loc[mask].copy())
    model=LinearModel(estimator='ridge',alpha=1.,fit_intercept=False,include_valid=False)
    model.fit(QlibPanelDataset(training,{'train':('20240131',end.strftime('%Y%m%d'))}))
    dataset=QlibPanelDataset(frame,{'test':('20251231','20260930')});prediction=model.predict(dataset,'test')
    restored=pickle.loads(pickle.dumps(model)).predict(dataset,'test');assert (prediction-restored).abs().max()==0
    prediction.rename('score').to_frame().to_parquet(output/'predictions.parquet')
    (output/'model.pkl').write_bytes(pickle.dumps(model));(output/'processor.pkl').write_bytes(pickle.dumps({'normalizer':processor,'medians':median}))
    result={'status':'PRE2026_MODEL_FROZEN','feature_fit_end':str(end.date()),'label_available_by':'20251231',
        'prediction_days':prediction.index.get_level_values('datetime').nunique(),'prediction_rows':len(prediction),
        'training_rows':int(mask.sum()),'latest_2026_model_reused':False,'real_orders_enabled':False,
        'membership_caveat':'shared existing historical snapshot membership; 2026 carried intervals are not original-vintage daily Top1000',
        'artifact_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}}
    save(output/'results.json',result);print(json.dumps(result),flush=True)

def data():
    from dotenv import load_dotenv
    from moneymore.data.store import ParquetStore
    from moneymore.challenger_diagnostics import trading_dates
    from moneymore.data.tushare_provider import TushareProvider
    calendar=trading_dates(ParquetStore(ROOT/'data'),'20260930')
    sessions=[d for d in calendar if '20260101'<=d<='20260930']
    days=sorted(set(sessions)|{d for d in calendar if '20251230'<=d<='20251231'})
    load_dotenv(ROOT/'.env');client=TushareProvider()._client;directory=OUT/'official';directory.mkdir(parents=True,exist_ok=True)
    journal=[]
    for day in days:
        for endpoint,param,nonempty in [('daily_basic','trade_date',True),('stk_limit','trade_date',True),('suspend_d','trade_date',False),('dividend','record_date',False)]:
            path=directory/(endpoint+'-'+day+'.parquet')
            if path.exists():journal.append({'file':path.name,'status':'CACHED'});continue
            good=False
            for attempt in range(2):
                try:
                    result=getattr(client,endpoint)(**{param:day})
                    if result is None or (nonempty and result.empty):raise ValueError('OFFICIAL_RESPONSE_EMPTY')
                    if not result.empty and not result[param].astype(str).eq(day).all():raise ValueError('DATE_MISMATCH')
                    if endpoint in ('daily_basic','stk_limit') and len(result)<1000:raise ValueError('TRUNCATED_CROSS_SECTION')
                    temp=path.with_suffix('.parquet.tmp');result.to_parquet(temp,index=False);temp.replace(path)
                    journal.append({'file':path.name,'status':'READY','rows':len(result)});good=True
                    break
                except Exception as error:
                    if attempt:journal.append({'file':path.name,'status':'FAILED','error_type':type(error).__name__})
                    else:time.sleep(3)
                finally:time.sleep(1.1)
            save(OUT/'data-progress.json',{'completed':len(journal),'total':len(days)*4,'failed':sum(r['status']=='FAILED' for r in journal),'latest_date':day,'requests':journal})
            if not good:print(json.dumps({'date':day,'endpoint':endpoint,'status':'FAILED_ERROR_BODY_SUPPRESSED'}),flush=True)
        print(json.dumps({'date':day,'completed':len(journal),'total':len(days)*4}),flush=True)
    save(OUT/'data-results.json',{'status':'READY' if not any(r['status']=='FAILED' for r in journal) else 'INCOMPLETE','sessions':sessions,'days':days,'requests':journal,'real_orders_enabled':False})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['train','data'],required=True);args=parser.parse_args()
    with threadpool_limits(limits=4):
        {'train':train,'data':data}[args.stage]()
