"""Fixed-budget dated-pool walk-forward ablation; never publishes or trades."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import yaml
from moneymore.data.store import ParquetStore
from moneymore.challenger_diagnostics import trading_dates
from moneymore.qlib_challenger import QlibPanelDataset
from moneymore.research.style_price_features import price_features
from moneymore.research.qlib_feature_preprocessing import fit_training_processor,transform_features
from moneymore.research.walkforward_protocol import purged_fold,block_interval
from moneymore.research.rank_learning_target import qlib_rank_learning_target


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--coverage-ablation',action='store_true')
    parser.add_argument('--exclude-sparse-turnover',action='store_true')
    parser.add_argument('--label-target',choices=['raw','qlib_csrank'],default='raw')
    parser.add_argument('--frozen-panel',type=Path);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_S3_EXPERIMENT')
    root=Path(__file__).resolve().parents[1];protocol_file=root/'configs/style_s3_walkforward_v1.yaml'
    protocol=yaml.safe_load(protocol_file.read_text());args.output.mkdir(parents=True)
    if args.coverage_ablation:
        protocol['variants']=['price_plus_market_factors']
        protocol['coverage_revision']={'excluded':['turnover_mean_20','missing_turnover_mean_20'],
            'reason':'original_experiment_90.835_percent_missing_not_IC_or_board_selection','model_hyperparameters_unchanged':True}
    if args.label_target=='qlib_csrank':
        protocol['learning_target']='native_qlib_CSRankNorm_label_train_valid_only'
        protocol['economic_label']='unchanged_raw_five_session_total_return'
        protocol['comparisons']=10
    protocol['exclude_sparse_turnover']=args.exclude_sparse_turnover or args.coverage_ablation
    (args.output/'protocol.yaml').write_text(yaml.safe_dump(protocol),encoding='utf-8')
    (args.output/'experiment-code.py').write_bytes(Path(__file__).read_bytes())
    status={'status':'BUILDING','pid':os.getpid(),'completed':[],'orders_enabled':False}
    def save(): (args.output/'status.json').write_text(json.dumps(status,indent=2,allow_nan=False))
    save()
    try:
        import lightgbm as lgb
        from qlib.contrib.model.linear import LinearModel
        audit=json.loads((root/'state/style-balance/s1-feature-paths-20261102-a/audit.json').read_text())
        pool_path=Path(audit['historical_pool']['path'])
        if hashlib.sha256(pool_path.read_bytes()).hexdigest()!=audit['historical_pool']['sha256']:raise ValueError('HISTORICAL_POOL_CHANGED')
        pool=pd.read_parquet(pool_path)
        if (pool.trade_date.astype(str)>=pool.signal_date.astype(str)).any() or pool.duplicated(['signal_date','ts_code']).any():raise ValueError('POOL_TIME_OR_IDENTITY_INVALID')
        store=ParquetStore(root/'data');days=[d for d in trading_dates(store,protocol['input_end']) if protocol['input_start']<=d<=protocol['input_end']]
        symbols=sorted(pool.ts_code.unique());filters=[('ts_code','in',symbols),('trade_date','>=',days[0]),('trade_date','<=',days[-1])]
        market=store.read('daily',columns=['ts_code','trade_date','open','high','low','close','vol','amount'],filters=filters)
        market=market.merge(store.read('adj_factor',columns=['ts_code','trade_date','adj_factor'],filters=filters),on=['ts_code','trade_date'],how='left',validate='one_to_one')
        basic=store.read('daily_basic',columns=['ts_code','trade_date','turnover_rate','source_trade_date'],filters=filters)
        stale=basic.source_trade_date.notna() & basic.source_trade_date.astype(str).ne(basic.trade_date.astype(str))
        basic.loc[stale,'turnover_rate']=np.nan
        market=market.merge(basic[['ts_code','trade_date','turnover_rate']],on=['ts_code','trade_date'],how='left',validate='one_to_one')
        pieces=[]
        for symbol,g in market.groupby('ts_code',sort=True):
            g=g.set_index('trade_date').reindex(days);price=g.close*g.adj_factor
            f=price_features(g,'adjusted');f['label']=price.shift(-5)/price-1
            f['turnover_mean_20']=g.turnover_rate.rolling(20,min_periods=20).mean()/100
            f['log_amount_20']=np.log((g.amount.rolling(20,min_periods=20).mean()*1000).where(lambda x:x>0))
            f['datetime']=pd.to_datetime(f.index);f['instrument']=symbol
            pieces.append(f.reset_index(drop=True).set_index(['datetime','instrument']))
        panel=pd.concat(pieces).sort_index();del pieces,market,basic
        dated=pool.copy();dated['datetime']=pd.to_datetime(dated.signal_date);dated['instrument']=dated.ts_code
        dated=dated.set_index(['datetime','instrument'])
        attrs=pd.DataFrame(index=dated.index)
        attrs['book_to_price']=1/pd.to_numeric(dated.pb,errors='coerce').where(lambda x:x>0)
        attrs['earnings_yield']=1/pd.to_numeric(dated.pe_ttm,errors='coerce').replace(0,np.nan)
        attrs['dividend_yield']=pd.to_numeric(dated.dv_ttm,errors='coerce')/100
        attrs['log_size']=np.log(pd.to_numeric(dated.total_mv,errors='coerce').where(lambda x:x>0)*10000)
        panel=panel.join(attrs,how='inner').replace([np.inf,-np.inf],np.nan)
        price_names=list(price_features(pd.DataFrame({'open':[1.],'high':[1.],'low':[1.],'close':[1.],'vol':[1.],'amount':[1.],'adj_factor':[1.]}),'adjusted').columns)
        optional=['book_to_price','earnings_yield','dividend_yield','log_size','turnover_mean_20','log_amount_20']
        if protocol['exclude_sparse_turnover']:optional.remove('turnover_mean_20')
        before=len(panel);panel=panel.dropna(subset=price_names+['label'])
        if args.frozen_panel is not None:
            frozen=pd.read_parquet(args.frozen_panel)
            pd.testing.assert_frame_equal(panel,frozen)
            panel=frozen
            status['frozen_panel_sha256']=hashlib.sha256(args.frozen_panel.read_bytes()).hexdigest()
            save()
        panel.to_parquet(args.output/'research-panel.parquet')
        (args.output/'coverage.json').write_text(json.dumps({'pool_rows':len(pool),'joined_rows':before,'usable_rows':len(panel),'optional_missing':panel[optional].isna().mean().to_dict(),
            'pool_sha256':audit['historical_pool']['sha256'],'historical_membership_only_from':str(pool.signal_date.min()),'not_full_PIT_attestation':True},indent=2))
        sessions=pd.DatetimeIndex(pd.to_datetime(days));first_train=pd.Timestamp(str(pool.signal_date.min()))
        results=[];daily_parts=[]
        for fold,(test_start,test_end) in enumerate(protocol['folds']):
            segments=purged_fold(sessions,first_train,test_start,test_end,horizon=protocol['horizon'],validation=protocol['validation_sessions'])
            dates=panel.index.get_level_values('datetime');train_mask=(dates>=pd.Timestamp(segments['train'][0]))&(dates<=pd.Timestamp(segments['train'][1]))
            for variant in protocol['variants']:
                names=price_names+(optional if variant=='price_plus_market_factors' else [])
                medians=panel.loc[train_mask,names].median();missing_reference=medians.index[medians.isna()].tolist();active=[n for n in names if n not in missing_reference]
                features=panel[active].fillna(medians[active]).copy()
                if variant!='price_only':
                    for name in optional:features['missing_'+name]=panel[name].isna().astype(float)
                features.columns=pd.MultiIndex.from_product([['feature'],features.columns]);frame=features.copy()
                frame[('label','TOTAL_RETURN')]=panel.label
                artifact=fit_training_processor(frame,*segments['train']);frame=transform_features(artifact,frame)
                if args.label_target=='qlib_csrank':
                    valid_mask=(dates>=pd.Timestamp(segments['valid'][0]))&(dates<=pd.Timestamp(segments['valid'][1]))
                    frame=qlib_rank_learning_target(frame,train_mask|valid_mask)
                models=['ridge','lightgbm']+(['shuffled_ridge'] if variant=='price_only' else [])
                for name in models:
                    current=frame.copy() if name=='shuffled_ridge' else frame
                    if name=='shuffled_ridge':
                        rng=np.random.default_rng(protocol['seed']+fold)
                        current.loc[train_mask,('label','TOTAL_RETURN')]=rng.permutation(current.loc[train_mask,('label','TOTAL_RETURN')].to_numpy())
                    dataset=QlibPanelDataset(current,segments);test=dataset.prepare('test',['feature','label'])
                    if name!='lightgbm':
                        model=LinearModel(estimator='ridge',alpha=protocol['ridge_alpha'],fit_intercept=True,include_valid=False);model.fit(dataset)
                        score=model.predict(dataset,'test')
                    else:
                        train=dataset.prepare('train',['feature','label']);valid=dataset.prepare('valid',['feature','label'])
                        model=lgb.train({'objective':'regression','verbosity':-1,'learning_rate':protocol['lightgbm_learning_rate'],
                            'num_leaves':protocol['lightgbm_leaves'],'num_threads':protocol['threads'],'seed':protocol['seed']},
                            lgb.Dataset(train['feature'],label=train['label'].iloc[:,0]),num_boost_round=protocol['lightgbm_rounds'],
                            valid_sets=[lgb.Dataset(valid['feature'],label=valid['label'].iloc[:,0])],callbacks=[lgb.early_stopping(protocol['lightgbm_early_stop'],verbose=False)])
                        score=pd.Series(model.predict(test['feature']),index=test.index)
                    prediction=pd.concat([score.rename('score'),panel.label.reindex(test.index).rename('economic_label')],axis=1)
                    ic=prediction.groupby(level='datetime').apply(lambda x:x.score.corr(x.economic_label,method='spearman'),include_groups=False)
                    key=f'fold{fold}-{variant}-{name}';prediction.to_parquet(args.output/(key+'-predictions.parquet'))
                    (args.output/(key+'-processor.pkl')).write_bytes(pickle.dumps({'normalizer':artifact,'medians':medians,'dropped_no_training_reference':missing_reference}))
                    (args.output/(key+'-model.pkl')).write_bytes(pickle.dumps(model))
                    daily_parts.append(pd.DataFrame({'date':ic.index,'fold':fold,'variant':variant,'model':name,'rank_ic':ic.to_numpy()}))
                    results.append({'fold':fold,'variant':variant,'model':name,'segments':segments,'test_rows':len(test),'test_days':len(ic),
                        'rank_ic':float(ic.mean()),'undefined_ic_days':int(ic.isna().sum()),'dropped_no_training_reference':missing_reference,
                        'best_round':model.best_iteration if name=='lightgbm' else None})
                    status.update(status='FITTING',completed=[f"{x['fold']}:{x['variant']}:{x['model']}" for x in results]);save()
                    print(json.dumps(results[-1]),flush=True)
        daily=pd.concat(daily_parts,ignore_index=True);daily.to_parquet(args.output/'daily-ic.parquet',index=False);summary=[]
        for (variant,name),g in daily.groupby(['variant','model']):
            if g['date'].duplicated().any():raise ValueError('OUTER_TEST_DAYS_OVERLAP')
            finite=g.rank_ic.dropna();ci=block_interval(finite,protocol['bootstrap_block'],protocol['bootstrap_repetitions'],protocol['bootstrap_seed'],protocol['comparisons'])
            summary.append({'variant':variant,'model':name,'rank_ic':ci,'screen_pass':ci['mean']>=.01 and ci['lower']>0,
                'not_publication_evidence':True})
        result={'status':'EXPLORATORY_WALKFORWARD_COMPLETE_NOT_RELEASED','summary':summary,'fold_results':results,'orders_enabled':False,
            'protocol_sha256':hashlib.sha256((args.output/'protocol.yaml').read_bytes()).hexdigest(),'limitations':protocol['limits'],
            'automatic_model_selection':False,'trained_models':len(results),'learning_target':args.label_target,
            'frozen_panel_sha256':status.get('frozen_panel_sha256')}
        (args.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False));status['status']=result['status'];save();print(json.dumps(summary),flush=True)
    except Exception as error:
        status.update(status='FAILED',error_type=type(error).__name__);save();raise


if __name__=='__main__':main()
