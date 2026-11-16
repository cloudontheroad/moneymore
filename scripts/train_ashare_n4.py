"""N4 separated feature families and purged horizons; official Qlib ensemble."""
import importlib.metadata
import inspect
import json
import os
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from moneymore.research import ashare_mainline as core
from prepare_four_strategy_2026 import save

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'state/ashare-mainline-20261112'
OUT = BASE / 'n4-training-v1'
CONFIG = ROOT / 'configs/ashare_n4_v1.json'


def features(spec):
    return spec['features'] + [p + str(w) for p in spec.get('prefixes', []) for w in spec['windows']]


def main():
    from qlib.contrib.model.double_ensemble import DEnsembleModel
    config = json.loads(CONFIG.read_bytes())
    if os.environ.get('PYTHONHASHSEED') != str(config['seed']):
        raise ValueError('REQUIRES_PYTHONHASHSEED_42')
    OUT.mkdir(parents=True, exist_ok=False)
    panel_file = BASE / 'panel-v2/panel.parquet'
    manifest = json.loads((BASE / 'panel-v2/results.json').read_bytes())
    if core.sha256(panel_file) != manifest['panel_sha256']:
        raise ValueError('PANEL_HASH_CHANGED')
    panel = pd.read_parquet(panel_file)
    core.assert_keys(panel)
    core.validate_financial_dates(panel)
    de_config = json.loads((ROOT / 'configs/ashare_mainline_v1.json').read_bytes())['doubleensemble']
    sources = [Path(__file__), CONFIG, Path(inspect.getsourcefile(core)), Path(inspect.getsourcefile(DEnsembleModel)), ROOT / 'configs/ashare_mainline_v1.json']
    save(OUT / 'runtime.json', {'panel_sha256': core.sha256(panel_file), 'source_hashes': {str(p): core.sha256(p) for p in sources}, 'versions': {p: importlib.metadata.version(p) for p in ['pyqlib','lightgbm','numpy','pandas','scikit-learn']}, 'production_enabled': False})
    for p in sources:
        dest = OUT / 'sources' / p.name
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(p.read_bytes())
    all_predictions, results = [], []
    started = time.monotonic()

    def progress(status, **extra):
        record = {'status': status, 'completed': len(results), 'total': 18, 'elapsed_seconds': time.monotonic()-started, **extra}
        save(OUT / 'progress.json', record)
        print(json.dumps(record), flush=True)

    try:
        for fold in config['folds']:
            for case, spec in config['sources'].items():
                progress('TRAINING', fold=fold, case=case)
                h = spec['horizon']
                view = panel.assign(label_20=panel[f'label_{h}'], label_end_20=panel[f'label_end_{h}'])
                split = core.purged_split(view, fold, config['validation_sessions'], config['minimum_train_sessions'])
                parts = {}
                for name, mask in split.items():
                    mask &= view.alpha_feature_fraction.ge(config['minimum_alpha_fraction'])
                    if name != 'test':
                        mask &= np.isfinite(view.label_20)
                    parts[name] = view.loc[mask].copy().reset_index(drop=True)
                if parts['train'].date.nunique() < config['minimum_train_sessions']:
                    raise ValueError('INSUFFICIENT_TRAINING')
                assert parts['train'].label_end_20.max() < parts['valid'].date.min()
                assert parts['valid'].label_end_20.max() < parts['test'].date.min()
                time_contract = {k: {'first': f.date.min(), 'last': f.date.max(), 'last_label_end': f.label_end_20.dropna().max(), 'dates': int(f.date.nunique()), 'rows': len(f)} for k,f in parts.items()}
                columns = features(spec)
                processor = core.fit_processor(parts['train'], columns, categorical=False)
                xs = {k: core.transform(processor, f) for k, f in parts.items()}
                target = {k: core.rank_target(parts[k]) for k in ['train', 'valid']}
                np.random.seed(config['seed'])
                model = DEnsembleModel(**de_config, num_threads=config['threads'], seed=config['seed'])
                model.fit(core.NativeDataset({k: core.learning_frame(xs[k], target[k]) for k in target}))
                score = core.predict(model, 'doubleensemble', xs['test'])
                if not np.isfinite(score).all():
                    raise ValueError('NONFINITE_SCORE')
                dest = OUT / fold / case
                dest.mkdir(parents=True)
                artifact = {'model': model, 'processor': processor, 'horizon': h, 'features': columns, 'time_contract': time_contract, 'production_enabled': False}
                with (dest / 'model.pkl').open('wb') as f:
                    pickle.dump(artifact, f)
                with (dest / 'model.pkl').open('rb') as f:
                    restored = pickle.load(f)
                second = core.predict(restored['model'], 'doubleensemble', core.transform(restored['processor'], parts['test']))
                error = float(np.max(np.abs(score-second)))
                if error > 1e-10:
                    raise ValueError('RELOAD_CHANGED')
                pred = parts['test'][['date','symbol','industry','label_20','label_end_20']].copy()
                pred['score'], pred['case'], pred['horizon'], pred['fold'] = score, case, h, fold
                pred.to_parquet(dest / 'predictions.parquet', index=False)
                daily = core.prediction_diagnostics(pred, pd.Period(fold, freq='Q').end_time.strftime('%Y%m%d'))
                daily.to_parquet(dest / 'diagnostics.parquet', index=False)
                model.get_feature_importance(importance_type='gain').rename('gain').rename_axis('feature').to_frame().to_parquet(dest / 'importance.parquet')
                row = {'case': case, 'fold': fold, 'horizon': h, 'mean_rank_ic': float(daily.rank_ic.mean()) if daily.rank_ic.notna().any() else None, 'mature_days': int(daily.rank_ic.notna().sum()), 'minimum_unique_scores': int(daily.distinct_scores.min()), 'trees': [m.num_trees() for m in model.ensemble], 'reload_error': error, 'time_contract': time_contract, 'features': columns, 'dropped': processor['dropped_training_constant_or_missing'], 'prediction_sha256': core.sha256(dest / 'predictions.parquet')}
                save(dest / 'results.json', row)
                results.append(row)
                all_predictions.append(pred)
                progress('CASE_COMPLETE', fold=fold, case=case, rank_ic=row['mean_rank_ic'])
        combined = pd.concat(all_predictions, ignore_index=True)
        if combined.duplicated(['case','date','symbol']).any():
            raise ValueError('DUPLICATE_PREDICTIONS')
        combined.to_parquet(OUT / 'predictions.parquet', index=False)
        save(OUT / 'results.json', {'status': 'COMPLETE', 'results': results, 'prediction_sha256': core.sha256(OUT / 'predictions.parquet'), 'config_sha256': core.sha256(CONFIG), 'production_enabled': False})
        progress('COMPLETE')
    except Exception as error:
        progress('FAILED', error_type=type(error).__name__, error=str(error))
        raise


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
