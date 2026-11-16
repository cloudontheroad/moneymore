"""Independent N4 sleeve books with frozen N3 rank allocator and actual broker."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import replay_ashare_rank_preserving as old
from moneymore.research.ashare_mainline import sha256
from prepare_four_strategy_2026 import save

ROOT = old.ROOT
BASE = ROOT / 'state/ashare-mainline-20261112'
CONFIG = ROOT / 'configs/ashare_n4_v1.json'
CORE = 'doubleensemble_full'
SOURCES = [CORE, 'fundamental', 'trend', 'short']
SCOPE = ROOT / 'configs/ashare_n4_execution_scope_v1.json'


def load_inputs(period):
    loaded = list(old.engine.load_inputs())
    config = json.loads(CONFIG.read_bytes())
    if period == 'evaluation':
        return tuple(loaded)
    dates = pd.DatetimeIndex(pd.to_datetime((ROOT / 'state/mature-multifactor/p1-native-20261103-a/provider/calendars/day.txt').read_text().splitlines())).strftime('%Y%m%d').tolist()
    start, end = config['history_window']
    sessions = [d for d in dates if start <= d <= end]
    signals = {d: dates[dates.index(d)-1] for d in sessions}
    evidence = json.loads((BASE / 'n4-data-v1/results.json').read_bytes())
    hashes = {r['file']: r['sha256'] for r in evidence['files']}
    official = {}
    for day in sorted(set(sessions) | set(signals.values())):
        official[day] = {}
        for name in ['daily_basic','stk_limit','suspend_d','dividend']:
            p = BASE / 'n4-data-v1/official' / f'{name}-{day}.parquet'
            if sha256(p) != hashes[p.name]:
                raise ValueError('OFFICIAL_CHANGED')
            official[day][name] = pd.read_parquet(p)
    # Engine legacy selector is ignored by the physical planner, but requires aligned keys.
    pred = pd.read_parquet(BASE / 'n2-summary-v1/predictions.parquet')
    pred = pred.loc[pred.case.eq(CORE)]
    dummy = {d: g.set_index('symbol').score for d,g in pred.groupby('date')}
    loaded[2:8] = [sessions, signals, {s: 1.0 for s in signals.values()}, 1.0, dummy, official]
    return tuple(loaded)


class SleevePlanner(old.RankPlanner):
    def __init__(self, output, period, loaded):
        self.n4 = json.loads(CONFIG.read_bytes())
        contract = json.loads(old.CONFIG.read_bytes())
        contract['window'] = self.n4[period + '_window']
        super().__init__(output, contract, loaded)
        training = BASE / 'n4-training-v1'
        runtime = json.loads((training / 'runtime.json').read_bytes())
        if runtime['source_hashes'][str(CONFIG)] != sha256(CONFIG):
            raise ValueError('TRAINING_CONTRACT_CHANGED')
        folds = sorted({str(pd.Period(pd.Timestamp(d),freq='Q')) for d in loaded[3].values()})
        parts, hashes = [], {}
        for fold in folds:
            for source in self.n4['sources']:
                folder = training / fold / source
                manifest = json.loads((folder / 'results.json').read_bytes())
                path = folder / 'predictions.parquet'
                if sha256(path) != manifest['prediction_sha256']:
                    raise ValueError('FOLD_PREDICTIONS_CHANGED')
                parts.append(pd.read_parquet(path))
                hashes[str(path)] = sha256(path)
        predictions = pd.concat(parts,ignore_index=True)
        self.scores.update({(case, d): g.set_index('symbol').score for (case,d),g in predictions.groupby(['case','date'])})
        self.source_hash = hashes

    def risk(self, signal):
        covariance, attrs, audit = super().risk(signal)
        events = json.loads(SCOPE.read_bytes())['events']
        excluded = sorted(e['symbol'] for e in events if e['available_from'] <= signal and e['symbol'] in attrs.index)
        if not excluded:
            return covariance, attrs, audit
        attrs = attrs.drop(index=excluded)
        return covariance, attrs, {**audit,'eligible':len(attrs),'known_unsupported_merger_exclusions':excluded,'scope_sha256':sha256(SCOPE)}

    def __call__(self, **c):
        case = c['kind'].split('--')[0]
        cadence = 20 if case == CORE else self.n4['sources'][case]['cadence']
        previous = self.contract
        try:
            self.contract = {**previous, 'rebalance_sessions': cadence}
            result = super().__call__(**c)
        finally:
            self.contract = previous
        decision_file = self.output / 'decisions' / c['kind'] / (c['day'] + '.json')
        decision = json.loads(decision_file.read_bytes())
        rights = c['night']['unsettled_stock_rights']
        equity = c['night']['equity']
        night_weights = {s: (c['night_holdings'].get(s,0)+rights.get(s,0))*c['last'][s]/equity for s in set(c['night_holdings']) | set(rights)}
        weights = dict(night_weights)
        if decision['status'] == 'OPEN_PLAN_VERIFIED':
            # These are previous-close share plans, before opening permissions/cash clipping.
            desired = decision['target_quantities']
            weights = {}
            for s in set(desired) | set(c['snapshot']['unsettled_stock_rights']):
                q = desired.get(s,0) + c['snapshot']['unsettled_stock_rights'].get(s,0)
                for event in c['actions'].loc[c['actions'].symbol.eq(s)&c['actions'].ex_date.eq(pd.Timestamp(c['day']))].itertuples():
                    q /= 1+event.stock_ratio
                weights[s] = q*c['last'][s]/equity
        for s in set(weights)-c['members']:
            # Intention is mandatory exit even if actual sleeve was blocked at next open.
            weights.pop(s)
        save(self.output / 'intents' / case / (c['day']+'.json'), {'signal_date': c['signal'], 'execution_date': c['day'], 'case': case, 'scheduled': c['index'] % cadence == 0, 'source_status': decision['status'], 'weights': weights, 'night_weights': night_weights, 'next_open_fills_used_for_selection': False})
        return result


def run(period, bps, output, smoke=False):
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(CONFIG.read_bytes())
    save(output / 'contract.json', config)
    loaded = load_inputs(period)
    planner = SleevePlanner(output, period, loaded)
    hashes = {str(p): sha256(p) for p in list(dict.fromkeys(old.source_files()+[Path(__file__),CONFIG,SCOPE]))}
    save(output / 'runtime.json', {'status':'RUNNING','pid':os.getpid(),'period':period,'bps':bps,'source_hashes':hashes,'prediction_sha256':planner.source_hash})
    old.engine.load_inputs = lambda: loaded
    old.engine.KINDS = [s+'--rank_cap' for s in SOURCES]
    old.engine.OUT = output
    try:
        old.engine.run(bps, smoke=smoke, execution_planner=planner, config_overrides={'initial_cash':config['initial_cash'],'max_position_weight':config['max_stock_weight']})
        save(output / 'completed.json', {'status':'COMPLETE','period':period,'bps':bps,'source_hashes':hashes,'production_modified':False})
    except Exception as error:
        save(output / 'failure.json', {'status':'FAILED','error_type':type(error).__name__,'error':str(error)})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--period', choices=['history','evaluation'], required=True)
    p.add_argument('--bps', type=int, choices=[10,20], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    with threadpool_limits(limits=2):
        run(args.period,args.bps,args.output,args.smoke)
