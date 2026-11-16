"""Native Riskfolio worker in isolated environment; strictly lagged net-return inputs."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
from pathlib import Path
import numpy as np
import pandas as pd
import riskfolio as rp

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'state/ashare-mainline-20261112'
CORE = 'doubleensemble_full'
SOURCES = [CORE,'fundamental','trend','short']
spec = importlib.util.spec_from_file_location('n4_pure_allocation',ROOT/'src/moneymore/research/n4_allocation.py')
pure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pure)
past_returns, validate_weights = pure.past_returns, pure.validate_weights


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main(runs, bps, output):
    config = json.loads((ROOT / 'configs/ashare_n4_v1.json').read_bytes())
    spec = config['allocation']
    if importlib.metadata.version('riskfolio-lib') != spec['riskfolio_version']:
        raise ValueError('RISKFOLIO_VERSION_MISMATCH')
    paths, periods = {}, []
    for period in ['history','evaluation']:
        columns = {}
        for case in SOURCES:
            p = runs / f'{period}-cost{bps}' / f'bps{bps}' / (case+'--rank_cap') / 'equity.parquet'
            frame = pd.read_parquet(p).set_index('date').equity
            columns[case] = frame / frame.shift(1).fillna(config['initial_cash']) - 1
            paths[str(p)] = sha256(p)
        periods.append(pd.DataFrame(columns))
    returns = pd.concat(periods).sort_index()
    calendar = pd.DatetimeIndex(pd.to_datetime((ROOT / 'state/mature-multifactor/p1-native-20261103-a/provider/calendars/day.txt').read_text().splitlines())).strftime('%Y%m%d').tolist()
    records = []
    for i, day in enumerate(periods[1].index):
        if i % spec['rebalance_sessions']:
            continue
        signal = calendar[calendar.index(day)-1]
        past = past_returns(returns, signal, spec['lookback'], spec['minimum_observations'])
        eligible = [s for s in SOURCES if s==CORE or float((1+past[s]).prod()-1)>0]
        for method in spec['methods']:
            chosen = eligible if method=='hrp_positive' else SOURCES
            if method=='equal':
                weights = dict.fromkeys(SOURCES,1/len(SOURCES))
            elif len(chosen)==1:
                weights = {CORE:1.}
            else:
                port = rp.HCPortfolio(returns=past[chosen],w_max=spec['maximum_sleeve_weight'])
                answer = port.optimization(model=spec['model'],codependence=spec['codependence'],rm=spec['risk_measure'],linkage=spec['linkage'])
                if answer is None:
                    raise ValueError('NATIVE_HRP_RETURNED_NONE')
                weights = answer['weights'].to_dict()
            weights = validate_weights(weights,SOURCES,spec['maximum_sleeve_weight'])
            covariance = past.cov().reindex(index=SOURCES,columns=SOURCES)
            risk = weights * (covariance @ weights)
            risk /= float(weights @ covariance @ weights)
            records.append({'method':method,'execution_date':day,'signal_date':signal,'return_start':past.index[0],'return_end':past.index[-1],'observations':len(past),'weights':weights.to_dict(),'ex_ante_variance_contribution':risk.to_dict(),'eligible':chosen,'prior_net_returns':((1+past).prod()-1).to_dict(),'correlation':past.corr().to_dict(),'native_called':method!='equal' and len(chosen)>1})
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        raise FileExistsError('IMMUTABLE_ALLOCATION')
    output.write_text(json.dumps({'status':'COMPLETE','bps':bps,'records':records,'input_hashes':paths,'worker_sha256':sha256(Path(__file__)),'native_source_sha256':sha256(Path(inspect.getsourcefile(rp.HCPortfolio))),'versions':{p:importlib.metadata.version(p) for p in ['riskfolio-lib','numpy','pandas','scipy']},'config_sha256':sha256(ROOT/'configs/ashare_n4_v1.json'),'production_enabled':False},indent=2),encoding='utf-8')
    print(json.dumps({'status':'COMPLETE','bps':bps,'allocations':len(records)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--bps',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    main(a.runs,a.bps,a.output)
