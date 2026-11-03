"""Portable finite integer problem inputs for same-input solver validation."""
import pandas as pd
from moneymore.config import BacktestConfig


def encode_case(targets,prices,quantities,available,cash,equity,config,styles,limits,**kwargs):
    return dict(targets=targets.to_dict(),prices=dict(prices),quantities=quantities,available=available,
                cash=float(cash),equity=float(equity),config=config.model_dump(),styles=styles.to_dict(orient='split'),
                limits=limits.to_dict(),kwargs={k:(v.to_dict() if isinstance(v,pd.Series) else v) for k,v in kwargs.items()})


def decode_case(case):
    return (pd.Series(case['targets']),pd.Series(case['prices']),case['quantities'],case['available'],
            case['cash'],case['equity'],BacktestConfig(**case['config']),pd.DataFrame(**case['styles']),pd.Series(case['limits'])),case['kwargs']
