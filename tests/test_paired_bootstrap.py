import numpy as np
import pandas as pd
import pytest
from moneymore.research.paired_bootstrap import paired_block_intervals


def test_same_series_zero_and_reproducible():
    x=pd.Series(np.random.default_rng(1).normal(0,.01,100),index=pd.date_range('2025-01-01',periods=100))
    result=paired_block_intervals(x,x,draws=100)
    assert result==paired_block_intervals(x,x,draws=100)
    assert not result['all_three_lower_bounds_positive']
    assert all(v==0 for metric in result['metrics'].values() for v in metric.values())


def test_reject_date_mismatch_and_missing():
    x=pd.Series(np.arange(100)/10000)
    with pytest.raises(ValueError,match='UNPAIRED'):paired_block_intervals(x,x.iloc[::-1],draws=100)
    y=x.copy();y.iloc[4]=np.nan
    with pytest.raises(ValueError,match='INVALID'):paired_block_intervals(y,x,draws=100)
