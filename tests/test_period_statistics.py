import numpy as np
import pytest
from moneymore.research.period_statistics import period_statistics

def test_statistics_include_first_day_and_consistent_annualization():
    r=np.array([.01,-.02,.03]);nav=100*np.cumprod(1+r)
    result=period_statistics(nav,100)
    assert result['net_return']==pytest.approx(np.prod(1+r)-1)
    assert result['annualized_volatility']==pytest.approx(r.std(ddof=1)*np.sqrt(252))
    assert result['sharpe']==pytest.approx(r.mean()/r.std(ddof=1)*np.sqrt(252))
    assert result['maximum_drawdown']==pytest.approx(.02)

def test_flat_return_does_not_invent_a_sharpe():
    assert period_statistics([100,100,100],100)['sharpe'] is None
    with pytest.raises(ValueError):period_statistics([100,float('nan')],100)
