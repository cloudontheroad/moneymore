import numpy as np
import pandas as pd
import pytest
from moneymore.research.named_factor_history import named_return,named_covariance


def test_named_ols_recovers_known_industry_and_style_not_pca_names():
    rng=np.random.default_rng(17);x=pd.DataFrame({'industry_a':[1.]*50+[0.]*50,'industry_b':[0.]*50+[1.]*50,'size':rng.normal(size=100)})
    beta=pd.Series({'industry_a':.01,'industry_b':-.02,'size':.003});y=x@beta
    fitted,residual,audit=named_return(x,y,'20240101','20240102')
    assert abs(fitted-beta).max()<1e-12 and abs(residual).max()<1e-12
    assert audit['rank']==3


def test_future_or_aliased_named_factors_are_rejected():
    x=pd.DataFrame({'a':np.ones(100),'alias':np.ones(100)});y=pd.Series(np.zeros(100))
    with pytest.raises(ValueError,match='NONCAUSAL'):named_return(x,y,'20240102','20240102')
    with pytest.raises(ValueError,match='NOT_IDENTIFIABLE'):named_return(x,y,'20240101','20240102')


def test_covariance_does_not_read_future_and_is_psd():
    dates=pd.date_range('20240101',periods=130).strftime('%Y%m%d');r=pd.DataFrame(np.random.default_rng(17).normal(size=(130,3)),index=dates)
    before,audit=named_covariance(r,dates[119]);r.loc[dates[120]:]=1e9
    after,_=named_covariance(r,dates[119]);pd.testing.assert_frame_equal(before,after)
    assert audit['last']==dates[119] and np.linalg.eigvalsh(before).min()>=0
    with pytest.raises(ValueError,match='INSUFFICIENT_ASOF'):named_covariance(r,dates[118])
