import numpy as np
import pandas as pd
import pytest
from moneymore.research.score_neutralization import neutralize_scores


def test_pure_style_score_has_no_residual_signal():
    c=pd.DataFrame({'size':np.arange(20.),'sector':[0.]*10+[1.]*10})
    score=3*c['size']+2*c.sector+7
    residual,meta=neutralize_scores(score,c)
    assert residual.eq(0).all() and not meta['future_labels_used']


def test_residual_is_orthogonal_and_bad_keys_are_rejected():
    rng=np.random.default_rng(17);c=pd.DataFrame({'vol':rng.normal(size=100),'constant':1.})
    score=pd.Series(2*c.vol+rng.normal(size=100));r,m=neutralize_scores(score,c)
    assert abs(float(r @ c.vol))<1e-9 and r.std()>0
    pd.testing.assert_series_equal(r,neutralize_scores(score,c)[0])
    with pytest.raises(ValueError):neutralize_scores(score,c.iloc[::-1])
