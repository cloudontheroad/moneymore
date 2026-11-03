import pandas as pd
import pytest
from moneymore.research.missing_group_drift import missing_group_drift


def test_label_migration_is_not_price_drift():
    w=pd.Series([.5,.5],index=["A","B"])
    result=missing_group_drift(pd.Series([0.,0.],index=w.index),pd.Series([1.,0.],index=w.index),
        w,w,pd.Series([.2,.8],index=w.index),pd.Series([.2,.8],index=w.index))
    assert result["portfolio_price_drift"]==0
    assert result["portfolio_label_migration"]==.5
    assert result["benchmark_label_migration"]==-.2
    assert abs(result["error"]-.3)<1e-12


def test_benchmark_entries_are_not_assigned_fake_old_labels():
    result=missing_group_drift(pd.Series({"A":1.,"B":0.}),pd.Series({"A":1.,"C":1.}),
        pd.Series({"A":1.}),pd.Series({"A":1.}),pd.Series({"A":.5,"B":.5}),pd.Series({"A":.25,"C":.75}))
    assert result["benchmark_entry"]==-.75
    assert result["benchmark_weight_drift"]==.25
    assert result["error"]==-.5
    assert abs(result["identity_residual"])<1e-12


def test_unobserved_held_label_is_not_zero_filled():
    with pytest.raises(ValueError,match="UNOBSERVABLE"):
        missing_group_drift(pd.Series({"A":1.}),pd.Series({"B":0.}),pd.Series({"A":1.}),
            pd.Series({"A":1.}),pd.Series({"A":1.}),pd.Series({"B":1.}))
