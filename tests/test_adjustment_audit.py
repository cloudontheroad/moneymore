import pandas as pd
import pytest
from moneymore.research.adjustment_audit import audit_adjustment_versions


def test_precision_and_constant_units_are_not_return_path_changes():
    rows=pd.DataFrame([{'date':d,'symbol':s,'cached':v,'official':o} for s,values in
        [('rounding',[(1.234567,1.2346),(1.234567,1.2346)]),('scale',[(2.,1.),(4.,2.)])]
        for d,(v,o) in zip(['1','2'],values)])
    result=audit_adjustment_versions(rows)
    assert not result.ratio_path_change.any()
    assert result.loc[result.symbol.eq('rounding'),'same_reported_precision'].all()
    assert result.loc[result.symbol.eq('scale'),'constant_scale_candidate'].all()


def test_changed_factor_path_and_invalid_duplicate_are_explicit():
    data=pd.DataFrame({'date':['1','2'],'symbol':['S','S'],'cached':[1.,1.03],'official':[1.,1.]})
    assert audit_adjustment_versions(data).ratio_path_change.all()
    with pytest.raises(ValueError):audit_adjustment_versions(pd.concat([data,data]))
