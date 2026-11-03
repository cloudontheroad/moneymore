import numpy as np
import pandas as pd
from moneymore.research.rank_learning_target import qlib_rank_learning_target


def test_native_rank_target_only_changes_supervised_labels():
    index=pd.MultiIndex.from_product([pd.to_datetime(['2024-01-02','2024-01-03']),['A','B','C']],names=['datetime','instrument'])
    frame=pd.DataFrame({('feature','x'):np.arange(6.),('label','TOTAL_RETURN'):[.1,.2,.3,.4,.5,.6]},index=index)
    mask=np.array([True]*3+[False]*3)
    result=qlib_rank_learning_target(frame,mask)
    expected=(pd.Series([.1,.2,.3]).rank(pct=True)-.5)*3.46
    np.testing.assert_allclose(result.loc[mask,('label','TOTAL_RETURN')],expected)
    pd.testing.assert_frame_equal(frame['feature'],result['feature'])
    pd.testing.assert_series_equal(frame.loc[~mask,('label','TOTAL_RETURN')],result.loc[~mask,('label','TOTAL_RETURN')])


def test_test_label_changes_cannot_affect_rank_training_target():
    index=pd.MultiIndex.from_product([pd.to_datetime(['2024-01-02','2024-01-03']),['A','B']],names=['datetime','instrument'])
    frame=pd.DataFrame({('feature','x'):[1.,2.,3.,4.],('label','TOTAL_RETURN'):[.1,.2,.3,.4]},index=index)
    mask=np.array([True,True,False,False]);changed=frame.copy();changed.loc[~mask,('label','TOTAL_RETURN')]=1e9
    a=qlib_rank_learning_target(frame,mask);b=qlib_rank_learning_target(changed,mask)
    pd.testing.assert_frame_equal(a.loc[mask],b.loc[mask])
