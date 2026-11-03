"""Native Qlib target processing, restricted to supervised train/valid rows."""
import pandas as pd


def qlib_rank_learning_target(frame, supervised):
    from qlib.data.dataset.processor import CSRankNorm
    if len(supervised)!=len(frame):raise ValueError('SUPERVISED_MASK_LENGTH_MISMATCH')
    original=frame.copy(deep=True)
    transformed=CSRankNorm(fields_group='label')(frame.loc[supervised].copy(deep=True))
    result=frame.copy(deep=True)
    result.loc[supervised,('label','TOTAL_RETURN')]=transformed[('label','TOTAL_RETURN')].to_numpy()
    pd.testing.assert_frame_equal(original['feature'],result['feature'])
    pd.testing.assert_frame_equal(original.loc[~supervised,'label'],result.loc[~supervised,'label'])
    return result
