import numpy as np
import pandas as pd
from moneymore.research.walkforward_protocol import purged_fold,block_interval


def test_labels_mature_strictly_before_next_segment():
    days=pd.bdate_range('2023-01-01',periods=400)
    segments=purged_fold(days,days[0],days[200],days[260])
    for left,right in [('train','valid'),('valid','test')]:
        assert days.get_loc(pd.Timestamp(segments[left][1]))+5<days.get_loc(pd.Timestamp(segments[right][0]))
    assert days.get_loc(pd.Timestamp(segments['test'][1]))+5==260


def test_day_block_intervals_are_deterministic_and_handle_constant_control():
    x=np.linspace(-.02,.03,80)
    assert block_interval(x)==block_interval(x)
    result=block_interval(np.zeros(50))
    assert result['lower']==result['upper']==0.
