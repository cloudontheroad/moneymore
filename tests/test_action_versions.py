import pandas as pd
import pytest
from moneymore.research.action_versions import latest_known_action_versions


def event(ann,cash):
    return dict(ts_code='S',end_date='20231231',record_date='20240723',ex_date='20240724',pay_date='20240724',
        div_listdate=None,ann_date=ann,imp_ann_date='20240718',div_proc='实施',cash_div_tax=cash,stk_div=0.)


def test_latest_proposal_known_by_record_date_not_largest_cash_is_selected():
    selected,audit=latest_known_action_versions(pd.DataFrame([event('20240322',.5),event('20240622',.3),event('20240725',1.)]))
    assert len(selected)==1 and selected.iloc[0].cash_div_tax==.3
    assert audit[0]['known_by_record']==2


def test_same_time_conflicts_are_not_silently_resolved():
    with pytest.raises(ValueError,match='SAME_TIME'):
        latest_known_action_versions(pd.DataFrame([event('20240622',.3),event('20240622',.5)]))
