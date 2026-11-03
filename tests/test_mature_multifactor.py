import numpy as np
import pandas as pd
import pytest
from moneymore.research.mature_multifactor import native_bar_fields, temporal_segments, dated_membership, topk_dropout_targets


def test_native_price_volume_vwap_units_and_no_fill():
    frame = pd.DataFrame({'trade_date':['20260101','20260102','20260105'], 'open':[10.,5.,6.], 'high':[11.,6.,7.],
        'low':[9.,4.,5.], 'close':[10.,5.,6.], 'vol':[100.,200.,0.], 'amount':[100.,100.,0.], 'adj_factor':[1.,2.,np.nan]})
    native, check = native_bar_fields(frame)
    assert native.close.iloc[:2].tolist() == [10.,10.]
    assert native.volume.iloc[:2].tolist() == [10000.,10000.]
    assert native.vwap.iloc[:2].tolist() == [10.,10.]
    assert pd.isna(native.vwap.iloc[2]) and pd.isna(native.close.iloc[2])
    assert check['invalid_vwap'] == 0


def test_split_purges_future_label_and_keeps_latest_unmatured_features():
    days = pd.bdate_range('20240101', periods=500)
    segments = temporal_segments(days)
    assert days[days.get_loc(segments['train'][1])+5] < pd.Timestamp(segments['valid'][0])
    assert days[days.get_loc(segments['valid'][1])+5] < pd.Timestamp(segments['test'][0])
    assert days[days.get_loc(segments['deployment_fit'][1])+5] == days[-1]
    assert pd.Timestamp(segments['latest'][0]) == days[-1]


def test_membership_never_backfills_future_snapshot():
    history = pd.DataFrame({'signal_date':['20260102'],'trade_date':['20251231'],'ts_code':['600000.SH']})
    snapshots = pd.DataFrame({'effective_date':['20260106'],'market_data_date':['20260106'],'symbol':['300001.SZ']})
    index, carried = dated_membership(['20260102','20260105','20260106'],history,snapshots,'20260102')
    assert index.get_level_values('instrument').tolist() == ['600000.SH','600000.SH','300001.SZ']
    assert carried == [{'date':'20260105','membership_as_of':'20260102','size':1}]
    snapshots['market_data_date'] = '20260107'
    with pytest.raises(ValueError,match='FUTURE_UNIVERSE_SOURCE'):
        dated_membership(['20260106'],history,snapshots,'20260102')


def test_topk_dropout_does_not_rebalance_survivors_or_sell_missing_data():
    score=pd.Series(np.arange(100,dtype=float),index=[str(i) for i in range(100)])
    target=topk_dropout_targets(score,pd.Series(dtype=float))
    assert len(target)==50 and abs(sum(target.values())-1)<1e-10
    held=pd.Series(.02,index=[str(i) for i in range(50)])
    target=topk_dropout_targets(score,held)
    assert len(set(held.index)-set(target))==5
    assert all(target[s]==.02 for s in set(held.index)&set(target))
    with pytest.raises(ValueError,match='NOT_AN_EXIT'):
        topk_dropout_targets(score.drop('0'),held)
