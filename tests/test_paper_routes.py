import pandas as pd
import pytest
from moneymore.research.paper_routes import earnings_yield, buffered_targets, monthly_inverse_variance


def test_ep_full_market_size_filter_and_missing():
    f = pd.DataFrame({'ts_code':list('abcde'), 'total_mv':[1,2,30,40,50], 'pe_ttm':[1,2,10,0,-5]})
    s = earnings_yield(f, set('abcde'))
    assert set(s.index) == {'c','e'}
    assert s.c == .1 and s.e == -.2


def test_buffer_retains_middle_rank_and_does_not_equal_weight_again():
    scores = pd.Series([5,4,3,2,1],index=list('abcde'))
    target = buffered_targets(scores,pd.Series({'c':.3,'e':.2}),topk=2,exit_rank=4)
    assert target == {'c':.3,'a':.7}


def test_variance_uses_no_current_incomplete_month():
    idx = pd.bdate_range('2024-01-01','2026-03-31')
    ret = pd.Series([.01 if i%2 else -.01 for i in range(len(idx))], index=idx)
    old, c = monthly_inverse_variance(ret,['20260205'])
    ret.loc['2026-02-06':] *= 30
    new, other_c = monthly_inverse_variance(ret,['20260205'])
    assert old == new and c == other_c


def test_buffer_keeps_reference_five_exit_cap():
    scores=pd.Series(range(200,0,-1),index=[str(i) for i in range(200)])
    held=pd.Series({str(i):.02 for i in range(150,200)})
    targets=buffered_targets(scores,held)
    assert len(set(held.index)-set(targets)) == 5
    assert len(targets) == 50
