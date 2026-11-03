import pandas as pd
import pytest

from moneymore.research.holding_eligibility import classify_holdings, explicit_score_membership


def test_exit_and_missing_score_are_independent_of_risk_coverage():
    result = classify_holdings(['exit', 'halt', 'ok'], ['halt', 'ok'], ['ok'], ['exit', 'ok'])
    assert result == {'exited_pool': ['exit'], 'in_pool_missing_score': ['halt'],
                      'missing_risk': ['halt'], 'eligible_scored': ['ok']}


def test_missing_in_pool_score_is_not_an_exit():
    with pytest.raises(ValueError, match='IN_POOL_SCORE_MISSING_NOT_EXIT:halt'):
        explicit_score_membership(pd.Index(['exit', 'halt', 'ok']), pd.Series({'ok': 1.}), ['halt', 'ok'])


def test_explicit_exit_does_not_require_invented_score():
    assert explicit_score_membership(pd.Index(['exit', 'ok']), pd.Series({'ok': 1.}), ['ok']).tolist() == [True, False]


def test_nan_score_blocks_but_zero_score_is_valid():
    index = pd.Index(['ok'])
    assert explicit_score_membership(index, pd.Series({'ok': 0.}), ['ok']).tolist() == [False]
    with pytest.raises(ValueError, match='IN_POOL_SCORE_MISSING'):
        explicit_score_membership(index, pd.Series({'ok': float('nan')}), ['ok'])


def test_optimizer_blocks_before_solving_when_membership_is_explicit():
    from moneymore.research.style_replay_inputs import optimize_daily
    index = pd.Index(['halt', 'ok'])
    with pytest.raises(ValueError, match='IN_POOL_SCORE_MISSING_NOT_EXIT:halt'):
        optimize_daily(pd.Series({'ok': 1.}), pd.DataFrame(index=index), None,
                       None, None, None, None, None, pd.Series(dtype=float),
                       candidate_symbols=['halt', 'ok'])
