import pytest
from moneymore.research.rebalance_outcomes import classify_rebalance


@pytest.mark.parametrize('reason,expected',[
    ('INTEGER_FEASIBILITY_USER_LIMIT','solver_incomplete'),
    ('WARM_START_SEED_USER_LIMIT','solver_incomplete'),
    ('IN_POOL_SCORE_MISSING_NOT_EXIT:A','data_policy_no_trade_not_model_success'),
    ('HELD_RISK_DATA_MISSING_NOT_A_POOL_EXIT:A','data_policy_no_trade_not_model_success'),
    ('CONTINUOUS_SOLVER_INFEASIBLE','constraint_conflict_requires_diagnosis'),
    ('UNEXPECTED','unclassified_failure')])
def test_no_trade_is_not_execution_success(reason,expected):
    assert classify_rebalance({'status':'BLOCKED','reason':reason})==expected


def test_only_explicit_pass_counts_as_executed():
    assert classify_rebalance({'status':'PASS_REFERENCE_QUOTES_ONLY'})=='executed'
    assert classify_rebalance({'status':'TERMINAL_EXIT_ATTEMPT'})!='executed'
