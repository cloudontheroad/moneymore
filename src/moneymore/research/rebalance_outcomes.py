"""Separate conservative no-trade decisions from execution infrastructure faults."""


def classify_rebalance(event):
    if event['status']=='PASS_REFERENCE_QUOTES_ONLY':return 'executed'
    if event['status']=='TERMINAL_EXIT_ATTEMPT':return 'terminal_attempt_not_normal_rebalance'
    reason=event.get('reason') or ''
    if reason.startswith(('INTEGER_FEASIBILITY_USER_LIMIT','WARM_START_SEED_USER_LIMIT','INTEGER_SOLVER_ERROR')):
        return 'solver_incomplete'
    if reason.startswith(('IN_POOL_SCORE_MISSING_NOT_EXIT:', 'HELD_RISK_DATA_MISSING_NOT_A_POOL_EXIT:')):
        return 'data_policy_no_trade_not_model_success'
    if reason=='CONTINUOUS_SOLVER_INFEASIBLE':return 'constraint_conflict_requires_diagnosis'
    return 'unclassified_failure'
