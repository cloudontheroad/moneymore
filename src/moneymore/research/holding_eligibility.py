"""Separate dated membership, signal availability and risk coverage.

No score imputation, stale-signal reuse, or implicit liquidation is authorized.
"""


def classify_holdings(held, candidates, scored, risk_covered):
    held, candidates, scored, risk_covered = map(set, (held, candidates, scored, risk_covered))
    return {
        "exited_pool": sorted(held - candidates),
        "in_pool_missing_score": sorted((held & candidates) - scored),
        "missing_risk": sorted(held - risk_covered),
        "eligible_scored": sorted(held & candidates & scored & risk_covered),
    }


def explicit_score_membership(selected, scores, candidate_symbols):
    """Only explicit dated exits may have a missing alpha score.

    Missing in-pool scores block planning, including held names; assigning zero
    would invent a forecast and treating them as exits would invent a sale.
    """
    candidates = set(candidate_symbols)
    missing = sorted((set(selected) & candidates) - set(scores.dropna().index))
    if missing:
        raise ValueError("IN_POOL_SCORE_MISSING_NOT_EXIT:" + ",".join(missing))
    return ~selected.isin(candidates)
