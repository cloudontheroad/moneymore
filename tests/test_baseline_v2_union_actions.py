from types import SimpleNamespace

import pandas as pd

from moneymore.research.baseline_v2_source_economics import audit_cached_actions


def test_action_alignment_keeps_conflicting_classes_and_unexplained_steps(tmp_path):
    cache = tmp_path / "actions.parquet"
    pd.DataFrame(
        [
            {"div_proc": "实施", "ex_date": "20240102", "cash_div_tax": 1, "stk_div": 0},
            {"div_proc": "实施", "ex_date": "20240103", "cash_div_tax": 1, "stk_div": 0},
            {"div_proc": "实施", "ex_date": "20240103", "cash_div_tax": 2, "stk_div": 0},
            {"div_proc": "预案", "ex_date": "20240104", "cash_div_tax": 1, "stk_div": 0},
        ]
    ).to_parquet(cache, index=False)
    changes = pd.DataFrame(
        {
            "ts_code": ["A"] * 3,
            "trade_date": ["20240102", "20240103", "20240104"],
            "previous_close": [10.0] * 3,
            "pre_close": [9.0] * 3,
        }
    )
    result = audit_cached_actions(
        changes, {"A": ("now", cache)}, SimpleNamespace(stable=lambda s: s), tmp_path
    )
    labels = pd.read_parquet(tmp_path / "full-union-action-alignment.parquet").action_alignment
    assert labels.tolist() == [
        "SIMPLE_ACTION_REFERENCE_ALIGNS_SAME_PROVIDER_ONLY",
        "DIFFERENTIATED_ACTION_CLASSES_REVIEW_REQUIRED",
        "NO_IMPLEMENTED_EVENT_IN_CACHE_UNRESOLVED",
    ]
    assert result["independent_economic_validation_passed"] is False
