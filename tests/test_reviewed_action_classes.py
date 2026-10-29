import pandas as pd
import pytest

from moneymore.research.reviewed_action_classes import select_reviewed_retail_actions


def sample():
    rule = {
        "symbol": "X",
        "end_date": "20231231",
        "record_date": "20240723",
        "ex_date": "20240724",
        "pay_date": "20240724",
        "imp_ann_date": "20240718",
        "observed_cash_classes": [0.265, 0.3158],
        "retail_cash_per_share": 0.3158,
    }
    raw = pd.DataFrame(
        [
            dict(
                ts_code="X",
                div_proc="实施",
                stk_div=0,
                cash_div_tax=c,
                **{
                    k: rule[k]
                    for k in ("end_date", "record_date", "ex_date", "pay_date", "imp_ann_date")
                },
            )
            for c in [0.265, 0.3158]
        ]
    )
    return raw, rule


def test_only_explicit_retail_class_selected_without_raw_mutation():
    raw, rule = sample()
    selected, audit = select_reviewed_retail_actions(raw, [rule])
    assert selected.cash_div_tax.tolist() == [0.3158]
    assert len(raw) == 2 and audit[0]["excluded_rows"] == 1


def test_unreviewed_or_changed_event_not_silently_resolved():
    raw, rule = sample()
    assert len(select_reviewed_retail_actions(raw, [])[0]) == 2
    raw.loc[0, "cash_div_tax"] = 0.266
    with pytest.raises(ValueError, match="AMOUNTS_CHANGED"):
        select_reviewed_retail_actions(raw, [rule])
