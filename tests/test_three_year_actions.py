import json
from pathlib import Path

import pandas as pd
import pytest

from moneymore.research.replay_actions import official_replay_actions
from moneymore.research.three_year_actions import AppendOnlyActionCache, reviewed_actions


def fixture():
    rule = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/ashare_three_year_action_evidence.json"
        ).read_bytes()
    )["cash_aggregate_overlap"][0]
    raw = pd.DataFrame(
        [
            {
                "ts_code": rule["symbol"],
                "end_date": end,
                "cash_div_tax": cash,
                "div_proc": "实施",
                "stk_div": 0.0,
                "div_listdate": None,
                **{k: rule[k] for k in ["record_date", "ex_date", "pay_date", "imp_ann_date"]},
            }
            for end, cash in rule["expected_rows"]
        ]
    )
    return rule, raw


def test_aggregate_is_not_counted_alongside_its_components():
    rule, raw = fixture()
    selected, audit = reviewed_actions(raw, [], [rule])
    assert len(raw) == 3
    assert len(audit) == 1
    assert audit[0]["excluded_rows"] == 2
    actual = official_replay_actions(selected)
    assert len(actual) == 1
    assert actual.cash_per_share.sum() == pytest.approx(1.4)


@pytest.mark.parametrize(
    "field,value", [("cash_div_tax", 2.0), ("pay_date", "20241108"), ("stk_div", 0.1)]
)
def test_evidence_cannot_silently_cover_changed_source(field, value):
    rule, raw = fixture()
    raw.loc[0, field] = value
    with pytest.raises(ValueError, match="EVIDENCED_AGGREGATE"):
        reviewed_actions(raw, [], [rule])


def test_cache_matches_uncached_and_never_releases_future_events():
    rule, raw = fixture()
    first, _ = reviewed_actions(raw, [], [rule])
    second = first.assign(
        ts_code="SECOND",
        record_date="20251106",
        ex_date="20251107",
        pay_date="20251107",
        imp_ann_date="20251031",
        end_date="20250630",
    )
    cache = AppendOnlyActionCache()
    assert cache(pd.DataFrame()).empty
    pd.testing.assert_frame_equal(cache(first), official_replay_actions(first))
    assert set(cache(first).symbol) == {rule["symbol"]}
    cumulative = pd.concat([first, second], ignore_index=True)
    pd.testing.assert_frame_equal(cache(cumulative), official_replay_actions(cumulative))
    changed = cumulative.copy()
    changed.loc[0, "cash_div_tax"] = 0.1
    with pytest.raises(ValueError, match="ACTION_CACHE_PREFIX_CHANGED"):
        cache(changed)
    with pytest.raises(ValueError, match="ACTION_CACHE_PREFIX_CHANGED"):
        cache(first)


def test_cache_still_blocks_conflicting_appended_version():
    rule, raw = fixture()
    selected, _ = reviewed_actions(raw, [], [rule])
    cache = AppendOnlyActionCache()
    cache(selected)
    duplicate = selected.assign(cash_div_tax=1.5)
    with pytest.raises(ValueError, match="CONFLICTING_IMPLEMENTED_ACTION_VERSIONS"):
        cache(pd.concat([selected, duplicate], ignore_index=True))


def test_duplicate_raw_versions_have_identical_event_values():
    rule, raw = fixture()
    selected, _ = reviewed_actions(raw, [], [rule])
    duplicated = pd.concat(
        [selected.assign(ann_date="20240829"), selected.assign(ann_date="20241030")],
        ignore_index=True,
    )
    cache = AppendOnlyActionCache()
    cache(duplicated.iloc[:1])
    pd.testing.assert_frame_equal(
        cache(duplicated).reset_index(drop=True),
        official_replay_actions(duplicated).reset_index(drop=True),
        check_exact=True,
    )
