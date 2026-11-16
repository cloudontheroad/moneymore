from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from moneymore.research.rank_preserving import allocate_ranked, sector_relative_scores


def test_sector_relative_rank_is_independent_of_other_sector_scale_and_preserves_ties():
    scores = pd.Series([2, 2, 5, 100, 200, 300, 9], index=list("abcdefg"), name="score")
    industries = pd.Series(list("AAABBBC"), index=scores.index)
    ranked = sector_relative_scores(scores, industries)
    assert ranked.to_dict() == {"a": 0.5, "b": 0.5, "c": 1.0,
                                "d": 1 / 3, "e": 2 / 3, "f": 1.0, "g": 0.5}
    transformed = scores.copy()
    transformed.loc[list("def")] = scores.loc[list("def")] * 1000 + 50000
    pd.testing.assert_series_equal(ranked, sector_relative_scores(transformed, industries))
    pd.testing.assert_series_equal(scores, pd.Series([2, 2, 5, 100, 200, 300, 9],
                                                    index=list("abcdefg"), name="score"))


def test_only_explicit_cross_section_is_read_and_future_panels_are_rejected():
    panel = pd.Series([2, 1, -999, 999], index=pd.MultiIndex.from_product(
        [["2026-11-11", "2026-11-12"], ["a", "b"]]
    ))
    industries = pd.Series({"a": "A", "b": "A", "future_only": "FUTURE"})
    current = panel.loc["2026-11-11"]
    before = sector_relative_scores(current, industries)
    panel.loc["2026-11-12"] = [999999, -999999]
    pd.testing.assert_series_equal(before, sector_relative_scores(panel.loc["2026-11-11"], industries))
    with pytest.raises(ValueError, match="SINGLE_CROSS_SECTION_REQUIRED"):
        sector_relative_scores(panel, industries)
    with pytest.raises(ValueError, match="SINGLE_CROSS_SECTION_REQUIRED"):
        allocate_ranked(panel, industries, {"A": 1.0}, [])


def test_default_fifty_equal_slots_target_995_and_respect_industry_caps():
    scores = pd.Series(np.arange(100, dtype=float), index=[f"s{i:03}" for i in range(100)])
    industries = pd.Series([f"sector{i % 5}" for i in range(100)], index=scores.index)
    targets, audit = allocate_ranked(scores, industries, {f"sector{i}": .2 for i in range(5)}, [])
    assert len(targets) == 50
    assert set(targets.values()) == {.0199}
    assert sum(targets.values()) == pytest.approx(.995)
    assert audit["status"] == "OK"
    assert audit["cash_weight"] == pytest.approx(.005)
    assert all(weight <= .28 for weight in audit["total_industry_weights"].values())
    assert audit["industry_count_caps"] == {f"sector{i}": 14 for i in range(5)}


@pytest.mark.parametrize("seed", range(10))
def test_unbuffered_partition_greedy_matches_exhaustive_maximum(seed):
    rng = np.random.default_rng(seed)
    scores = pd.Series(rng.normal(size=8), index=list("abcdefgh"))
    industries = pd.Series(list("AAABBBCC"), index=scores.index)
    targets, audit = allocate_ranked(scores, industries, {"A": .2, "B": .2, "C": .1}, [],
                                     slots=4, slot_weight=.1, industry_active_limit=0)
    feasible = [group for group in combinations(scores.index, 4)
                if all(sum(industries[s] == sector for s in group) <= capacity
                       for sector, capacity in audit["industry_count_caps"].items())]
    assert scores.loc[list(targets)].sum() == pytest.approx(
        max(scores.loc[list(group)].sum() for group in feasible)
    )


def test_score_ties_are_deterministic_but_buffer_never_churns_equal_scores():
    scores = pd.Series(1.0, index=list("dcba"))
    industries = pd.Series("A", index=scores.index)
    kwargs = {"slots": 2, "slot_weight": .1, "industry_active_limit": 0}
    unbuffered, _ = allocate_ranked(scores, industries, {"A": 1}, [], **kwargs)
    shuffled, _ = allocate_ranked(scores.iloc[::-1], industries.iloc[::-1], {"A": 1}, [], **kwargs)
    assert list(unbuffered) == list(shuffled) == ["a", "b"]
    buffered, audit = allocate_ranked(scores, industries, {"A": 1}, ["c", "d"],
                                      max_optional_exits=5, **kwargs)
    assert set(buffered) == {"c", "d"}
    assert audit["optional_exits"] == []


def test_buffer_drops_at_most_five_and_zero_buffer_retains_old_members():
    scores = pd.Series(np.arange(100, dtype=float), index=[f"s{i:03}" for i in range(100)])
    industries = pd.Series("A", index=scores.index)
    old = list(scores.index[:50])
    targets, audit = allocate_ranked(scores, industries, {"A": 1}, old, max_optional_exits=5)
    assert len(targets) == 50
    assert len(set(old) - set(targets)) == 5
    assert set(targets) - set(old) == set(scores.index[-5:])
    assert audit["optional_exits"] == list(scores.index[:5])
    assert audit["risk_forced_exits"] == []
    assert all(swap["score_gain"] > 0 for swap in audit["beneficial_swaps"])
    unchanged, unchanged_audit = allocate_ranked(scores, industries, {"A": 1}, old,
                                                max_optional_exits=0)
    assert set(unchanged) == set(old)
    assert unchanged_audit["membership_turnover"] == {"entries": 0, "exits": 0}


def test_sector_risk_repairs_are_mandatory_and_outside_optional_exit_budget():
    scores = pd.Series([1, 2, 3, 4, 5, 6, 7], index=list("abcdefg"))
    industries = pd.Series(list("AAAABBB"), index=scores.index)
    targets, audit = allocate_ranked(scores, industries, {"A": .1, "B": .3}, list("abcd"),
                                     slots=4, slot_weight=.1, industry_active_limit=0,
                                     max_optional_exits=0)
    assert set(targets) == set("defg")
    assert audit["risk_forced_exits"] == list("abc")
    assert audit["optional_exits"] == []
    assert audit["total_industry_weights"] == {"A": .1, "B": pytest.approx(.3)}
    assert audit["status"] == "OK"


def test_buffer_can_swap_across_sectors_but_cannot_overfill_destination_sector():
    scores = pd.Series({"old_a": 0., "old_b": 10., "new_b": 9., "new_c": 8.})
    industries = pd.Series({"old_a": "A", "old_b": "B", "new_b": "B", "new_c": "C"})
    targets, audit = allocate_ranked(scores, industries, {"A": .1, "B": .1, "C": .1},
                                     ["old_a", "old_b"], slots=2, slot_weight=.1,
                                     industry_active_limit=0, max_optional_exits=1)
    assert set(targets) == {"old_b", "new_c"}
    assert audit["optional_exits"] == ["old_a"]


@pytest.mark.parametrize("seed", range(20))
def test_buffer_is_deterministic_maximum_cardinality_and_has_bounded_optional_exits(seed):
    rng = np.random.default_rng(seed)
    scores = pd.Series(rng.integers(-5, 6, size=20), index=[f"s{i:02}" for i in range(20)])
    industries = pd.Series(rng.choice(list("ABCD"), size=20), index=scores.index)
    old = list(rng.choice(scores.index, size=9, replace=False))
    benchmark = dict(zip(list("ABCD"), [.15, .20, .15, .10], strict=True))
    kwargs = {"slots": 8, "slot_weight": .05, "industry_active_limit": 0,
              "max_optional_exits": 2}
    targets, audit = allocate_ranked(scores, industries, benchmark, old, **kwargs)
    reversed_targets, reversed_audit = allocate_ranked(
        scores.iloc[::-1], industries.iloc[::-1], benchmark, old[::-1], **kwargs
    )
    assert targets == reversed_targets
    assert audit == reversed_audit
    assert len(targets) == min(audit["effective_slots"], audit["available_capacity"])
    assert len(audit["optional_exits"]) <= 2
    assert set(audit["removed_symbols"]) == (
        set(audit["risk_forced_exits"]) | set(audit["slot_capacity_forced_exits"])
        | set(audit["optional_exits"])
    )
    assert all(audit["total_industry_weights"][sector] <= benchmark[sector] + 1e-12
               for sector in benchmark)
    no_swaps, no_swaps_audit = allocate_ranked(scores, industries, benchmark, old,
                                              **{**kwargs, "max_optional_exits": 0})
    assert len(targets) == len(no_swaps)
    assert scores.loc[list(targets)].sum() >= scores.loc[list(no_swaps)].sum()
    assert no_swaps_audit["optional_exits"] == []


def test_frozen_weights_consume_total_and_sector_capacity_and_fraction_stays_cash():
    scores = pd.Series(np.arange(10), index=[f"s{i}" for i in range(10)])
    industries = pd.Series(list("AAAAABBBBB"), index=scores.index)
    targets, audit = allocate_ranked(scores, industries, {"A": .3, "B": .2}, [], slots=5,
                                     slot_weight=.1, industry_active_limit=0,
                                     frozen_industry_weights={"A": .15})
    assert len(targets) == 3
    assert audit["effective_slots"] == 3
    assert audit["industry_count_caps"] == {"A": 1, "B": 2}
    assert audit["total_weight"] == pytest.approx(.45)
    assert audit["unallocated_budget"] == pytest.approx(.05)
    assert audit["total_industry_weights"] == {"A": .25, "B": .2}
    assert audit["status"] == "OK"


def test_slot_budget_repairs_are_audited_separately_from_sector_repairs():
    scores = pd.Series([1, 2, 3], index=list("abc"))
    industries = pd.Series("A", index=scores.index)
    targets, audit = allocate_ranked(scores, industries, {"A": 1}, list("abc"), slots=2,
                                     slot_weight=.1, max_optional_exits=0)
    assert set(targets) == set("bc")
    assert audit["risk_forced_exits"] == []
    assert audit["slot_capacity_forced_exits"] == ["a"]
    assert audit["optional_exits"] == []


def test_infeasible_capacity_is_partial_and_never_silently_relaxes_caps():
    scores = pd.Series([3, 2, 1], index=list("abc"))
    industries = pd.Series("A", index=scores.index)
    targets, audit = allocate_ranked(scores, industries, {"A": .1}, [], slots=3,
                                     slot_weight=.1, industry_active_limit=0)
    assert targets == {"a": .1}
    assert audit["status"] == "INSUFFICIENT_CAPACITY"
    assert audit["available_capacity"] == 1
    assert audit["unallocated_budget"] == pytest.approx(.2)


def test_empty_candidate_set_is_explicitly_infeasible_not_filled_with_invalid_names():
    targets, audit = allocate_ranked(pd.Series(dtype=float), pd.Series(dtype=object), {"A": 1}, [])
    assert targets == {}
    assert audit["status"] == "INSUFFICIENT_CAPACITY"
    assert audit["selected_count"] == 0


@pytest.mark.parametrize("frozen,error", [({"A": .4}, "FROZEN_INDUSTRY_CAP_VIOLATION"),
                                         ({"A": .6}, "FROZEN_EXPOSURE_EXCEEDS_TARGET")])
def test_infeasible_frozen_weights_fail_closed(frozen, error):
    with pytest.raises(ValueError, match=error):
        allocate_ranked(pd.Series({"a": 1}), pd.Series({"a": "A"}), {"A": .3}, [],
                        slots=5, slot_weight=.1, industry_active_limit=0,
                        frozen_industry_weights=frozen)


def test_missing_old_score_is_not_assumed_permission_to_exit():
    with pytest.raises(ValueError, match="HELD_SCORE_MISSING_NOT_AN_EXIT"):
        allocate_ranked(pd.Series({"a": 1}), pd.Series({"a": "A"}), {"A": 1}, ["missing"])


@pytest.mark.parametrize("scores,industries,error", [
    (pd.Series([1, 2], index=["a", "a"]), pd.Series({"a": "A"}), "DUPLICATE"),
    (pd.Series({"a": np.nan}), pd.Series({"a": "A"}), "NONFINITE"),
    (pd.Series({"a": np.inf}), pd.Series({"a": "A"}), "NONFINITE"),
    (pd.Series({"a": 1}), pd.Series({"other": "A"}), "MISSING_OR_INVALID_INDUSTRY"),
])
def test_invalid_cross_sections_fail_closed(scores, industries, error):
    with pytest.raises(ValueError, match=error):
        sector_relative_scores(scores, industries)
    with pytest.raises(ValueError, match=error):
        allocate_ranked(scores, industries, {"A": 1}, [])
