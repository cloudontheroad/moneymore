import numpy as np
from moneymore.research.style_constraint_checks import constraint_violations


def test_extreme_style_violates_constraints_benchmark_passes():
    benchmark = np.full(20, .05)
    exposure = np.r_[np.ones(10), np.zeros(10)][:, None]
    assert constraint_violations(benchmark, benchmark, benchmark, exposure, [0]) == []
    extreme = np.r_[np.full(10, .1), np.zeros(10)]
    assert "NAMED_EXPOSURE" in constraint_violations(extreme, benchmark, benchmark, exposure, [0])
    assert "TURNOVER" in constraint_violations(extreme, benchmark, benchmark, exposure, [0])


def test_nonfinite_is_never_accepted():
    assert constraint_violations([np.nan], [1], [1], [[1]], [0]) == ["NONFINITE_WEIGHTS"]


def test_missing_exposure_cannot_compare_as_false_and_pass():
    assert constraint_violations([1.], [1.], [1.], [[np.nan]], [0.]) == ["INVALID_OR_MISSING_CONSTRAINT_INPUTS"]
    assert constraint_violations([1.], [1.], [1.], [[1.]], [-.1]) == ["INVALID_OR_MISSING_CONSTRAINT_INPUTS"]
