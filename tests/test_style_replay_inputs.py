from moneymore.research.style_replay_inputs import prospective_capture


def test_constant_scores_are_not_misreported_as_missing_or_optimized():
    import numpy as np
    import pandas as pd
    import pytest
    from moneymore.research.style_replay_inputs import optimize_daily
    for count in [1, 20]:
        index=pd.Index([str(i) for i in range(count)])
        with pytest.raises(ValueError,match='NONINFORMATIVE_SCORE_SCALE'):
            optimize_daily(pd.Series(1.,index=index),pd.DataFrame(index=index),None,
                           None,None,None,None,None,pd.Series(dtype=float),candidate_symbols=index)


def test_contradictory_style_and_position_limits_do_not_silently_relax():
    import numpy as np
    import pandas as pd
    import pytest
    from moneymore.research.style_replay_inputs import optimize_daily
    index=pd.Index([str(i) for i in range(20)])
    controls=pd.DataFrame({'concentrated':[1.]+[0.]*19},index=index)
    wb=pd.Series([.9]+[.1/19]*19,index=index)
    risk=(np.ones((20,1))*.001,np.array([[.0001]]),np.full(20,.0001))
    with pytest.raises(ValueError,match='CONTINUOUS_SOLVER_INFEASIBLE'):
        optimize_daily(pd.Series(np.arange(20.),index=index),pd.DataFrame(index=index),controls,
                       *risk,wb,pd.Series({'concentrated':.01}),pd.Series(dtype=float))


def test_capture_must_be_after_signal_close_before_execution_open():
    assert prospective_capture("20261016", "2026-10-16T10:35:53+00:00", "20261019")
    assert not prospective_capture("20261016", "2026-10-19T02:00:00+00:00", "20261019")
    assert not prospective_capture("20261016", "2026-10-16T06:00:00+00:00", "20261019")
    assert not prospective_capture("20261016", None, "20261019")


def test_turnover_uses_actual_holdings_and_is_absent_only_for_cash_entry():
    import numpy as np
    import pandas as pd
    from moneymore.research.style_replay_inputs import optimize_daily
    index = pd.Index([str(i) for i in range(20)])
    scores = pd.Series(np.arange(20.), index=index)
    selected = pd.DataFrame(index=index)
    controls = pd.DataFrame({"control": np.zeros(20)}, index=index)
    wb = pd.Series(.05, index=index)
    limits = pd.Series({"control": .1})
    risk = (np.ones((20, 1))*.001, np.array([[.0001]]), np.full(20, .0001))
    cold = optimize_daily(scores, selected, controls, *risk, wb, limits, pd.Series(dtype=float))
    assert abs(cold-wb).sum() > .20
    # Actual holdings deliberately differ from benchmark.
    actual = pd.Series([.10]*10+[0.]*10, index=index)
    rebalance = optimize_daily(scores, selected, controls, *risk, wb, limits, actual*1_000_000)
    assert abs(rebalance-actual).sum() <= .20+1e-6


def test_statistical_risk_contributions_reconcile_not_named_factor_attribution():
    import numpy as np
    import pandas as pd
    from moneymore.research.style_replay_inputs import measured_style
    controls = pd.DataFrame({"size": [-1., 1.]}, index=["A", "B"])
    wb = pd.Series({"A": .5, "B": .5})
    result = measured_style(pd.Series({"A": .8, "B": .2}), controls,
        np.array([[.5], [1.]]), np.array([[.01]]), np.array([.02, .03]), wb)
    variance = result["statistical_factor_variance"]+result["specific_variance"]
    assert abs(sum(result["asset_active_variance_contributions"].values())-variance) < 1e-12
    assert abs(result["active_tracking_volatility"]**2-variance*252) < 1e-12
    assert result["not_named_factor_return_attribution"]


def test_asymmetric_buffer_is_enforced_and_cannot_relax_original_limits():
    import numpy as np
    import pandas as pd
    import pytest
    from moneymore.research.style_replay_inputs import optimize_daily
    index=pd.Index([str(i) for i in range(20)])
    controls=pd.DataFrame({"style":np.linspace(-1,1,20)},index=index)
    scores=controls["style"]
    risk=(np.ones((20,1))*.001,np.array([[.0001]]),np.full(20,.0001))
    wb=pd.Series(.05,index=index);limits=pd.Series({"style":.1})
    args=(scores,pd.DataFrame(index=index),controls,*risk,wb,limits,pd.Series(dtype=float))
    target=optimize_daily(*args,lower_limits=pd.Series({"style":-.02}),upper_limits=pd.Series({"style":.05}))
    exposure=float((target-wb) @ controls["style"])
    assert -.020001<=exposure<=.050001
    with pytest.raises(ValueError,match="RELAXED"):
        optimize_daily(*args,lower_limits=pd.Series({"style":-.2}),upper_limits=pd.Series({"style":.05}))


def test_explicit_quality_contract_does_not_force_missing_holdings_or_override_legacy():
    import numpy as np
    import pandas as pd
    import pytest
    from moneymore.research.style_replay_inputs import optimize_daily
    index=pd.Index([str(i) for i in range(20)])
    controls=pd.DataFrame({"unknown_x":[1.]*10+[0.]*10},index=index)
    risk=(np.ones((20,1))*.001,np.array([[.0001]]),np.full(20,.0001))
    wb=pd.Series(.05,index=index);limits=pd.Series({"unknown_x":.01})
    args=(pd.Series(np.arange(20.),index=index),pd.DataFrame(index=index),controls,*risk,wb,limits,pd.Series(dtype=float))
    old=optimize_daily(*args)
    assert float(old @ controls.unknown_x)>=.49-1e-6
    envelope={"unknown_x":{"upper":0.,"lower":0.}}
    new=optimize_daily(*args,research_quality_envelope=envelope)
    assert float(new @ controls.unknown_x)<.01
    with pytest.raises(ValueError,match="CANNOT_MIX"):
        optimize_daily(*args,research_quality_envelope=envelope,lower_limits=-limits,upper_limits=limits)
