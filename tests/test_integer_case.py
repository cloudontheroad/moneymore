import json
import pandas as pd
from moneymore.config import BacktestConfig
from moneymore.research.integer_case import encode_case,decode_case


def test_frozen_case_roundtrip_preserves_problem_and_identity():
    cfg=BacktestConfig(initial_cash=10000,commission_rate=.00015,minimum_commission=5,
        stamp_duty_rate=.0005,transfer_fee_rate=.00001,slippage_bps=10,lot_size=100,
        max_position_weight=.15,max_gross_exposure=1,max_drawdown=.2)
    controls=pd.DataFrame({'size':[1.,-1.]},index=['A','B'])
    target=pd.Series({'A':.1,'B':.1});prices=pd.Series({'A':10.,'B':20.});limits=pd.Series({'size':.1})
    encoded=encode_case(target,prices,{'A':100},{'A':100},9000.,10000.,cfg,controls,limits,
                        benchmark_weights=pd.Series({'A':.5,'B':.5}),can_sell={'A':True},style_controls_prestandardized=True)
    args,kwargs=decode_case(json.loads(json.dumps(encoded,allow_nan=False)))
    pd.testing.assert_series_equal(args[0],target)
    pd.testing.assert_series_equal(args[1],prices)
    pd.testing.assert_frame_equal(args[7],controls)
    assert args[2:6]==({'A':100},{'A':100},9000.,10000.)
    assert args[6]==cfg and kwargs['benchmark_weights']=={'A':.5,'B':.5}
