import pandas as pd
from moneymore.research.quote_permissions import quote_permissions


def test_2_and_10bps_use_distinct_real_price_limit_tests():
    limits=pd.DataFrame({'up_limit':[10.,10.],'down_limit':[9.,9.]},index=['UP','DOWN'])
    low=quote_permissions({'UP':9.995,'DOWN':9.005},set(),limits,2)
    high=quote_permissions({'UP':9.995,'DOWN':9.005},set(),limits,10)
    assert low[0]['UP'] and not high[0]['UP']
    assert low[1]['DOWN'] and not high[1]['DOWN']


def test_missing_gates_halts_and_bad_quotes_never_allow_trade():
    limits=pd.DataFrame({'up_limit':[10.],'down_limit':[9.]},index=['HALT'])
    buy,sell=quote_permissions({'HALT':9.5,'UNKNOWN':9.5,'ZERO':0.}, {'HALT'},limits,2)
    assert not any(buy.values()) and not any(sell.values())
