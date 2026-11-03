import pytest
from moneymore.research.order_chunks import liquidation_chunks,market_order_cap


@pytest.mark.parametrize('symbol,quantity',[('688001.SH',50100),('688001.SH',100025),('600000.SH',1000050),('600000.SH',0)])
def test_terminal_chunks_preserve_total_and_single_order_caps(symbol,quantity):
    chunks=liquidation_chunks(symbol,quantity)
    assert sum(chunks)==quantity and all(0<q<=market_order_cap(symbol) for q in chunks)
    assert all(q%100==0 for q in chunks[:-1])


def test_invalid_liquidation_does_not_produce_order():
    with pytest.raises(ValueError):liquidation_chunks('600000.SH',-1)
    with pytest.raises(ValueError):liquidation_chunks('600000.SH',100.5)
