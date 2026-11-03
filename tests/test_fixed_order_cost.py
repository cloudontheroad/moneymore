import pytest
from moneymore.research.fixed_order_cost import counterfactual_cash_delta


@pytest.mark.parametrize('side',['BUY','SELL'])
def test_lower_cost_same_orders_preserves_more_cash_even_with_taxes(side):
    p={'commission_rate':.00015,'minimum_commission':5.,'transfer_fee_rate':.00001,'stamp_duty_rate':.0005}
    ref=10.;quantity=1000;price=ref*(1.001 if side=='BUY' else .999)
    notional=price*quantity;fee=max(5.,notional*.00015)+notional*.00001+(notional*.0005 if side=='SELL' else 0.)
    fill={'side':side,'quantity':quantity,'price':price,'fee':fee}
    assert abs(counterfactual_cash_delta(fill,ref,p,10))<1e-10
    assert counterfactual_cash_delta(fill,ref,p,2)>0
