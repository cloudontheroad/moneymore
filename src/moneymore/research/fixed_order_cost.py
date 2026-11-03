"""Pure-cost counterfactual for the exact same actual order schedule, no refit."""
import numpy as np


def counterfactual_cash_delta(fill,reference,protocol,new_bps):
    side=fill['side'];quantity=fill['quantity']
    if side not in ['BUY','SELL'] or quantity<=0 or reference<=0:raise ValueError('INVALID_COUNTERFACTUAL_ORDER')
    price=reference*(1+(1 if side=='BUY' else -1)*new_bps/10000)
    notional=price*quantity
    fee=max(protocol['minimum_commission'],protocol['commission_rate']*notional)+protocol['transfer_fee_rate']*notional
    if side=='SELL':fee+=protocol['stamp_duty_rate']*notional
    direction=1 if side=='SELL' else -1
    delta=direction*(price-fill['price'])*quantity+fill['fee']-fee
    if not np.isfinite(delta):raise ValueError('NONFINITE_COUNTERFACTUAL_CASH')
    return float(delta)
