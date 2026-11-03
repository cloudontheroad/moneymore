"""Frozen conservative market-order caps, including terminal liquidation."""


def market_order_cap(symbol):
    return 50000 if symbol.startswith(('688','689')) else 1000000


def liquidation_chunks(symbol,quantity):
    if int(quantity)!=quantity or quantity<0:raise ValueError('INVALID_LIQUIDATION_QUANTITY')
    cap=market_order_cap(symbol);quantity=int(quantity)
    full,tail=divmod(quantity,cap)
    # The final residual is a full remaining-position sale, not a partial odd-lot.
    return [cap]*full+([tail] if tail else [])
