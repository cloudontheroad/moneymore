"""Conservative reference-quote gates using the actual experiment cost setting."""
import numpy as np


def quote_permissions(prices,halted,limits,slippage_bps):
    slip=slippage_bps/10000
    buy={s:bool(np.isfinite(p) and p>0 and s not in halted and p*(1+slip)<limits.up_limit.get(s,np.nan)-1e-8) for s,p in prices.items()}
    sell={s:bool(np.isfinite(p) and p>0 and s not in halted and p*(1-slip)>limits.down_limit.get(s,np.nan)+1e-8) for s,p in prices.items()}
    return buy,sell
