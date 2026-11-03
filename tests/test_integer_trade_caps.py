import numpy as np

from moneymore.research.integer_execution import redundant_trade_caps


def test_tighter_caps_preserve_every_original_feasible_quantity():
    # Enumerate both retain-odd choices, locked rights, T+1 and order caps.
    for current in [0, 100, 125, 799]:
      for available in [0, current // 2, current]:
       for locked in [0, 25, 100]:
        for price in [1., 37., 1347.]:
         for equity in [10000., 100000.]:
          for cap in [200, 50000]:
            buy_cap, sell_cap = redundant_trade_caps(
                np.array([price]), np.array([current]), np.array([available]),
                np.array([locked]), equity, .15, 100, np.array([cap]))
            for lots in range(int(.15*equity/price/100)+1):
             for retain in [0, 1]:
                final = 100*lots+(current % 100)*retain
                if current % 100 and lots and not retain:
                    continue  # Original odd-position constraint.
                if price*(final+locked) > .15*equity+1e-9:
                    continue
                buy, sell = max(0, final-current), max(0, current-final)
                if buy > cap or sell > min(available, cap):
                    continue
                assert buy <= buy_cap[0]+1e-9
                assert sell <= sell_cap[0]+1e-9
