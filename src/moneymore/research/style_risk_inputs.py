"""Risk descriptors are not profitability screening rules."""
import numpy as np
import pandas as pd


def signed_inverse(series):
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return 1 / values.where(values != 0)
