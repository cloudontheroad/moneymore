"""Explicit issuer-evidenced retail action selection; never last-row-wins."""

import numpy as np
import pandas as pd


def select_reviewed_retail_actions(raw, evidence):
    result, audit = raw.copy(), []
    for rule in evidence:
        mask = (
            result.ts_code.eq(rule["symbol"])
            & result.end_date.eq(rule["end_date"])
            & result.record_date.eq(rule["record_date"])
            & result.div_proc.eq("实施")
        )
        rows = result.loc[mask]
        if rows.empty:
            continue
        amounts = set(pd.to_numeric(rows.cash_div_tax).tolist())
        if amounts != set(rule["observed_cash_classes"]):
            raise ValueError("REVIEWED_ACTION_CLASS_AMOUNTS_CHANGED")
        for field in ("ex_date", "pay_date", "imp_ann_date"):
            if not rows[field].eq(rule[field]).all():
                raise ValueError("REVIEWED_ACTION_CLASS_TIMELINE_CHANGED")
        if not np.isclose(pd.to_numeric(rows.stk_div), 0).all():
            raise ValueError("REVIEWED_ACTION_CLASS_STOCK_RATIO_CHANGED")
        selected = rows.cash_div_tax.eq(rule["retail_cash_per_share"])
        if selected.sum() != 1:
            raise ValueError("REVIEWED_ACTION_RETAIL_CLASS_NOT_UNIQUE")
        result = result.drop(index=rows.index[~selected])
        audit.append({**rule, "excluded_rows": int((~selected).sum())})
    return result, audit
