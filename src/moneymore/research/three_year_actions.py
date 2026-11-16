"""Evidence-scoped source duplicate handling, only for the three-year replay."""

from __future__ import annotations

import numpy as np
import pandas as pd

from moneymore.research.replay_actions import official_replay_actions
from moneymore.research.reviewed_action_classes import select_reviewed_retail_actions


class AppendOnlyActionCache:
    """Parse each appended raw event once; reject any change to the prior prefix.

    This mirrors the replay engine's cumulative, deduplicated, chronological input.
    No future action is returned before its actual input row arrives.
    """

    def __init__(self):
        self.fingerprints = np.array([], dtype="uint64")
        self.actions = official_replay_actions(pd.DataFrame())

    def __call__(self, raw):
        if raw.empty:
            if len(self.fingerprints):
                raise ValueError("ACTION_CACHE_PREFIX_REMOVED")
            return self.actions.copy()
        canonical = pd.DataFrame(index=raw.index)
        for field in [
            "ts_code",
            "end_date",
            "ann_date",
            "record_date",
            "ex_date",
            "pay_date",
            "div_listdate",
            "imp_ann_date",
            "div_proc",
        ]:
            canonical[field] = (
                raw.get(field, pd.Series(index=raw.index, dtype=object)).fillna("").astype(str)
            )
        for field in ["cash_div_tax", "stk_div", "stk_bo_rate", "stk_co_rate"]:
            canonical[field] = pd.to_numeric(
                raw.get(field, pd.Series(index=raw.index, dtype=float)), errors="raise"
            ).astype("float64")
        hashes = pd.util.hash_pandas_object(canonical, index=False).to_numpy()
        n = len(self.fingerprints)
        if len(hashes) < n or not np.array_equal(hashes[:n], self.fingerprints):
            raise ValueError("ACTION_CACHE_PREFIX_CHANGED")
        if len(hashes) > n:
            added = official_replay_actions(raw.iloc[n:])
            combined = pd.concat([self.actions, added], ignore_index=True).drop_duplicates()
            if combined.action_key.duplicated().any():
                raise ValueError("CONFLICTING_IMPLEMENTED_ACTION_VERSIONS")
            self.actions = combined
            self.fingerprints = hashes
        return self.actions.copy()


def reviewed_actions(raw, retail_rules, aggregate_rules):
    if raw.empty:
        return raw.copy(), []
    result, audit = select_reviewed_retail_actions(raw, retail_rules)
    for rule in aggregate_rules:
        mask = (
            result.ts_code.eq(rule["symbol"])
            & result.record_date.eq(rule["record_date"])
            & result.div_proc.eq("实施")
        )
        rows = result.loc[mask]
        if rows.empty:
            continue
        observed = sorted(zip(rows.end_date.astype(str), rows.cash_div_tax.astype(float)))
        if observed != sorted((str(d), float(v)) for d, v in rule["expected_rows"]):
            raise ValueError("EVIDENCED_AGGREGATE_ROWS_CHANGED")
        for field in ["ex_date", "pay_date", "imp_ann_date"]:
            if not rows[field].eq(rule[field]).all():
                raise ValueError("EVIDENCED_AGGREGATE_DATES_CHANGED")
        if not np.isclose(rows.stk_div.astype(float), 0).all():
            raise ValueError("EVIDENCED_AGGREGATE_STOCK_CHANGED")
        selected = rows.end_date.eq(rule["keep_end_date"]) & rows.cash_div_tax.eq(
            rule["keep_cash_per_share"]
        )
        if selected.sum() != 1:
            raise ValueError("EVIDENCED_AGGREGATE_NOT_UNIQUE")
        result = result.drop(index=rows.index[~selected])
        audit.append({**rule, "excluded_rows": int((~selected).sum())})
    return result, audit
