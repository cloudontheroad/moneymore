"""Evidence-bound research identities. Never infer price continuity from name similarity."""

from __future__ import annotations

import numpy as np
import pandas as pd


class SecurityIdentity:
    def __init__(self, changes):
        self.changes = [dict(row) for row in changes]
        seen = set()
        for row in changes:
            if not row.get("source") or row["old"] == row["new"] or {row["old"], row["new"]} & seen:
                raise ValueError("UNSUPPORTED_OR_AMBIGUOUS_IDENTITY_CHANGE")
            pd.to_datetime(row["effective"], format="%Y%m%d", errors="raise")
            seen.update([row["old"], row["new"]])

    def stable(self, symbol):
        for row in self.changes:
            if symbol in (row["old"], row["new"]):
                return row["old"]
        return symbol

    def tradable(self, symbol, date):
        stamp = pd.Timestamp(date).strftime("%Y%m%d")
        for row in self.changes:
            if symbol in (row["old"], row["new"]):
                return row["new"] if stamp >= row["effective"] else row["old"]
        return symbol

    def conflicts(self, frame, keys, fields):
        rows = []
        for change in self.changes:
            old = frame.loc[frame.ts_code.eq(change["old"])]
            new = frame.loc[frame.ts_code.eq(change["new"])]
            paired = old.merge(new, on=keys, suffixes=("_old", "_new"), validate="one_to_one")
            for field in fields:
                a, b = (
                    pd.to_numeric(paired[field + "_old"], errors="coerce"),
                    pd.to_numeric(paired[field + "_new"], errors="coerce"),
                )
                # Tight source-value agreement, not the looser modelling tolerance.
                mismatch = ~np.isclose(a, b, rtol=0, atol=1e-10, equal_nan=True)
                for idx in paired.index[mismatch]:
                    rows.append(
                        {
                            **{k: paired.loc[idx, k] for k in keys},
                            "security_id": change["old"],
                            "field": field,
                            "old_value": a.loc[idx],
                            "new_value": b.loc[idx],
                        }
                    )
        return pd.DataFrame(rows, columns=[*keys, "security_id", "field", "old_value", "new_value"])

    def market_rows(self, frame, date_column, fields, *, strict=True, record_keys=None):
        keys = record_keys or [date_column]
        conflicts = self.conflicts(frame, keys, fields)
        if strict and not conflicts.empty:
            raise ValueError("ALIAS_SOURCE_VALUE_CONFLICT")
        result = frame.copy()
        result["source_symbol"] = result.ts_code
        dates = pd.to_datetime(result[date_column], errors="raise").dt.strftime("%Y%m%d")
        expected = result.ts_code.copy()
        for row in self.changes:
            mask = result.ts_code.isin([row["old"], row["new"]])
            expected.loc[mask] = np.where(
                dates.loc[mask].ge(row["effective"]), row["new"], row["old"]
            )
        result = result.loc[result.ts_code.eq(expected)].copy()
        result["security_id"] = result.ts_code.map(self.stable)
        result["ts_code"] = result.security_id
        if result.duplicated(["ts_code", *keys]).any():
            raise ValueError("DUPLICATE_SECURITY_IDENTITY_DATE")
        return result

    def holdings(self, quantities, as_of):
        result = {}
        for symbol, quantity in quantities.items():
            if quantity == 0:
                continue
            stable = self.stable(symbol)
            if stable in result:
                raise ValueError("DUPLICATE_POSITION_ALIASES_REQUIRE_RECONCILIATION")
            result[stable] = {"quantity": quantity, "trading_symbol": self.tradable(symbol, as_of)}
        return result

    def industry_rows(self, frame):
        result = frame.copy()
        result["source_symbol"] = result.ts_code
        result["ts_code"] = result.ts_code.map(self.stable)
        return result  # Effective in/out intervals remain unchanged; no current-label backfill.

    def execution_targets(self, weights, execution_date):
        """Research order intent only; analytic IDs must not be submitted as stale tickers."""
        rows, seen = [], set()
        for symbol, weight in weights.items():
            stable = self.stable(symbol)
            if stable in seen:
                raise ValueError("DUPLICATE_TARGET_ALIASES_REQUIRE_RECONCILIATION")
            if not np.isfinite(weight) or weight < 0:
                raise ValueError("INVALID_IDENTITY_TARGET_WEIGHT")
            seen.add(stable)
            rows.append(
                {
                    "security_id": stable,
                    "trading_symbol": self.tradable(symbol, execution_date),
                    "target_weight": weight,
                }
            )
        return pd.DataFrame(rows, columns=["security_id", "trading_symbol", "target_weight"])
