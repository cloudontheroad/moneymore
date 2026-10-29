"""Shared frozen-score -> portfolio target adapter, not an order submitter.

The caller must validate publication evidence and apply the existing risk and
integer-execution layers. No model fitting, account creation or clock access.
"""

from copy import deepcopy

import pandas as pd

from .baseline_v2_daily_readiness import classify_held_inputs, construct_daily_targets
from .baseline_v2_daily_scores import score_pair


class FrozenSignalPlanner:
    def __init__(self, contract):
        if (
            contract.get("native_model") != "qlib.contrib.model.linear.LinearModel"
            or contract.get("score_semantics")
            != "ONCE_NORMALIZED_DIRECTION_ADJUSTED_FEATURES_DOT_COEFFICIENTS_NOT_EXPECTED_RETURN"
            or contract.get("real_orders_enabled") is not False
        ):
            raise ValueError("UNSUPPORTED_FROZEN_SIGNAL_CONTRACT")
        self.policy = deepcopy(contract["factor_policy"])
        self.weights = {
            "fixed_weights": self.policy["factors"],
            "diagnostic_coefficients_not_released": deepcopy(contract["coefficients"]),
        }

    def targets(self, raw, holdings, prices, signal_date, execution_date, *, universe_members):
        if raw.empty or not pd.to_datetime(raw.date).eq(pd.Timestamp(signal_date)).all():
            raise ValueError("EXACT_SIGNAL_CROSS_SECTION_REQUIRED")
        classification = classify_held_inputs(
            holdings, raw.symbol, universe_members=universe_members
        )
        if classification["input_faults"]:
            return {
                "status": "HOLD_AND_ALERT",
                "target_weights": {},
                "clear_previous_plan": True,
                "real_orders_enabled": False,
                **classification,
            }
        prices = prices.copy()
        prices.index = pd.to_datetime(prices.index)
        if prices.index.has_duplicates or prices.columns.has_duplicates:
            raise ValueError("AMBIGUOUS_PRICE_HISTORY")
        prices = prices.sort_index()
        scores = score_pair(raw, self.policy, self.weights)
        scores["score"] = scores.score_learned
        selected, weights, ranking = construct_daily_targets(
            scores,
            holdings,
            prices,
            signal_date,
            execution_date,
            self.policy,
            universe_members=universe_members,
        )
        return {
            "status": "TARGET_READY_NOT_ORDER_SUBMISSION",
            "selected": selected,
            "target_weights": weights,
            "clear_previous_plan": False,
            "ranked_symbols": ranking.symbol.tolist(),
            "real_orders_enabled": False,
            **classification,
        }
