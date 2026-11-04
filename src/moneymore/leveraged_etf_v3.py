"""Independent $100k forward paper account for the experimental TQQQ v3."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from .leveraged_etf import Allocation, LeveragedEtfPaper
from .tqqq_v3 import V3Parameters, raw_targets

V3_ACCOUNT_ID = "tqqq_momentum_dip_v3_paper"
V3_STRATEGY_ID = "cycle_momentum_dip_v3_experimental"
V3_INITIAL_CASH = 100_000.0


class LeveragedEtfV3Paper(LeveragedEtfPaper):
    """WildWest-inspired TQQQ/cash experiment; never touches the v2 ledger."""

    def __init__(self, database: str | Path, cache_dir: str | Path) -> None:
        super().__init__(
            database,
            cache_dir,
            account_id=V3_ACCOUNT_ID,
            strategy_id=V3_STRATEGY_ID,
            initial_cash=V3_INITIAL_CASH,
        )

    @staticmethod
    def allocation(
        data: dict[str, pd.DataFrame], index: int = -1, previous_tqqq: float | None = None
    ) -> Allocation:
        position = index if index >= 0 else len(data["QQQ"]) + index
        sliced = {symbol: frame.iloc[: position + 1].copy() for symbol, frame in data.items()}
        dates = pd.to_datetime(sliced["QQQ"]["trade_date"])

        def series(symbol: str) -> pd.Series:
            return pd.Series(sliced[symbol]["close"].astype(float).to_numpy(), index=dates)

        qqq = series("QQQ")
        tqqq_price = series("TQQQ")
        params = V3Parameters()
        decision = raw_targets(
            qqq,
            tqqq_price,
            series("^VIX"),
            series("^VIX3M"),
            params,
        ).iloc[-1]
        desired = float(decision.raw_target)
        target = (
            desired
            if previous_tqqq is None or abs(desired - previous_tqqq) > params.rebalance_band
            else previous_tqqq
        )
        regime = str(decision.regime)
        if bool(decision.dip_buy):
            regime = "BEAR_DIP_BUY"
        elif regime == "BEAR":
            regime = "BEAR_DEFENSIVE"
        returns = qqq.pct_change()
        annualized_vol = float(returns.rolling(20).std(ddof=1).iloc[-1] * math.sqrt(252))
        ma250 = float(qqq.rolling(250).mean().iloc[-1])
        price = float(qqq.iloc[-1])
        vix = series("^VIX")
        return Allocation(
            model_version=V3_STRATEGY_ID,
            signal_date=str(sliced["QQQ"].iloc[-1]["trade_date"]),
            regime=regime,
            tqqq=round(target, 6),
            sqqq=0.0,
            cash=round(1.0 - target, 6),
            score=round(float(decision.momentum) * 100.0, 4),
            qqq_close=price,
            ma180=ma250,
            vix=float(vix.iloc[-1]),
            vix_sma20=float(vix.rolling(20).mean().iloc[-1]),
            annualized_volatility=annualized_vol,
            macd_histogram=0.0,
            ma_slope=float(qqq.rolling(250).mean().pct_change(20).iloc[-1] * 100.0),
            base_tqqq=round(desired, 6),
            estimated_position_volatility=round(3.0 * annualized_vol * target, 6),
            risk_scalar=1.0,
            no_trade_band=params.rebalance_band,
            risk_on_step=0.0,
            fear_rank=0.0,
            rate_rank=0.0,
            term_structure=0.0,
            rate_change_60d=0.0,
            ma250=ma250,
            distance_from_ma250=price / ma250 - 1.0,
            crash_guard_active=False,
            momentum=float(decision.momentum),
            relative_volatility=float(decision.relative_volatility),
            rsi=float(decision.rsi),
            dip_buy=bool(decision.dip_buy),
        )
