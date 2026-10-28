from __future__ import annotations

import os
from typing import Any

import pandas as pd


class TushareProvider:
    """Tushare Pro adapter.

    A client can be injected in tests; production construction reads the token
    from TUSHARE_TOKEN and never persists it.
    """

    def __init__(self, token: str | None = None, client: Any | None = None) -> None:
        if client is not None:
            self._client = client
            return
        resolved_token = token or os.getenv("TUSHARE_TOKEN")
        if not resolved_token:
            raise RuntimeError(
                "TUSHARE_TOKEN is not set. Copy .env.example to .env and add it locally."
            )
        import tushare as ts

        self._client = ts.pro_api(resolved_token)

    @property
    def name(self) -> str:
        return "tushare"

    def instruments(self) -> pd.DataFrame:
        fields = (
            "ts_code,symbol,name,area,industry,market,exchange,curr_type,"
            "list_status,list_date,delist_date,is_hs"
        )
        frames = [
            self._client.stock_basic(exchange="", list_status=status, fields=fields)
            for status in ("L", "D", "P", "G")
        ]
        nonempty = [frame for frame in frames if frame is not None and not frame.empty]
        if not nonempty:
            return pd.DataFrame(columns=fields.split(","))
        return pd.concat(nonempty, ignore_index=True).drop_duplicates("ts_code", keep="last")

    def trading_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        return self._client.trade_cal(exchange="SSE", start_date=start_date, end_date=end_date)

    def daily_bars(self, trade_date: str) -> pd.DataFrame:
        return self._client.daily(trade_date=trade_date)

    def adjustment_factors(self, trade_date: str) -> pd.DataFrame:
        return self._client.adj_factor(trade_date=trade_date)

    def historical_adjustments(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Bounded per-symbol official history for isolated research consistency checks."""
        return self._client.adj_factor(ts_code=ts_code, start_date=start_date, end_date=end_date)

    def etf_instruments(self) -> pd.DataFrame:
        frames = [self._client.etf_basic(list_status=status) for status in ("L", "D", "P")]
        nonempty = [frame for frame in frames if frame is not None and not frame.empty]
        if not nonempty:
            return pd.DataFrame(columns=["ts_code", "list_status", "list_date"])
        return pd.concat(nonempty, ignore_index=True).drop_duplicates("ts_code", keep="last")

    def etf_daily_bars(self, trade_date: str) -> pd.DataFrame:
        return self._client.fund_daily(trade_date=trade_date)

    def etf_adjustment_factors(self, trade_date: str) -> pd.DataFrame:
        return self._client.fund_adj(trade_date=trade_date)

    def index_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self._client.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)

    def stock_limits(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self._client.stk_limit(ts_code=ts_code, start_date=start_date, end_date=end_date)

    def suspensions(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self._client.suspend_d(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            suspend_type="S",
        )

    def daily_basic(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self._client.daily_basic(ts_code=ts_code, start_date=start_date, end_date=end_date)

    def dividends(self, ts_code: str) -> pd.DataFrame:
        return self._client.dividend(ts_code=ts_code)

    def financial_indicators(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self._client.fina_indicator(
            ts_code=ts_code, start_date=start_date, end_date=end_date
        )

    def income_statement(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Keep report types and actual announcement dates for PIT income diagnostics."""
        return self._client.income(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields="ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,end_type,total_revenue,revenue,update_flag",
        )

    def financial_indicators_revisioned(
        self, ts_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Research-core fields with explicit version flag; legacy broad endpoint unchanged."""
        return self._client.fina_indicator(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields="ts_code,ann_date,end_date,roe,q_sales_yoy,ocfps,or_yoy,q_gr_yoy,update_flag",
        )
