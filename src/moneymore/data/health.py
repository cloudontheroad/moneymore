from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .store import ParquetStore


@dataclass(frozen=True)
class QualityCheck:
    code: str
    category: str
    status: str
    message: str
    affected_count: int = 0
    affected_symbols: str = ""


@dataclass(frozen=True)
class DataHealthReport:
    trade_date: str
    status: str
    checks: list[dict[str, object]]
    target_count: int
    blocking_count: int
    warning_count: int


def run_data_health_checks(
    store: ParquetStore,
    trade_date: str,
    symbols: set[str],
    disclosure_date: str | None = None,
    score_table: str = "sector_model_scores",
) -> DataHealthReport:
    ordered = sorted(symbols)
    checks: list[QualityCheck] = []
    try:
        calendar = store.read("trade_calendar")
    except FileNotFoundError:
        calendar = pd.DataFrame(columns=["cal_date", "is_open"])
    current_calendar = calendar.loc[
        calendar["cal_date"].astype(str) == trade_date
    ]
    checks.append(
        QualityCheck(
            "OFFICIAL_CALENDAR_COVERAGE",
            "calendar",
            "PASS" if len(current_calendar) == 1 else "BLOCK",
            (
                "正式交易日历已覆盖当日"
                if len(current_calendar) == 1
                else "正式交易日历未覆盖当日，禁止使用工作日估算交易"
            ),
            0 if len(current_calendar) == 1 else 1,
            "" if len(current_calendar) == 1 else trade_date,
        )
    )
    open_dates = sorted(
        calendar.loc[calendar["is_open"].astype(str) == "1", "cal_date"].astype(str)
    )
    prior_dates = [date for date in open_dates if date < trade_date]
    previous_trade_date = prior_dates[-1] if prior_dates else trade_date

    daily = _read_symbols(store, "daily", ordered)
    checks.extend(_market_checks(daily, trade_date, ordered))
    adjustment = _read_symbols(store, "adj_factor", ordered)
    checks.extend(
        _freshness_and_duplicates(
            adjustment, "adj_factor", trade_date, ordered, "BLOCK"
        )
    )
    if not adjustment.empty:
        invalid = adjustment.loc[
            pd.to_numeric(adjustment["adj_factor"], errors="coerce").fillna(0) <= 0
        ]
        checks.append(
            _issue(
                "ADJ_FACTOR_VALUES",
                "adjustment",
                "BLOCK" if not invalid.empty else "PASS",
                "复权因子必须为正数",
                invalid["ts_code"].astype(str).unique().tolist(),
            )
        )

    daily_basic = _read_symbols(store, "daily_basic", ordered)
    checks.extend(
        _freshness_and_duplicates(
            daily_basic,
            "daily_basic",
            previous_trade_date,
            ordered,
            "BLOCK",
            minimum_date=True,
        )
    )
    financial = _read_symbols(store, "fina_indicator", ordered)
    qmt_financial = _read_symbols(store, "qmt_financial_indicator", ordered)
    covered_financial = (
        set(financial["ts_code"].astype(str)) if not financial.empty else set()
    ) | (
        set(qmt_financial["ts_code"].astype(str)) if not qmt_financial.empty else set()
    )
    checks.append(
        _coverage_check(
            "FINANCIAL_COVERAGE", "fundamental", ordered, covered_financial, "BLOCK"
        )
    )
    if not financial.empty:
        null_ann = financial.loc[financial["ann_date"].isna()]
        checks.append(
            _issue(
                "FINANCIAL_ANN_DATE",
                "fundamental",
                "BLOCK" if not null_ann.empty else "PASS",
                "财务数据公告日期必须完整",
                null_ann["ts_code"].astype(str).unique().tolist(),
            )
        )
    if not qmt_financial.empty:
        null_qmt_ann = qmt_financial.loc[qmt_financial["ann_date"].isna()]
        checks.append(
            _issue(
                "QMT_FINANCIAL_ANN_DATE",
                "fundamental",
                "BLOCK" if not null_qmt_ann.empty else "PASS",
                "QMT财务数据公告日期必须完整",
                null_qmt_ann["ts_code"].astype(str).unique().tolist(),
            )
        )

    dividends = _read_symbols(store, "dividend", ordered)
    covered_dividend = set(dividends["ts_code"].astype(str)) if not dividends.empty else set()
    checks.append(
        _coverage_check(
            "DIVIDEND_COVERAGE", "corporate_action", ordered, covered_dividend, "WARN"
        )
    )
    checks.extend(
        _duplicate_check(
            dividends,
            "dividend",
            ["ts_code", "end_date", "ann_date", "div_proc"],
        )
    )

    try:
        scores = store.read(score_table)
        score_symbols = set(
            scores.loc[
                scores["date"].astype(str).str.replace("-", "").str[:8] == trade_date,
                "symbol",
            ].astype(str)
        )
        non_bank = set(ordered) if score_table=='global_factor_recommendation' else {symbol for symbol in ordered if symbol in set(scores["symbol"].astype(str))}
        checks.append(
            _coverage_check(
                "GLOBAL_SCORE_FRESHNESS" if score_table=='global_factor_recommendation' else "SECTOR_SCORE_FRESHNESS",
                "research",
                sorted(non_bank),
                score_symbols,
                "BLOCK",
            )
        )
    except FileNotFoundError:
        checks.append(
            QualityCheck(
                "SECTOR_SCORE_FRESHNESS",
                "research",
                "BLOCK",
                "行业模型评分表不存在",
                len(ordered),
                ",".join(ordered),
            )
        )

    if disclosure_date:
        age = (pd.Timestamp(trade_date) - pd.Timestamp(disclosure_date)).days
        checks.append(
            QualityCheck(
                "ETF_DISCLOSURE_AGE",
                "universe",
                "WARN" if age > 120 else "PASS",
                f"ETF成分披露距当前交易日{age}天",
            )
        )

    records = [asdict(check) for check in checks]
    blocking = sum(check.status == "BLOCK" for check in checks)
    warnings = sum(check.status == "WARN" for check in checks)
    status = "BLOCKED" if blocking else "WARNING" if warnings else "HEALTHY"
    report = DataHealthReport(
        trade_date=trade_date,
        status=status,
        checks=records,
        target_count=len(ordered),
        blocking_count=blocking,
        warning_count=warnings,
    )
    _persist_health(store, report)
    return report


def _read_symbols(
    store: ParquetStore, table: str, symbols: list[str]
) -> pd.DataFrame:
    try:
        return store.read(table, filters=[("ts_code", "in", symbols)])
    except FileNotFoundError:
        return pd.DataFrame()


def _market_checks(
    frame: pd.DataFrame, trade_date: str, symbols: list[str]
) -> list[QualityCheck]:
    checks = _freshness_and_duplicates(frame, "daily", trade_date, symbols, "BLOCK")
    if frame.empty:
        return checks
    current = frame.loc[frame["trade_date"].astype(str) == trade_date]
    prices = current[["open", "high", "low", "close"]].apply(
        pd.to_numeric, errors="coerce"
    )
    invalid = current.loc[
        prices.isna().any(axis=1)
        | (prices <= 0).any(axis=1)
        | (prices["high"] < prices[["open", "close", "low"]].max(axis=1))
        | (prices["low"] > prices[["open", "close", "high"]].min(axis=1))
    ]
    checks.append(
        _issue(
            "DAILY_OHLC_VALUES",
            "market",
            "BLOCK" if not invalid.empty else "PASS",
            "当日OHLC价格必须为正且高低价关系有效",
            invalid["ts_code"].astype(str).unique().tolist(),
        )
    )
    return checks


def _freshness_and_duplicates(
    frame: pd.DataFrame,
    table: str,
    required_date: str,
    symbols: list[str],
    severity: str,
    minimum_date: bool = False,
) -> list[QualityCheck]:
    if frame.empty:
        return [
            QualityCheck(
                f"{table.upper()}_FRESHNESS",
                "freshness",
                severity,
                f"{table}数据不存在",
                len(symbols),
                ",".join(symbols),
            )
        ]
    date_column = "trade_date"
    latest = (
        frame.assign(_date=frame[date_column].astype(str))
        .groupby("ts_code")["_date"]
        .max()
    )
    missing = [
        symbol
        for symbol in symbols
        if symbol not in latest
        or (
            latest[symbol] < required_date
            if minimum_date
            else latest[symbol] != required_date
        )
    ]
    checks = [
        _issue(
            f"{table.upper()}_FRESHNESS",
            "freshness",
            severity if missing else "PASS",
            f"{table}最新日期要求{'不早于' if minimum_date else '等于'}{required_date}",
            missing,
        )
    ]
    checks.extend(_duplicate_check(frame, table, ["ts_code", date_column]))
    return checks


def _duplicate_check(
    frame: pd.DataFrame, table: str, keys: list[str]
) -> list[QualityCheck]:
    if frame.empty or not set(keys).issubset(frame.columns):
        return []
    duplicates = frame.loc[frame.duplicated(keys, keep=False)]
    return [
        _issue(
            f"{table.upper()}_DUPLICATES",
            "integrity",
            "BLOCK" if not duplicates.empty else "PASS",
            f"{table}主键不得重复",
            duplicates["ts_code"].astype(str).unique().tolist(),
        )
    ]


def _coverage_check(
    code: str,
    category: str,
    expected: list[str],
    actual: set[str],
    severity: str,
) -> QualityCheck:
    missing = sorted(set(expected) - actual)
    return _issue(
        code,
        category,
        severity if missing else "PASS",
        f"覆盖{len(expected) - len(missing)}/{len(expected)}只目标股票",
        missing,
    )


def _issue(
    code: str,
    category: str,
    status: str,
    message: str,
    symbols: list[str],
) -> QualityCheck:
    return QualityCheck(
        code,
        category,
        status,
        message,
        len(symbols),
        ",".join(sorted(symbols)),
    )


def _persist_health(store: ParquetStore, report: DataHealthReport) -> None:
    summary = pd.DataFrame(
        [
            {
                "trade_date": report.trade_date,
                "status": report.status,
                "target_count": report.target_count,
                "blocking_count": report.blocking_count,
                "warning_count": report.warning_count,
            }
        ]
    )
    store.merge_curated("data_health_daily", [summary], ["trade_date"])
    checks = pd.DataFrame(report.checks).assign(trade_date=report.trade_date)
    store.merge_curated("data_health_checks", [checks], ["trade_date", "code"])
