"""Resolve duplicate financial payloads without null-overwrite or arbitrary conflict wins."""

from __future__ import annotations

import pandas as pd

KEYS = ["ts_code", "ann_date", "end_date"]
FIELDS = ["roe", "q_sales_yoy"]


def reconcile_indicator_payload(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "_payload_capture" in frame:
        latest = frame.groupby(KEYS, dropna=False)["_payload_capture"].transform("max")
        frame = frame.loc[frame._payload_capture == latest].copy()
    # Explicitly requested revision metadata: use current-version rows when present.
    # If multiple current rows disagree, still block; never pick an arbitrary row.
    if "update_flag" in frame:
        current = pd.to_numeric(frame.update_flag, errors="coerce").eq(1)
        has_current = current.groupby([frame[k] for k in KEYS], dropna=False).transform("any")
        frame = frame.loc[~has_current | current].copy()
    indexed = frame.drop_duplicates(KEYS, keep="last").set_index(KEYS)
    grouped = frame.groupby(KEYS, dropna=False)
    for field in FIELDS:
        numeric = pd.to_numeric(frame[field], errors="coerce").replace(
            [float("inf"), -float("inf")], float("nan")
        )
        values = frame[KEYS].copy()
        values[field] = numeric
        group = values.groupby(KEYS, dropna=False)[field]
        conflict = group.nunique() > 1
        indexed[field] = group.first().where(~conflict)
        indexed[f"conflict__{field}"] = conflict
    indexed["payload_duplicate_rows"] = grouped.size() - 1
    return indexed.reset_index()


def derive_quarterly_revenue_growth(
    income: pd.DataFrame, as_of: str, *, allow_adjusted: bool = False, single_quarter: bool = False
) -> pd.DataFrame:
    """Revenue (not total_revenue), consolidated cumulative type 1, four-point PIT difference."""
    rows = []
    data = income.copy()
    allowed = (
        (["2", "3"] if allow_adjusted else ["2"])
        if single_quarter
        else (["1", "4"] if allow_adjusted else ["1"])
    )
    data = data.loc[data.report_type.astype(str).isin(allowed)].copy()
    ann = pd.to_datetime(data.ann_date, format="%Y%m%d", errors="coerce")
    actual = pd.to_datetime(data.f_ann_date, format="%Y%m%d", errors="coerce")
    data["publication"] = pd.concat([ann, actual], axis=1).max(axis=1)
    data["revenue"] = pd.to_numeric(data.revenue, errors="coerce")
    data = data.loc[
        ann.notna()
        & (data.publication + pd.offsets.Day(1) <= pd.Timestamp(as_of))
        & data.revenue.notna()
    ].copy()
    for symbol, frame in data.groupby("ts_code"):
        # One observation per period, latest disclosed revision, conflict-safe.
        available = {}
        for period, group in frame.groupby("end_date"):
            group = group.loc[group.publication == group.publication.max()]
            adjusted_type = "3" if single_quarter else "4"
            if allow_adjusted and group.report_type.astype(str).eq(adjusted_type).any():
                group = group.loc[group.report_type.astype(str).eq(adjusted_type)]
            if "update_flag" in group and group.update_flag.astype(str).eq("1").any():
                group = group.loc[group.update_flag.astype(str).eq("1")]
            if group.revenue.nunique() != 1 or group.comp_type.astype(str).nunique() != 1:
                available[str(period)] = None
            else:
                available[str(period)] = group.iloc[0]
        for period in sorted(available, reverse=True):
            if period[4:] not in {"0331", "0630", "0930", "1231"}:
                continue
            year = int(period[:4])
            quarter = {"0331": 1, "0630": 2, "0930": 3, "1231": 4}[period[4:]]
            previous_ends = {2: "0331", 3: "0630", 4: "0930"}
            terms = [period, f"{year - 1}{period[4:]}"]
            if quarter > 1 and not single_quarter:
                terms += [f"{year}{previous_ends[quarter]}", f"{year - 1}{previous_ends[quarter]}"]
            sources = [available.get(term) for term in terms]
            if any(value is None for value in sources):
                rows.append(
                    {
                        "ts_code": symbol,
                        "end_date": period,
                        "status": "MISSING_OR_CONFLICTING_COMPONENT",
                        "q_sales_yoy": None,
                        "source_periods": terms,
                    }
                )
                continue
            if len({str(value.comp_type) for value in sources}) != 1:
                rows.append(
                    {
                        "ts_code": symbol,
                        "end_date": period,
                        "status": "COMPANY_TYPE_MISMATCH",
                        "q_sales_yoy": None,
                        "source_periods": terms,
                    }
                )
                continue
            current = float(sources[0].revenue) - (
                float(sources[2].revenue) if quarter > 1 and not single_quarter else 0
            )
            prior = float(sources[1].revenue) - (
                float(sources[3].revenue) if quarter > 1 and not single_quarter else 0
            )
            if prior <= 0:
                rows.append(
                    {
                        "ts_code": symbol,
                        "end_date": period,
                        "status": "NONPOSITIVE_COMPARISON_BASE",
                        "q_sales_yoy": None,
                        "source_periods": terms,
                    }
                )
                continue
            publication = max(value.publication for value in sources)
            rows.append(
                {
                    "ts_code": symbol,
                    "end_date": period,
                    "ann_date": publication.strftime("%Y%m%d"),
                    "q_sales_yoy": (current / prior - 1) * 100,
                    "status": "DERIVED_FROM_SINGLE_QUARTER_REVENUE"
                    if single_quarter
                    else "DERIVED_FROM_CONSOLIDATED_REVENUE",
                    "quarter_revenue": current,
                    "prior_year_quarter_revenue": prior,
                    "source_periods": terms,
                    "source_publications": [
                        value.publication.strftime("%Y%m%d") for value in sources
                    ],
                    "source_revenues": [float(value.revenue) for value in sources],
                    "source_report_types": [str(value.report_type) for value in sources],
                    "source_field": "revenue",
                }
            )
    return pd.DataFrame(rows)
