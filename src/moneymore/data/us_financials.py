"""Point-in-time US company data shared by research consumers."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .store import ParquetStore

FILINGS = "us_sec_filings"
FACTS = "us_reported_financial_facts"
SPLITS = "us_observed_stock_splits"
FACT_CONTEXT = ["cik", "taxonomy", "tag", "unit", "period_start", "period_end", "duration_months"]
EXPANSION_TAGS = (
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
    "ResearchAndDevelopmentExpense",
    "ProceedsFromIssuanceOfCommonStock",
)


def cik_keys(values: pd.Series) -> pd.Series:
    return values.astype("string").str.zfill(10)


def decision_day(as_of: str | pd.Timestamp) -> pd.Timestamp:
    """Use the US exchange date and exclude disclosures from the decision day."""
    day = pd.Timestamp(as_of)
    if day.tzinfo is not None:
        day = day.tz_convert("America/New_York").tz_localize(None)
    return day.normalize()


def reported_facts_as_of(
    store: ParquetStore,
    as_of: str | pd.Timestamp,
    *,
    tags: Sequence[str],
    ciks: Sequence[str | int] | None = None,
    industries: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return the latest available filing vintage for each comparable observation."""
    filters = [("public_date", "<", decision_day(as_of)), ("tag", "in", list(tags) or [""])]
    if ciks is not None:
        filters.append(("cik", "in", [str(cik).zfill(10) for cik in ciks] or [""]))
    facts = store.read(FACTS, filters=filters)
    keys = [*FACT_CONTEXT, "period_precision"]
    versions = facts.groupby(keys, dropna=False, sort=False)
    latest = facts.loc[facts.public_date.eq(versions.public_date.transform("max"))]
    agreement = latest.groupby(keys, dropna=False, sort=False).value.transform("nunique").eq(1)
    selected = latest.loc[agreement].sort_values(["accepted", "accession", "source"], na_position="first")
    selected = selected.drop_duplicates(keys, keep="last")
    if industries is not None:
        selected = selected.loc[selected.industry.isin(industries)]
    return selected.reset_index(drop=True)


def observed_splits_as_of(
    store: ParquetStore, as_of: str | pd.Timestamp, *, ciks: Sequence[str | int]
) -> pd.DataFrame:
    """Corporate-action observations with independently established issuer identity."""
    return store.read(SPLITS, filters=[
        ("identity_verified", "=", True),
        ("known_date", "<", decision_day(as_of)),
        ("ex_date", "<=", decision_day(as_of)),
        ("cik", "in", [str(cik).zfill(10) for cik in ciks] or [""]),
    ])


def annual_expansion_as_of(
    store: ParquetStore,
    as_of: str | pd.Timestamp,
    *,
    ciks: Sequence[str | int] | None = None,
    industries: Sequence[str] | None = None,
    max_age_days: int = 550,
) -> pd.DataFrame:
    """Pair comparable, disclosed fiscal-year values with the previous fiscal year."""
    facts = reported_facts_as_of(store, as_of, tags=EXPANSION_TAGS, ciks=ciks)
    annual = facts.loc[
        facts.taxonomy.eq("us-gaap") & facts.unit.eq("USD") & facts.duration_months.eq(12)
        & facts.form.isin(["10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"])
    ].copy()
    keys = ["cik", "tag", "period_precision"]
    period_keys = [*keys, "period_end"]
    matching_period = annual.groupby(period_keys, sort=False).value.transform("nunique").eq(1)
    annual = annual.loc[matching_period].sort_values("public_date").drop_duplicates(
        period_keys, keep="last"
    )
    latest = annual.sort_values("period_end").drop_duplicates(keys, keep="last")
    age = (decision_day(as_of) - latest.period_end).dt.days
    latest = latest.loc[age.between(0, max_age_days)]
    if industries is not None:
        latest = latest.loc[latest.industry.isin(industries)]
    previous = annual[[*keys, "period_end", "public_date", "value"]].rename(columns={
        "period_end": "prior_period_end", "public_date": "prior_public_date", "value": "prior_value"
    })
    pairs = latest.merge(previous, on=keys)
    spacing = (pairs.period_end - pairs.prior_period_end).dt.days
    pairs = pairs.loc[spacing.between(330, 400)]
    pairs = pairs.assign(fiscal_month=pairs.period_end.dt.to_period("M"), exact=pairs.period_precision.eq("day"))
    pairs = pairs.sort_values(["fiscal_month", "exact", "period_end", "prior_period_end", "public_date"])
    pairs = pairs.drop_duplicates(["cik", "tag"], keep="last").drop(columns=["fiscal_month", "exact"])
    pairs["change"] = pairs.value - pairs.prior_value
    pairs["growth"] = pairs.change.div(pairs.prior_value.where(pairs.prior_value.gt(0)))
    return pairs.reset_index(drop=True)
