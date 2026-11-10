"""Publish the staged US SEC and corporate action data to MoneyMore's shared store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from moneymore.data.store import ParquetStore
from moneymore.data.us_financials import FACT_CONTEXT, FACTS, FILINGS, SPLITS, cik_keys


def publication_date(filings: pd.DataFrame) -> pd.Series:
    accepted = pd.to_datetime(filings.accepted).dt.normalize()
    filed = pd.to_datetime(filings.filed).dt.normalize()
    return pd.concat([accepted, filed], axis=1).max(axis=1)


def prepare_facts(frame: pd.DataFrame, filings: pd.DataFrame, source: str) -> pd.DataFrame:
    dates = filings[["accession", "cik", "industry", "industry_basis", "form", "filed", "accepted", "public_date"]]
    if source == "secdatabase":
        frame = frame.rename(columns={
            "start_date": "period_start", "end_date": "period_end", "period_month": "duration_months",
            "segment_label": "segments",
        })
        frame["duration_months"] = frame.duration_months.where(frame.period_start.notna(), 0)
        frame["period_precision"] = "day"
    else:
        frame = frame.rename(columns={"fact_date": "period_end", "uom": "unit"})
        frame["period_start"] = pd.NaT
        frame["duration_months"] = frame.qtrs * 3
        frame["period_precision"] = "month"
    frame["taxonomy"] = frame.version.str.split("/").str[0]
    frame = frame.loc[
        frame.segments.fillna("").eq("") & frame.taxonomy.isin(["us-gaap", "ifrs-full", "dei"])
    ]
    fields = [
        "accession", "tag", "unit", "period_start", "period_end", "duration_months",
        "period_precision", "taxonomy", "value",
    ]
    joined = frame[fields].merge(dates, on="accession", how="inner", validate="many_to_one")
    joined["period_start"] = pd.to_datetime(joined.period_start)
    joined["period_end"] = pd.to_datetime(joined.period_end)
    joined["duration_months"] = joined.duration_months.astype("Int64")
    joined["source"] = source
    return joined.loc[joined.period_end.le(joined.public_date)].copy()


def prepare_companyfacts(
    index: pd.DataFrame, facts: pd.DataFrame, filings: pd.DataFrame, source: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = index.loc[index.form.isin(["10-K", "10-Q", "20-F", "40-F", "10-K/A", "10-Q/A", "20-F/A", "40-F/A"])].copy()
    index["cik"] = cik_keys(index.cik)
    index["accepted"] = pd.to_datetime(index.accepted, utc=True).dt.tz_convert(
        "America/New_York"
    ).dt.tz_localize(None)
    index["period"] = pd.to_datetime(index.period)
    index["filed"] = pd.to_datetime(index.filed)
    index["public_date"] = publication_date(index)
    index = index.loc[index.public_date.notna() & index.period.le(index.public_date)]
    index = index[["accession", "cik", "name", "form", "period", "filed", "accepted", "public_date", "instance"]]
    history = filings[["cik", "industry", "public_date", "accepted"]].sort_values(["public_date", "accepted"])
    index = pd.merge_asof(
        index.sort_values("public_date"), history.drop(columns="accepted"),
        on="public_date", by="cik", allow_exact_matches=False,
    )
    index["industry_basis"] = "prior_sec_sic"
    index["source"] = source
    new = index.loc[~index.accession.isin(filings.accession)]
    filing_index = pd.concat([filings, new], ignore_index=True)

    facts["cik"] = cik_keys(facts.cik)
    for field in ("period_start", "period_end", "fact_filed"):
        facts[field] = pd.to_datetime(facts[field])
    days = (facts.period_end - facts.period_start).dt.days + 1
    facts["duration_months"] = days.div(30.4375).round().where(facts.period_start.notna(), 0).astype("Int64")
    facts["period_precision"] = "day"
    dates = filing_index[["accession", "cik", "industry", "industry_basis", "form", "filed", "accepted", "public_date"]]
    facts = facts.merge(dates, on=["accession", "cik", "form"], how="inner", validate="many_to_one")
    facts["public_date"] = facts[["public_date", "fact_filed"]].max(axis=1)
    facts["source"] = source
    columns = [*FACT_CONTEXT, "period_precision", "accession", "source", "value",
               "industry", "industry_basis", "form", "filed", "accepted", "public_date"]
    usable = facts.fact_filed.notna() & facts.period_end.le(facts.public_date) & facts.value.notna()
    return new, facts.loc[usable, columns]


def companyfacts_archive(raw: Path, filings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.read_json(raw / "sec_filing_index.jsonl.gz", lines=True).rename(columns={
        "accessionNumber": "accession", "filingDate": "filed", "reportDate": "period",
        "acceptanceDateTime": "accepted", "entity_name": "name", "primaryDocument": "instance",
    })
    facts = pd.read_json(raw / "sec_financial_observations.jsonl.gz", lines=True).rename(columns={
        "accn": "accession", "concept": "tag", "namespace": "taxonomy", "val": "value",
        "start": "period_start", "end": "period_end", "filed": "fact_filed",
    })
    return prepare_companyfacts(index, facts, filings, "jevonsintel_sec_companyfacts")


def cached_companyfacts(paths: list[Path], filings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    index_rows = []
    fact_rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cik = str(int(payload["cik"]))
        if path.stem != cik:
            raise ValueError(f"Unexpected CIK {cik} in {path.name}")
        index_rows.extend(
            dict(item, cik=cik, name=payload["name"], accepted=None) for item in payload["filings"]
        )
        for name, definition in payload["facts"].items():
            taxonomy, tag = name.split(":", 1) if ":" in name else ("us-gaap", name)
            for unit, observations in definition["units"].items():
                fact_rows.extend(
                    dict(row, cik=cik, taxonomy=taxonomy, tag=tag, unit=unit)
                    for row in observations if {"accn", "filed", "form", "end", "val"} <= row.keys()
                )
    index = pd.DataFrame(index_rows).rename(columns={"accn": "accession", "doc": "instance"})
    facts = pd.DataFrame(fact_rows).rename(columns={
        "accn": "accession", "val": "value", "start": "period_start",
        "end": "period_end", "filed": "fact_filed",
    })
    return prepare_companyfacts(index, facts, filings, "filings_terminal_sec_companyfacts")


def split_sources(root: Path) -> pd.DataFrame:
    yahoo = pd.read_parquet(root / "yahoo_splits_observed.parquet")
    yahoo = yahoo[["ticker", "ex_date", "numerator", "denominator"]].assign(
        cik=pd.NA, known_date=pd.NaT, identity_verified=False, source="yahoo_ticker_snapshot"
    )
    verified = pd.read_parquet(root / "varve_splits.parquet")
    ratio = verified.ratio.str.extract(r"^([0-9.]+)-for-([0-9.]+)$").astype(float)
    verified = verified.assign(
        numerator=ratio[0], denominator=ratio[1], known_date=verified.announced,
        identity_verified=verified.confidence.eq("high") & verified.text_confirmed.eq(True),
        source="varve_sample",
    )
    splits = pd.concat([yahoo, verified[yahoo.columns]], ignore_index=True)
    splits["cik"] = cik_keys(splits.cik)
    for key in ("known_date", "ex_date"):
        splits[key] = pd.to_datetime(splits[key])
    return splits.loc[splits.numerator.gt(0) & splits.denominator.gt(0) & splits.ex_date.notna()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    store = ParquetStore(args.data)
    staging = store.curated / "tqqq-expansion"
    filings = pd.read_parquet(staging / "fsds-remote" / "filings.parquet")
    filings["cik"] = cik_keys(filings.cik)
    filings["filed"] = pd.to_datetime(filings.filed)
    filings["public_date"] = publication_date(filings)
    filings["industry_basis"] = "filing_sec_sic"
    filings["source"] = "sec_fsds"
    facts = [
        prepare_facts(pd.read_parquet(path), filings, "secdatabase")
        for path in sorted((staging / "s3-history").glob("facts-*.parquet"))
    ]
    for folder in ("fsds-remote", "fsds-bridge", "fsds-recent"):
        path = staging / folder / "facts.parquet"
        if path.is_file():
            facts.append(prepare_facts(pd.read_parquet(path), filings, "sec_fsds"))
    raw = store.raw / "us-financials" / "jevonsintel-2026-09-09"
    if (raw / "sec_financial_observations.jsonl.gz").is_file():
        additional_filings, additional_facts = companyfacts_archive(raw, filings)
        filings = pd.concat([filings, additional_filings], ignore_index=True)
        facts.append(additional_facts)
    cached = sorted((store.raw / "us-financials" / "filings-terminal").glob("*.json"))
    if cached:
        additional_filings, additional_facts = cached_companyfacts(cached, filings)
        filings = pd.concat([filings, additional_filings], ignore_index=True)
        facts.append(additional_facts)
    store.merge_curated(FILINGS, [filings], ["accession"])
    keys = [*FACT_CONTEXT, "period_precision", "accession", "source", "value"]
    store.merge_curated(FACTS, facts, keys)
    splits = split_sources(staging / "coverage-sources")
    store.merge_curated(
        SPLITS, [splits], ["source", "ticker", "cik", "ex_date", "numerator", "denominator"]
    )
    print(f"Published {len(filings):,} filing rows and {len(splits):,} split observations.")
    print(store.read(FACTS).groupby("source").agg(
        observations=("value", "size"), issuers=("cik", "nunique"), latest_disclosure=("public_date", "max")
    ).to_string())


if __name__ == "__main__":
    main()
