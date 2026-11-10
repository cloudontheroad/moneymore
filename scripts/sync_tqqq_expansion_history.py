"""Backfill a sample of historical XBRL facts, with SEC filing identity and dates."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb

from build_tqqq_expansion_data import BALANCE_TAGS, FLOW_TAGS, build_coverage, save_table

SOURCE = (
    "https://s3.amazonaws.com/dataset.secdatabase.com/sec_financial_statements/parquet/"
    "20200930/data_point/20201004_143134_00089_tm888_bucket-"
)


def export_shard(index: int, filings: Path, output: Path) -> int:
    target = output / f"facts-{index:03d}.parquet"
    if target.is_file():
        return index
    temporary = target.with_suffix(".parquet.tmp")
    with duckdb.connect(config={"threads": 2}) as db:
        db.execute("LOAD httpfs; SET http_timeout=45; SET http_retries=2")
        db.execute("""
            CREATE TEMP TABLE shard AS
            SELECT DISTINCT f.accession, f.cik, f.industry, f.form, f.period, f.accepted,
                p.datapoint_name AS tag, p.version, p.segment_label, p.start_date, p.end_date,
                p.period_month, p.unit, p.numeric_value AS value
            FROM read_parquet(?) p JOIN read_parquet(?) f
              ON p.accession_number_int = TRY_CAST(REPLACE(f.accession, '-', '') AS BIGINT)
             AND p.cik = TRY_CAST(f.cik AS INTEGER)
            WHERE f.industry IN ('Chips', 'Hardw', 'Softw', 'LabEq')
              AND list_contains(?, p.datapoint_name) AND p.numeric_value IS NOT NULL
              AND p.end_date <= CAST(f.accepted AS DATE)
        """, [f"{SOURCE}{index:05d}", str(filings), BALANCE_TAGS + FLOW_TAGS])
        db.execute("COPY shard TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(temporary)])
    temporary.replace(target)
    return index


def summarize(filings: Path, output: Path, shards: list[int]) -> dict:
    paths = [str(output / f"facts-{index:03d}.parquet") for index in shards]
    with duckdb.connect() as db:
        db.execute("CREATE TEMP TABLE raw_facts AS SELECT DISTINCT * FROM read_parquet(?)", [paths])
        db.execute("""
            CREATE TEMP TABLE filings AS
            SELECT * FROM read_parquet(?) WHERE accession IN (SELECT accession FROM raw_facts)
        """, [str(filings)])
        db.execute("""
            CREATE TEMP TABLE facts AS
            SELECT accession, tag, version, LAST_DAY(end_date) AS fact_date,
                CASE WHEN start_date IS NULL THEN 0 ELSE TRY_CAST(period_month / 3 AS INTEGER) END AS qtrs,
                unit AS uom, NULLIF(segment_label, '') AS segments, value
            FROM raw_facts
        """)
        build_coverage(db)
        for table in ("annual_noa", "coverage"):
            save_table(db, table, output)
        summary = db.execute("""
            SELECT COUNT(*) AS fact_rows, COUNT(DISTINCT accession) AS filings,
                COUNT(DISTINCT cik) AS companies, COUNT(DISTINCT tag) AS tags,
                MIN(accepted) AS first_accepted, MAX(accepted) AS last_accepted,
                COUNT(DISTINCT accession) FILTER (form IN ('10-K', '20-F', '40-F')) AS annual_filings,
                COUNT(DISTINCT accession) FILTER (tag IN ('PaymentsToAcquirePropertyPlantAndEquipment',
                    'PaymentsToAcquireProductiveAssets')) AS capex_filings,
                COUNT(DISTINCT accession) FILTER (tag = 'ProceedsFromIssuanceOfCommonStock') AS issuance_filings,
                (SELECT COUNT(*) FROM annual_noa) AS strict_annual_noa
            FROM raw_facts
        """).fetchdf().iloc[0].to_dict()
    return {"source": SOURCE, "shards": len(shards), "shard_indices": shards, "coverage": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filings", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shards", type=int, default=12, choices=range(1, 241), metavar="1-240")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with duckdb.connect() as db:
        db.execute("INSTALL httpfs")
    failures = {}
    completed = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        jobs = {
            pool.submit(export_shard, index, args.filings, args.output): index
            for index in range(args.shards)
        }
        for job in as_completed(jobs):
            try:
                index = job.result()
                completed.append(index)
                print(f"Saved shard {index:03d}", flush=True)
            except duckdb.Error as error:
                failures[jobs[job]] = str(error).splitlines()[0]
    if completed:
        manifest = summarize(args.filings, args.output, sorted(completed))
        manifest["unavailable_shards"] = sorted(failures)
        payload = json.dumps(manifest, default=str, indent=2)
        (args.output / "manifest.json").write_text(payload + "\n", encoding="utf-8")
        print(payload)
    if failures:
        raise SystemExit(f"Unavailable shards: {failures}")


if __name__ == "__main__":
    main()
