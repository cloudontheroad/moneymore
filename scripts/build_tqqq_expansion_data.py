"""Extract point-in-time company expansion inputs from an SEC FSDS archive."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from zipfile import ZipFile

import duckdb
import pandas as pd

BALANCE_TAGS = """
Assets AssetsCurrent Liabilities LiabilitiesCurrent LiabilitiesNoncurrent
LiabilitiesAndStockholdersEquity StockholdersEquity
StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest
CashAndCashEquivalentsAtCarryingValue CashAndShortTermInvestments
CashCashEquivalentsAndShortTermInvestments ShortTermInvestments
ShortTermInvestmentsAvailableForSale MarketableSecuritiesCurrent
AvailableForSaleSecuritiesCurrent AvailableForSaleDebtSecuritiesCurrent
ShortTermBorrowings ShortTermDebt CommercialPaper
DebtCurrent LongTermDebtCurrent LongTermDebtNoncurrent LongTermDebt
LongTermDebtAndFinanceLeaseObligationsCurrent LongTermDebtAndFinanceLeaseObligationsNoncurrent
FinanceLeaseLiabilityCurrent FinanceLeaseLiabilityNoncurrent
PropertyPlantAndEquipmentNet PropertyPlantAndEquipmentGross
InventoryNet AccountsReceivableNetCurrent AccountsPayableCurrent
Goodwill IntangibleAssetsNetExcludingGoodwill
CommonStockSharesOutstanding CommonStockSharesIssued EntityCommonStockSharesOutstanding
""".split()
FLOW_TAGS = """
Revenues RevenueFromContractWithCustomerExcludingAssessedTax SalesRevenueNet
NetIncomeLoss OperatingIncomeLoss ResearchAndDevelopmentExpense
NetCashProvidedByUsedInOperatingActivities PaymentsToAcquirePropertyPlantAndEquipment
PaymentsToAcquireProductiveAssets PaymentsToAcquireBusinessesNetOfCashAcquired
ProceedsFromIssuanceOfCommonStock ProceedsFromStockOptionsExercised
ProceedsFromIssuanceOfSharesUnderIncentiveAndShareBasedCompensationPlansIncludingStockOptions
ProceedsFromSaleOfTreasuryStock PaymentsForRepurchaseOfCommonStock
PaymentsForRepurchaseOfEquity ShareBasedCompensation
WeightedAverageNumberOfSharesOutstandingBasic WeightedAverageNumberOfDilutedSharesOutstanding
""".split()
FORMS = ("10-K", "10-Q", "20-F", "40-F", "10-K/A", "10-Q/A", "20-F/A", "40-F/A")


def sic_mapping(path: Path) -> pd.DataFrame:
    with ZipFile(path) as archive:
        lines = archive.read(archive.namelist()[0]).decode("latin-1").splitlines()
    rows = {}
    for line in lines:
        header = re.match(r"^\s*\d{1,2}\s+(\w+)\s+\S", line)
        code_range = re.match(r"^\s*(\d{4})-(\d{4})\s", line)
        if header:
            industry = header[1]
        if code_range:
            for sic in range(int(code_range[1]), int(code_range[2]) + 1):
                if sic in rows:
                    raise ValueError(f"Overlapping industry assignment for SIC {sic}")
                rows[sic] = industry
    return pd.DataFrame(rows.items(), columns=["sic_code", "industry"])


def save_table(db, table: str, output: Path) -> None:
    target = output / f"{table}.parquet"
    temporary = target.with_suffix(".parquet.tmp")
    db.execute(f"COPY {table} TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(temporary)])
    temporary.replace(target)


def extract_filings(db, mapping: pd.DataFrame, through: str) -> None:
    db.register("sic_mapping", mapping)
    db.execute("""
        CREATE TEMP TABLE filings AS
        SELECT s.adsh AS accession, s.cik, s.name, s.sic, COALESCE(m.industry, 'Other') AS industry,
               s.countryba, s.countryinc, s.form, s.period, s.fy, s.fp, s.filed, s.accepted, s.instance
        FROM submissions s
        LEFT JOIN sic_mapping m ON TRY_CAST(s.sic AS INTEGER) = m.sic_code
        WHERE list_contains(?, s.form) AND CAST(s.accepted AS DATE) <= CAST(? AS DATE)
          AND s.accepted IS NOT NULL AND s.period <= CAST(s.accepted AS DATE)
    """, [list(FORMS), through])


def extract_facts(db, industries: list[str], since: str | None) -> None:
    db.execute("""
        CREATE TEMP TABLE facts AS
        SELECT DISTINCT f.accession, n.tag, n.version, n.ddate AS fact_date, n.qtrs, n.uom,
               NULLIF(n.segments, '') AS segments, n.value
        FROM numbers n
        JOIN filings f ON n.adsh = f.accession
        WHERE ((n.qtrs = 0 AND list_contains(?, n.tag)) OR
               (n.qtrs BETWEEN 1 AND 4 AND list_contains(?, n.tag)))
          AND (len(?) = 0 OR list_contains(?, f.industry))
          AND (CAST(? AS DATE) IS NULL OR CAST(f.accepted AS DATE) >= CAST(? AS DATE))
          AND (CAST(? AS DATE) IS NULL OR n.ddate >= CAST(? AS DATE) - INTERVAL '18 months')
          AND NULLIF(n.coreg, '') IS NULL
          AND n.value IS NOT NULL AND n.ddate <= CAST(f.accepted AS DATE)
    """, [BALANCE_TAGS, FLOW_TAGS, industries, industries, since, since, since, since])


def build_coverage(db) -> None:
    db.execute("""
        CREATE TEMP TABLE filing_fields AS
        WITH unique_facts AS (
            SELECT accession, fact_date, tag, MIN(value) AS value
            FROM facts
            WHERE qtrs = 0 AND segments IS NULL AND version LIKE 'us-gaap/%' AND uom = 'USD'
            GROUP BY accession, fact_date, tag
            HAVING COUNT(DISTINCT value) = 1
        ),
        periods AS (
            SELECT u.accession, u.fact_date,
                MAX(value) FILTER (tag = 'Assets') AS assets,
                MAX(value) FILTER (tag = 'Liabilities') AS liabilities,
                MAX(value) FILTER (tag = 'CashAndCashEquivalentsAtCarryingValue') AS cash,
                COALESCE(
                    MAX(value) FILTER (tag = 'CashAndShortTermInvestments'),
                    MAX(value) FILTER (tag = 'CashCashEquivalentsAndShortTermInvestments'),
                    MAX(value) FILTER (tag = 'CashAndCashEquivalentsAtCarryingValue') +
                    COALESCE(MAX(value) FILTER (tag = 'ShortTermInvestments'),
                             MAX(value) FILTER (tag = 'MarketableSecuritiesCurrent'))
                ) AS cash_and_investments,
                COALESCE(MAX(value) FILTER (tag = 'ShortTermBorrowings'),
                         MAX(value) FILTER (tag = 'ShortTermDebt')) AS short_borrowings,
                MAX(value) FILTER (tag = 'DebtCurrent') AS current_debt,
                MAX(value) FILTER (tag = 'LongTermDebtCurrent') AS long_debt_current,
                MAX(value) FILTER (tag = 'LongTermDebtNoncurrent') AS long_debt_noncurrent
            FROM unique_facts u
            GROUP BY u.accession, u.fact_date
        )
        SELECT f.accession, f.cik, f.industry, f.period, f.accepted, f.form, p.fact_date,
            p.assets, p.liabilities, p.cash, p.cash_and_investments,
            COALESCE(p.current_debt, p.short_borrowings + p.long_debt_current) AS short_debt,
            p.long_debt_noncurrent AS long_debt
        FROM filings f
        JOIN periods p ON p.accession = f.accession AND
            (p.fact_date = f.period OR p.fact_date = LAST_DAY(f.period - INTERVAL '1 year'))
    """)
    db.execute("""
        CREATE TEMP TABLE annual_noa AS
        WITH observed AS (
            SELECT *, assets - cash_and_investments - liabilities + short_debt + long_debt AS noa
            FROM filing_fields
            WHERE form IN ('10-K', '20-F', '40-F') AND assets > 0
        )
        SELECT c.accession, c.cik, c.industry, c.period, c.accepted,
               p.noa AS prior_noa, c.noa, p.assets AS prior_assets, c.assets,
               (c.noa - p.noa) / ((c.assets + p.assets) / 2) AS delta_noa
        FROM observed c
        JOIN observed p ON p.accession = c.accession AND p.fact_date = LAST_DAY(c.period - INTERVAL '1 year')
        WHERE c.fact_date = c.period AND c.noa IS NOT NULL AND p.noa IS NOT NULL
    """)
    db.execute("""
        CREATE TEMP TABLE coverage AS
        WITH available AS (
            SELECT accession,
                BOOL_OR(tag = 'Assets' AND qtrs = 0) AS assets,
                BOOL_OR(tag = 'Liabilities' AND qtrs = 0) AS liabilities,
                BOOL_OR(tag IN ('PaymentsToAcquirePropertyPlantAndEquipment', 'PaymentsToAcquireProductiveAssets')) AS capex,
                BOOL_OR(tag IN ('CommonStockSharesOutstanding', 'EntityCommonStockSharesOutstanding')) AS shares,
                BOOL_OR(tag = 'ProceedsFromIssuanceOfCommonStock') AS issuance_proceeds
            FROM facts JOIN filings USING (accession)
            WHERE fact_date = period AND segments IS NULL
            GROUP BY accession
        )
        SELECT YEAR(f.accepted) AS filing_year, f.industry,
            COUNT(*) AS filings, COUNT(DISTINCT f.cik) AS companies,
            COUNT(*) FILTER (a.assets) AS assets,
            COUNT(*) FILTER (a.liabilities) AS liabilities,
            COUNT(*) FILTER (a.capex) AS capex,
            COUNT(*) FILTER (a.shares) AS shares,
            COUNT(*) FILTER (a.issuance_proceeds) AS issuance_proceeds,
            COUNT(*) FILTER (n.delta_noa IS NOT NULL) AS strict_annual_noa
        FROM filings f
        LEFT JOIN available a USING (accession)
        LEFT JOIN annual_noa n USING (accession)
        GROUP BY filing_year, f.industry
        ORDER BY filing_year, f.industry
    """)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, help="Local DuckDB archive or HTTPS URL")
    parser.add_argument("--sic", required=True, type=Path)
    parser.add_argument("--through", required=True, help="Latest EDGAR acceptance date (YYYY-MM-DD)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--filings-only", action="store_true", help="Export only the filing index")
    parser.add_argument("--industries", nargs="*", default=[], help="Limit fact extraction to these FF49 industries")
    parser.add_argument("--fact-since", help="Earliest acceptance date for facts; keep preceding comparative periods")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    remote = args.archive.startswith("https://")
    with duckdb.connect(":memory:" if remote else args.archive, read_only=not remote, config={"threads": 8}) as db:
        if remote:
            db.execute("INSTALL httpfs; LOAD httpfs")
            db.execute("SET http_timeout=75; SET http_retries=3")
            db.execute("ATTACH ? AS fsds (READ_ONLY)", [args.archive])
            db.execute("USE fsds")
        extract_filings(db, sic_mapping(args.sic), args.through)
        save_table(db, "filings", args.output)
        print("Filing metadata saved", flush=True)
        if args.filings_only:
            return
        extract_facts(db, args.industries, args.fact_since)
        build_coverage(db)
        for table in ("facts", "annual_noa", "coverage"):
            save_table(db, table, args.output)
        summary = db.execute("""
            SELECT COUNT(*) AS filings, COUNT(DISTINCT cik) AS companies,
                   MIN(accepted) AS first_acceptance, MAX(accepted) AS last_acceptance,
                   (SELECT COUNT(*) FROM facts) AS facts,
                   (SELECT COUNT(DISTINCT tag) FROM facts) AS distinct_tags,
                   (SELECT COUNT(*) FROM annual_noa) AS strict_annual_noa,
                   (SELECT COUNT(DISTINCT cik) FROM annual_noa) AS strict_noa_companies
            FROM filings
        """).fetchdf().iloc[0].to_dict()
    manifest = {"archive": args.archive, "through": args.through, "fact_since": args.fact_since,
                "fact_industries": args.industries, "coverage": summary}
    output = json.dumps(manifest, default=str, indent=2)
    (args.output / "manifest.json").write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
