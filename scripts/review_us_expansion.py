"""Query one shared expansion dataset for TQQQ or US Overnight Top-N research."""

from __future__ import annotations

import argparse
from pathlib import Path

from moneymore.data.store import ParquetStore
from moneymore.data.us_financials import annual_expansion_as_of


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--as-of", required=True, help="US exchange session date")
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--industry", nargs="+", help="TQQQ research by historical FF49 industry")
    scope.add_argument("--cik", nargs="+", help="Top-N research by verified issuer CIK")
    args = parser.parse_args()
    pairs = annual_expansion_as_of(
        ParquetStore(args.data), args.as_of, ciks=args.cik, industries=args.industry
    )
    if args.industry:
        report = pairs.assign(expanding=pairs.change.gt(0)).groupby(["industry", "tag"]).agg(
            paired_issuers=("cik", "nunique"), expanding_issuers=("expanding", "sum"),
        )
    else:
        report = pairs[["cik", "tag", "period_end", "public_date", "prior_public_date", "value", "prior_value", "growth"]]
    print(report.to_string())


if __name__ == "__main__":
    main()
