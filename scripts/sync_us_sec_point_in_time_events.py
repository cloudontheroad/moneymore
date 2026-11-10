"""Build a point-in-time SEC filing event table for the current research universe."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE = max((ROOT / "data/us_overnight_paper_only/universe").glob("top1000-*.parquet"))
OUTPUT = ROOT / "data/us_overnight_corrected/events/sec"
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "")
HEADERS = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}


def get_json(url: str, attempts: int = 4) -> dict:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.load(response)
        except Exception as exc:  # noqa: BLE001 - bounded network retry
            error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"{url}: {error}")


def filing_rows(symbol: str, cik: int) -> list[dict[str, object]]:
    payload = get_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    recent = payload.get("filings", {}).get("recent", {})
    rows: list[dict[str, object]] = []
    size = len(recent.get("accessionNumber", []))
    for index in range(size):
        accepted = recent.get("acceptanceDateTime", [None] * size)[index]
        filing_date = recent.get("filingDate", [None] * size)[index]
        if filing_date and filing_date < "2023-01-01":
            continue
        rows.append(
            {
                "symbol": symbol,
                "cik": cik,
                "accepted_at": pd.to_datetime(accepted, utc=True, errors="coerce"),
                "filing_date": filing_date,
                "report_date": recent.get("reportDate", [None] * size)[index],
                "form": recent.get("form", [None] * size)[index],
                "items": recent.get("items", [None] * size)[index],
                "accession_number": recent.get("accessionNumber", [None] * size)[index],
                "primary_document": recent.get("primaryDocument", [None] * size)[index],
                "source": "SEC_SUBMISSIONS",
            }
        )
    return rows


def main() -> int:
    if not SEC_USER_AGENT:
        raise RuntimeError(
            "SEC_USER_AGENT is required, for example 'MoneyMore your-email@example.com'"
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tickers = get_json("https://www.sec.gov/files/company_tickers.json")
    mapping = {
        str(value["ticker"]).upper().replace("-", "."): int(value["cik_str"])
        for value in tickers.values()
    }
    universe = pd.read_parquet(UNIVERSE)
    requests = []
    missing = []
    for symbol in universe["code"].astype(str):
        ticker = symbol.removeprefix("US.").upper()
        cik = mapping.get(ticker)
        if cik is None:
            missing.append(symbol)
        else:
            requests.append((symbol, cik))
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(filing_rows, symbol, cik): symbol for symbol, cik in requests}
        for count, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                rows.extend(future.result())
            except Exception as exc:  # noqa: BLE001 - retain per-symbol failure
                failures.append({"symbol": symbol, "error": str(exc)})
            if count % 100 == 0:
                print(json.dumps({"processed": count, "events": len(rows), "failures": len(failures)}), flush=True)
    frame = pd.DataFrame(rows).sort_values(["accepted_at", "symbol"])
    frame.to_parquet(OUTPUT / "filings.parquet", index=False)
    manifest = {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "universe": str(UNIVERSE),
        "mapped_symbols": len(requests),
        "unmapped_symbols": missing,
        "events": len(frame),
        "failures": failures,
        "causal_rule": "usable only when accepted_at is earlier than model decision timestamp",
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
