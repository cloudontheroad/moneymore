# US overnight event warehouse

This warehouse supplies point-in-time inputs for the open(t) to open(t+1)
selector. It is research-only and does not call trading APIs.

## Time contract

- Timestamped events are eligible only when their source event timestamp is
  earlier than the model decision timestamp.
- Date-only ratings become eligible on the next US regular trading session.
- Current plate membership is a snapshot and is never backfilled before
  `first_observed_at_utc`.
- A macro observation without `release_time` is usable only from
  `first_observed_at_utc`; historical backtests require a vintage source.
- Every run preserves an immutable snapshot and incrementally merges a canonical
  table without replacing the first-observed timestamp.

## Tables

`earnings`, `economic`, `macro`, and `ratings` are market-wide structured event
feeds; `plates` is a point-in-time snapshot. Verified sparse discoveries live in
`web_events`. They are stored under
`data/us_overnight_event_warehouse/<table>/`. Each directory contains
`canonical.parquet` and timestamped files in `snapshots/`.

## Commands

```powershell
.\.venv\Scripts\python.exe scripts\sync_us_event_data_opend.py --start 2026-11-01
.\.venv\Scripts\python.exe scripts\import_us_web_event_discovery.py data\discovery\verified-events.jsonl
.\.venv\Scripts\python.exe scripts\audit_us_event_warehouse.py
```

News is sparse and must not be polled once per symbol. The production workflow
uses the market-wide query plan in `configs/us_event_websearch_queries.yaml`, maps
entities found in results to the Top1000, then imports only verified discoveries
with `scripts/import_us_web_event_discovery.py`. Per-symbol news polling has been
removed from the OpenD synchronization script. A discovered event may still be
verified against the issuer's IR page, SEC filing, or a second independent source.

SEC collection additionally requires a truthful `SEC_USER_AGENT` containing a
contact address. FRED/ALFRED vintage collection requires `FRED_API_KEY`. Missing
credentials must remain an explicit blocked source, never be replaced with a
fabricated identity or revised modern values.
