"""Build the point-in-time US top-1000 market-cap paper universe from OpenD."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from moneymore.us_overnight.config import USOvernightConfig
from moneymore.us_overnight.data import FutuOpenDReadOnly

ROOT = Path(__file__).resolve().parents[1]


def build_top_market_cap_universe(
    basic: pd.DataFrame, snapshots: pd.DataFrame, captured_at: datetime, limit: int = 1000
) -> pd.DataFrame:
    columns = ["code", "name", "listing_date", "exchange_type", "delisting"]
    merged = snapshots.merge(basic[columns], on="code", how="inner", suffixes=("", "_basic"))
    market_cap = pd.to_numeric(merged["total_market_val"], errors="coerce")
    last_price = pd.to_numeric(merged["last_price"], errors="coerce")
    issued = pd.to_numeric(merged["issued_shares"], errors="coerce")
    eligible = merged.loc[
        market_cap.gt(0)
        & last_price.gt(0)
        & issued.gt(0)
        & merged["equity_valid"].fillna(False).astype(bool)
        & ~merged["delisting"].fillna(False).astype(bool)
        & merged["exchange_type"].isin(["US_NYSE", "US_NASDAQ", "US_AMEX"])
    ].copy()
    eligible["market_cap"] = market_cap.loc[eligible.index]
    eligible["captured_at_utc"] = captured_at.isoformat()
    eligible = eligible.sort_values(["market_cap", "code"], ascending=[False, True]).head(limit)
    eligible["rank"] = range(1, len(eligible) + 1)
    return eligible[
        [
            "rank",
            "code",
            "name",
            "market_cap",
            "last_price",
            "issued_shares",
            "circular_market_val",
            "exchange_type",
            "listing_date",
            "sec_status",
            "update_time",
            "captured_at_utc",
        ]
    ].reset_index(drop=True)


def main() -> int:
    config = USOvernightConfig.from_yaml(ROOT / "configs/us_overnight_paper_only.yaml")
    adapter = FutuOpenDReadOnly(config.opend_host, config.opend_port)
    captured_at = datetime.now(UTC)
    basic = adapter.stock_basicinfo()
    supported_exchange = basic["exchange_type"].isin(["US_NYSE", "US_NASDAQ", "US_AMEX"])
    active = ~basic["delisting"].fillna(False).astype(bool)
    symbols = basic.loc[supported_exchange & active, "code"].astype(str).tolist()
    snapshots = adapter.market_snapshots(symbols)
    universe = build_top_market_cap_universe(
        basic, snapshots, captured_at, limit=config.universe_top_n
    )
    if len(universe) != config.universe_top_n:
        raise RuntimeError(
            f"expected {config.universe_top_n} eligible stocks, got {len(universe)}"
        )
    target = ROOT / config.output_root / "universe"
    target.mkdir(parents=True, exist_ok=True)
    stamp = captured_at.strftime("%Y-%m-%dT%H%M%SZ")
    parquet = target / f"top1000-{stamp}.parquet"
    csv = target / f"top1000-{stamp}.csv"
    universe.to_parquet(parquet, index=False)
    universe.to_csv(csv, index=False, encoding="utf-8-sig")
    manifest = {
        "captured_at_utc": captured_at.isoformat(),
        "source": "FUTU_OPEND_READ_ONLY",
        "ranking": "total_market_val_desc",
        "rows": len(universe),
        "parquet": parquet.name,
        "csv": csv.name,
        "real_orders_enabled": False,
    }
    (target / f"top1000-{stamp}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {**manifest, "top_10": universe.head(10).to_dict("records")},
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
