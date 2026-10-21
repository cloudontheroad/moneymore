"""Execute selected paper-account orders at QMT's immutable session open."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from moneymore.bank_daily import BANK_ACCOUNT
from moneymore.baseline_exposure_daily import BASELINE_EXPOSURE_ACCOUNT
from moneymore.config import BacktestConfig
from moneymore.data.realtime import create_realtime_provider
from moneymore.execution.paper import PaperBroker
from moneymore.open_execution import execute_accounts_at_qmt_open


ROOT = Path(__file__).resolve().parents[1]
SHANGHAI = ZoneInfo("Asia/Shanghai")
ACCOUNT_ALIASES = {
    "bank": BANK_ACCOUNT,
    "controlled": BASELINE_EXPOSURE_ACCOUNT,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trade-date", default=datetime.now(SHANGHAI).strftime("%Y%m%d"))
    parser.add_argument("--account", action="append", choices=sorted(ACCOUNT_ALIASES), required=True)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")

    account_ids = [ACCOUNT_ALIASES[value] for value in args.account]
    result = execute_accounts_at_qmt_open(
        provider=create_realtime_provider(root=ROOT),
        broker=PaperBroker(ROOT / "state" / "paper_orders.sqlite3"),
        config=BacktestConfig.from_yaml(ROOT / "configs" / "default.yaml"),
        trade_date=args.trade_date,
        account_ids=account_ids,
        audit_dir=ROOT / "state" / "open-execution-catchup",
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
