"""Bounded data-only repair; no paper/broker orders or history writes."""
import json
from pathlib import Path
import sys

from dotenv import load_dotenv
from moneymore.data.store import ParquetStore
from moneymore.data.sync import sync_daily_frontier
from moneymore.data.tushare_provider import TushareProvider

root = Path(__file__).resolve().parents[1]
load_dotenv(root / ".env")
day = sys.argv[1]
result = sync_daily_frontier(TushareProvider(), ParquetStore(root / "data"), day)
print(json.dumps(result.__dict__), flush=True)
