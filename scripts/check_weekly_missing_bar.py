"""One bounded official suspension query; isolated evidence, no production writes."""
import json
from pathlib import Path
from dotenv import load_dotenv
from moneymore.data.tushare_provider import TushareProvider

root = Path(__file__).resolve().parents[1]
load_dotenv(root / ".env", override=False)
output = root / "state/weekly-learning/20261030-missing-evidence"
output.mkdir(exist_ok=True)
try:
    frame = TushareProvider().suspensions("601326.SH", "20261021", "20261029")
    frame.to_parquet(output / "suspension.parquet")
    result = {"status": "QUERIED", "rows": len(frame), "records": json.loads(frame.to_json(orient="records")), "requests": 1}
except Exception as exc:
    result = {"status": "FAILED", "error_type": type(exc).__name__, "requests": 1}
(output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False))
