"""Exactly four bounded provider reads for one isolated continuation day."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from dotenv import load_dotenv
from moneymore.data.tushare_provider import TushareProvider


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("immutable evidence")
    args.output.mkdir(parents=True)
    load_dotenv(Path(__file__).resolve().parents[1]/".env")
    client = TushareProvider()._client
    requests = []
    for endpoint, kwargs, filename in [
        ("stk_limit", {"trade_date":"20261027"}, "stk_limit-20261027.parquet"),
        ("suspend_d", {"trade_date":"20261027"}, "suspend_d-20261027.parquet"),
        ("dividend", {"ex_date":"20261027"}, "dividend-ex-20261027.parquet"),
        ("dividend", {"record_date":"20261027"}, "dividend-record-20261027.parquet"),
    ]:
        try:
            frame = getattr(client, endpoint)(**kwargs)
            if frame is None or (endpoint == "stk_limit" and frame.empty):
                raise ValueError("NULL_OR_INCOMPLETE_RESPONSE")
            path = args.output/filename; frame.to_parquet(path, index=False)
            requests.append({"endpoint":endpoint, "file":filename, "rows":len(frame), "sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
        except Exception as error:
            requests.append({"endpoint":endpoint, "file":filename, "error_type":type(error).__name__})
        time.sleep(1.1)
    report = {"status":"READY_FOR_ISOLATED_EXTENSION" if all("sha256" in r for r in requests) else "BLOCKED_EXTENSION_INPUTS",
        "orders_enabled":False, "date":"20261027", "max_requests":4, "requests":requests,
        "provider_error_bodies_suppressed":True}
    (args.output/"manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
