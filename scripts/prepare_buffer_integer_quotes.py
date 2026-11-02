"""Ten bounded reads; no production data update or broker access."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import yaml
from dotenv import load_dotenv
from moneymore.data.tushare_provider import TushareProvider


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError("immutable evidence")
    root=Path(__file__).resolve().parents[1];load_dotenv(root/".env")
    protocol=yaml.safe_load((root/"configs/style_buffer_integer_v1.yaml").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True);client=TushareProvider()._client;rows=[]
    for date in protocol["dates"]:
        for endpoint in ["stk_limit","suspend_d"]:
            try:
                frame=getattr(client,endpoint)(trade_date=date)
                if frame is None or (endpoint=="stk_limit" and frame.empty):raise ValueError("INCOMPLETE_RESPONSE")
                path=args.output/f"{endpoint}-{date}.parquet";frame.to_parquet(path,index=False)
                rows.append({"date":date,"endpoint":endpoint,"file":path.name,"rows":len(frame),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
            except Exception as error:
                rows.append({"date":date,"endpoint":endpoint,"error_type":type(error).__name__})
            time.sleep(1.1)
    result={"status":"READY" if all("sha256" in r for r in rows) else "BLOCKED","orders_enabled":False,"max_requests":10,"requests":rows}
    (args.output/"manifest.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"status":result["status"],"requests":len(rows)}))


if __name__=="__main__":main()
