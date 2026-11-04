import json
from pathlib import Path
import sys
from moneymore.close_valuation import recover_close_valuations

root = Path(__file__).resolve().parents[1]
result = recover_close_valuations(root, sys.argv[1])
folder = root / "state/close-valuation-recovery"
folder.mkdir(exist_ok=True)
(folder / (sys.argv[1]+".json")).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False), flush=True)
