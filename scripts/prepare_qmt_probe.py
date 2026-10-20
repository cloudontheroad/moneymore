"""Generate a local-only QMT probe with private account settings embedded.

Run locally, then paste the generated source into QMT. QMT itself never reads
the private configuration file. Never log account settings or commit output.
"""
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    private = root / "state" / "private"
    settings = json.loads((private / "qmt_account.json").read_text(encoding="utf-8"))
    account = str(settings["account_id"])
    account_type = settings.get("account_type", "STOCK")
    if not account.isdigit() or account_type not in {"STOCK", "CREDIT"}:
        raise ValueError("Invalid private account settings")
    source = (root / "scripts" / "qmt_builtin_stage2_probe.py").read_text(encoding="utf-8")
    if source.count('ACCOUNT_ID = ""') != 1:
        raise ValueError("Account placeholder must occur exactly once")
    source = source.replace('ACCOUNT_ID = ""', "ACCOUNT_ID = " + json.dumps(account), 1)
    source = source.replace('ACCOUNT_TYPE = "STOCK"', "ACCOUNT_TYPE = " + json.dumps(account_type), 1)
    target = private / "qmt_stage2_ready.py"
    target.write_text(source, encoding="utf-8")
    print("Prepared local-only QMT probe; account identifier not printed.")


if __name__ == "__main__":
    main()
