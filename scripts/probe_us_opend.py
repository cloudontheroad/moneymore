"""Print a sanitized, quote-only Futu OpenD connectivity and quota probe."""

from __future__ import annotations

import json
from pathlib import Path

from moneymore.us_overnight.config import USOvernightConfig
from moneymore.us_overnight.data import FutuOpenDReadOnly

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    config = USOvernightConfig.from_yaml(ROOT / "configs/us_overnight_paper_only.yaml")
    result = FutuOpenDReadOnly(config.opend_host, config.opend_port).probe(timeout=1.0)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result["status"] == "ONLINE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
