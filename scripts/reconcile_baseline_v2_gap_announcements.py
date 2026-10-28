"""Apply reviewed announcement explanations to a NEW artifact, not to price data."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.research.baseline_v2_gap_evidence import announcement_gap_reason

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    base = root / "state/baseline-v2-validation/daily-inputs"
    report = json.loads((base / "gap-investigation.json").read_text(encoding="utf-8"))
    source = Path(report["output_directory"]) / "price-gaps.parquet"
    holes = pd.read_parquet(source)
    config = root / "configs/baseline_v2_market_evidence.yaml"
    evidence = yaml.safe_load(config.read_text(encoding="utf-8"))
    reviewed = holes.copy()
    reviewed["announcement_source"] = None
    for idx, row in holes.loc[holes.status.eq("UNEXPLAINED_PRICE_GAP")].iterrows():
        reason, url = announcement_gap_reason(row.symbol, row.trade_date, evidence)
        reviewed.loc[idx, ["status", "announcement_source"]] = [reason, url]
    directory = base / "announcement-reviews" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True)
    reviewed.to_parquet(directory / "reviewed-price-gaps.parquet", index=False)
    result = {
        "status": "EXPLAINED_NOT_AUTOMATICALLY_REPAIRED",
        "output_directory": str(directory),
        "source_gap_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "reviewed_evidence_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "reason_counts": reviewed.status.value_counts().to_dict(),
        "blocked_by_security_identity_migration": bool(
            reviewed.status.eq("SYMBOL_CHANGE_REQUIRES_IDENTITY_MAPPING").any()
        ),
        "prices_filled": False,
        "eligibility_changed": False,
        "real_orders_enabled": False,
    }
    write_diagnostic(directory / "report.json", result)
    write_diagnostic(base / "announcement-review.json", result)
    print(json.dumps(result))
