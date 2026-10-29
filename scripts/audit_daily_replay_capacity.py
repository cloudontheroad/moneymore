"""Audit original paired replay plus all completed stress accounts, read only."""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.data.store import ParquetStore
from moneymore.research.baseline_v2_release import digest
from moneymore.research.daily_capacity import daily_participation
from moneymore.research.security_identity import SecurityIdentity

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    base = root / "state/baseline-v2-validation/daily-inputs"
    reports = [base / "daily-replay.json", *sorted((base / "stress-replays").glob("*/latest.json"))]
    destination = (
        root
        / "state/baseline-v2-release/capacity"
        / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    )
    destination.mkdir(parents=True, exist_ok=True)
    all_results, artifacts = (
        [],
        {
            str(p): digest(p)
            for p in (
                Path(__file__),
                root / "src/moneymore/research/daily_capacity.py",
                root / "src/moneymore/research/security_identity.py",
            )
        },
    )
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        source = Path(report["output_directory"])
        frozen_path = source / "frozen-protocol.json"
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        identity = SecurityIdentity(frozen["scoring"]["coverage_audit"]["identity_policy"])
        artifacts.update({str(p): digest(p) for p in (path, frozen_path)})
        for run in report["runs"]:
            variant = run["variant"]
            fill_path = source / f"{variant}-fills.parquet"
            fill_sha256 = digest(fill_path)
            fills = pd.read_parquet(fill_path)
            symbols = set(fills.symbol)
            native = symbols | {c["new"] for c in identity.changes if c["old"] in symbols}
            bars = ParquetStore(root / "data").read(
                "daily",
                columns=["ts_code", "trade_date", "vol", "amount"],
                filters=[
                    ("ts_code", "in", sorted(native)),
                    ("trade_date", ">=", str(fills.trade_date.min())),
                    ("trade_date", "<=", str(fills.trade_date.max())),
                ],
            )
            market = identity.market_rows(bars, "trade_date", ["vol", "amount"]).rename(
                columns={"ts_code": "symbol"}
            )
            scenario = report.get("stress_scenario") or "reference_2bps"
            tag = scenario + "-" + variant
            liquidity_path = destination / f"{tag}-observed-liquidity.parquet"
            market.to_parquet(liquidity_path, index=False)
            rows, result = daily_participation(fills, market)
            if digest(fill_path) != fill_sha256:
                raise ValueError("EXECUTED_FILLS_CHANGED_DURING_CAPACITY_AUDIT")
            output = destination / f"{tag}-participation.parquet"
            rows.to_parquet(output, index=False)
            artifacts.update({str(p): digest(p) for p in (fill_path, liquidity_path, output)})
            all_results.append({"scenario": scenario, "variant": variant, **result})
    result = {
        "status": "EX_POST_DAILY_LIQUIDITY_AUDITED_NOT_PRETRADE_OR_OPEN_DEPTH_PROOF",
        "units_source": "https://tushare.pro/document/2?doc_id=27",
        "units": {"vol_to_shares": 100, "amount_to_cny": 1000},
        "source": "CURRENT_CURATED_DAILY_CAPTURED_AS_HASHED_ISOLATED_LIQUIDITY_SUBSETS",
        "runs": all_results,
        "artifact_hashes": artifacts,
        "release_gate_passed": False,
        "real_orders_enabled": False,
        "output_directory": str(destination),
    }
    write_diagnostic(destination / "report.json", result)
    write_diagnostic(root / "state/baseline-v2-release/daily-capacity.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "artifact_hashes"}))
