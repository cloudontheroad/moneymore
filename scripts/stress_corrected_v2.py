"""Predeclared cost/risk replays of saved corrected signals; no tuning or real orders."""

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from replay_corrected_v2 import audit_scores, variant_scores

from moneymore.challenger_diagnostics import trading_dates, write_diagnostic
from moneymore.config import BacktestConfig
from moneymore.data.store import ParquetStore
from moneymore.execution_replay import run_execution_replay
from moneymore.research.baseline_v2_continuous_replay import (
    EMPTY,
    audit_fill_rules,
    audit_ledger,
    prepare_market,
)
from moneymore.research.baseline_v2_daily_replay import (
    DailyOpeningPlans,
    MissingExecutionEvidence,
    diagnostic_performance,
)
from moneymore.research.baseline_v2_platform_risk import PlatformRiskReductionPlans
from moneymore.research.baseline_v2_release import digest, read_json


def stress_market(bars, scenario):
    result = bars.copy()
    if scenario != "cost_10bps":
        return result
    # Re-evaluate opening tradeability: increased slippage must not cross a price limit.
    active = ~result.verified_full_session_suspension.fillna(False).astype(bool)
    result["can_buy"] = active & (result.open * 1.001).lt(result.up_limit - 1e-8)
    result["can_sell"] = active & (result.open * 0.999).gt(result.down_limit + 1e-8)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario", choices=["reference", "native_risk", "cost_10bps"], required=True
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report_path = root / "state/baseline-v2-release/corrected-daily-replay.json"
    reference = read_json(report_path)
    if args.scenario != "reference" and reference["status"] != "COMPLETED_RESEARCH_NOT_PUBLISHED":
        raise ValueError("COMPLETED_CORRECTED_REFERENCE_REQUIRED")
    source = Path(reference["output_directory"])
    frozen = read_json(source / "frozen-inputs.json")
    for name, checksum in frozen["input_hashes"].items():
        if digest(Path(name)) != checksum:
            raise ValueError("CORRECTED_REFERENCE_INPUT_CHANGED:" + name)
    protocol = dict(frozen["protocol"])
    if protocol["real_orders_enabled"] or protocol["automatic_promotion"]:
        raise ValueError("RESEARCH_ONLY_REQUIRED")
    if args.scenario == "cost_10bps":
        protocol["slippage_bps"] = 10
    config = BacktestConfig.model_validate({k: protocol[k] for k in BacktestConfig.model_fields})
    # Identify native execution data explicitly from the frozen list, not newest mutable pointers.
    bound = {Path(p).name: Path(p) for p in frozen["input_hashes"]}
    bars = stress_market(pd.read_parquet(bound["execution-bars.parquet"]), args.scenario)
    actions = pd.read_parquet(bound["execution-actions.parquet"])
    extension_hashes = {}
    for marker in sorted(
        (root / "state/baseline-v2-release/corrected-execution-data/execution-extensions").glob(
            "*/report.json"
        )
    ):
        extra = read_json(marker)
        if (
            extra["score_panel_sha256"]
            != frozen["input_hashes"][str(bound["daily-scores.parquet"])]
        ):
            continue
        if (
            extra["status"] != "READY_FOR_BOUNDED_DAILY_REFERENCE_EXECUTION"
            or extra["start"] > "20240131"
            or extra["end"] < "20251231"
        ):
            raise ValueError("INVALID_CORRECTED_EXECUTION_EXTENSION")
        symbols = set(extra["symbols"])
        if symbols <= set(bars.symbol):
            continue
        if symbols & set(bars.symbol):
            raise ValueError("PARTIALLY_OVERLAPPING_EXTENSION")
        location = Path(extra["output_directory"])
        for name, checksum in extra["artifact_hashes"].items():
            path = location / name
            if digest(path) != checksum:
                raise ValueError("CORRECTED_EXTENSION_CHANGED")
            extension_hashes[str(path)] = checksum
        extension_hashes[str(marker)] = digest(marker)
        extra_actions = pd.read_parquet(location / "replay-actions-extra.parquet")
        market = pd.read_parquet(location / "market-data-readiness-extra.parquet")
        market = market.loc[~market.after_verified_delisting.fillna(False).astype(bool)]
        bars = pd.concat(
            [bars, prepare_market(market, extra_actions, config.slippage_bps)], ignore_index=True
        )
        actions = pd.concat([actions, extra_actions], ignore_index=True)
    if actions.action_key.duplicated().any() or bars.duplicated(["date", "symbol"]).any():
        raise ValueError("DUPLICATE_EXTENDED_MARKET_OR_ACTION")
    pool = pd.read_parquet(bound["identity-top1000-preview.parquet"])
    overlay = pd.read_parquet(bound["native-with-derived-factors.parquet"])
    overlay["price"] = overlay.close * overlay.derived_adj_factor
    prices = overlay.pivot(index="trade_date", columns="ts_code", values="price")
    calendar = trading_dates(ParquetStore(root / "data"), "20261028")
    old_protocol = read_json(bound["frozen-protocol.json"])
    start = old_protocol["scoring"]["coverage_audit"]["identity_scope_start"]
    prices = prices.reindex([d for d in calendar if start <= d <= "20251230"])
    prices.index = pd.to_datetime(prices.index)
    panel = pd.read_parquet(bound["daily-scores.parquet"])
    audit_scores(panel, frozen["policy"], frozen["weights"])
    directory = source / args.scenario / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "RUNNING",
        "scenario": args.scenario,
        "runs": [],
        "output_directory": str(directory),
        "source_directory": str(source),
        "reference_report_sha256": digest(report_path),
        "real_orders_enabled": False,
        "production_changed": False,
        "release_authorized": False,
        "risk_recovery": "EXPLICIT_ONLY_NO_AUTOMATIC_RESUME"
        if args.scenario == "native_risk"
        else "REFERENCE_NO_EXTRA_DRAWDOWN_STOP",
    }
    pointer = (
        root
        / "state/baseline-v2-release"
        / (
            "corrected-daily-replay.json"
            if args.scenario == "reference"
            else "corrected-" + args.scenario + ".json"
        )
    )
    write_diagnostic(
        directory / "frozen-inputs.json",
        {
            **frozen,
            "reference": frozen,
            "protocol": protocol,
            "execution_extension_hashes": extension_hashes,
            "source_hashes": {
                str(p): digest(p)
                for p in [bound["daily-scores.parquet"], source / "frozen-inputs.json"]
            },
        },
    )
    write_diagnostic(pointer, report)
    try:
        for variant in ("fixed", "learned"):
            completed = next(
                (
                    r
                    for r in reference.get("runs", [])
                    if r.get("variant") == variant and r.get("days") == 464
                ),
                None,
            )
            if args.scenario == "reference" and completed:
                equity = pd.read_parquet(source / (variant + "-equity.parquet"))
                fills = pd.read_parquet(source / (variant + "-fills.parquet")).to_dict("records")
                proof, _, _ = audit_ledger(
                    source / (variant + ".sqlite3"), equity, bars, actions, config.initial_cash
                )
                audit_fill_rules(fills, bars, config)
                for suffix in (
                    "-equity.parquet",
                    "-fills.parquet",
                    "-opening-audits.json",
                    "-closing-decisions.json",
                ):
                    shutil.copy2(source / (variant + suffix), directory / (variant + suffix))
                proof.to_parquet(directory / (variant + "-ledger-audit.parquet"), index=False)
                report["runs"].append(
                    {
                        **completed,
                        "reused_completed_account": str(source / (variant + ".sqlite3")),
                        "rechecked_ledger_max_error": float(proof.equity_error.abs().max()),
                    }
                )
                write_diagnostic(pointer, report)
                continue
            database = directory / (variant + ".sqlite3")
            planner_class = (
                PlatformRiskReductionPlans if args.scenario == "native_risk" else DailyOpeningPlans
            )
            kwargs = (
                {"risk_database": database, "account_id": variant}
                if args.scenario == "native_risk"
                else {}
            )
            planner = planner_class(
                variant_scores(panel, variant),
                pool,
                prices,
                frozen["policy"],
                config,
                protocol,
                calendar,
                "20251231",
                set(bars.symbol),
                **kwargs,
            )
            replay = run_execution_replay(
                bars=bars,
                targets=EMPTY,
                config=config,
                database=database,
                strategy_id=variant,
                account_id=variant,
                closing_target_provider=planner.closing,
                opening_plan_provider=planner.opening,
                corporate_actions=actions,
                accrue_action_receivables=True,
            )
            expected = [d for d in calendar if "20240131" <= d <= "20251231"]
            if replay.equity.trade_date.astype(str).tolist() != expected:
                raise ValueError("STRESS_CALENDAR_INCOMPLETE")
            proof, _, _ = audit_ledger(database, replay.equity, bars, actions, config.initial_cash)
            replay.equity.to_parquet(directory / (variant + "-equity.parquet"), index=False)
            pd.DataFrame(replay.fills).to_parquet(
                directory / (variant + "-fills.parquet"), index=False
            )
            proof.to_parquet(directory / (variant + "-ledger-audit.parquet"), index=False)
            opening = replay.summary["opening_plan_audits"]
            write_diagnostic(directory / (variant + "-opening-audits.json"), {"plans": opening})
            write_diagnostic(
                directory / (variant + "-closing-decisions.json"), {"events": planner.events}
            )
            entry = {
                "variant": variant,
                "days": len(replay.equity),
                "fills": len(replay.fills),
                "ledger_max_error": float(proof.equity_error.abs().max()),
                "fill_rules": audit_fill_rules(replay.fills, bars, config),
                "reconciliation": replay.reconciliation,
                "performance": diagnostic_performance(
                    replay.equity, replay.fills, config.initial_cash
                ),
                "input_hold_days": sum(
                    e["status"] == "BLOCKED_HELD_INPUT_HOLD_AND_ALERT" for e in planner.events
                ),
            }
            report["runs"].append(entry)
            write_diagnostic(pointer, report)
            print(json.dumps({"scenario": args.scenario, **entry}), flush=True)
        report["status"] = "COMPLETED_RESEARCH_NOT_PUBLISHED"
    except MissingExecutionEvidence as exc:
        report.update(
            status="FAILED",
            error="MissingExecutionEvidence:DAILY_EXECUTION_EVIDENCE_MISSING",
            missing_execution_symbols=exc.symbols,
            blocked_signal_date=planner.events[-1]["signal_date"],
        )
        raise
    except Exception as exc:
        report.update(status="FAILED", error=type(exc).__name__ + ":" + str(exc))
        raise
    finally:
        write_diagnostic(directory / "report.json", report)
        write_diagnostic(pointer, report)


if __name__ == "__main__":
    main()
