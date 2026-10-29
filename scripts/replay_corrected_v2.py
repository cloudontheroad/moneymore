"""Rebuild corrected daily signals and replay isolated integer-cost paper accounts."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from moneymore.challenger_diagnostics import trading_dates, write_diagnostic
from moneymore.config import BacktestConfig
from moneymore.data.fundamental_research import research_financial_frame
from moneymore.data.store import ParquetStore
from moneymore.execution_replay import run_execution_replay
from moneymore.research.baseline_v2_continuous_replay import EMPTY, audit_fill_rules, audit_ledger
from moneymore.research.baseline_v2_daily_replay import (
    DailyOpeningPlans,
    MissingExecutionEvidence,
    diagnostic_performance,
)
from moneymore.research.baseline_v2_daily_scores import score_pair
from moneymore.research.baseline_v2_financials import common_period_features, required_period
from moneymore.research.baseline_v2_industry import industry_as_of
from moneymore.research.baseline_v2_release import digest, read_json
from moneymore.research.security_identity import SecurityIdentity


def corrected_features(prices):
    returns = prices.pct_change(fill_method=None)
    return {
        "momentum_252_21": prices.shift(21) / prices.shift(252) - 1,
        "return_120": prices / prices.shift(120) - 1,
        "volatility_60": returns.rolling(60, min_periods=60).std(ddof=1) * np.sqrt(242),
        "drawdown_120": prices / prices.rolling(120, min_periods=120).max() - 1,
    }


def audit_scores(panel, policy, weights):
    if panel.duplicated(["date", "symbol"]).any():
        raise ValueError("DUPLICATE_DAILY_SCORE")
    names = list(policy["factors"])
    vectors = {
        "fixed": np.array([weights["fixed_weights"][n] for n in names]),
        "learned": np.array([weights["diagnostic_coefficients_not_released"][n] for n in names]),
    }
    vectors["fixed"] /= vectors["fixed"].sum()
    for variant, vector in vectors.items():
        error = np.abs(
            np.einsum("ij,j->i", panel[names].to_numpy(), vector) - panel["score_" + variant]
        )
        if not np.isfinite(error).all() or error.max() > 1e-12:
            raise ValueError("SCORE_ARITHMETIC_MISMATCH")
        for _, frame in panel.groupby("date"):
            ordered = frame.sort_values(["score_" + variant, "symbol"], ascending=[False, True])
            if not np.array_equal(ordered["rank_" + variant], np.arange(1, len(frame) + 1)):
                raise ValueError("DAILY_RANK_MISMATCH")


def variant_scores(panel, variant):
    if variant not in {"fixed", "learned"}:
        raise ValueError("UNKNOWN_FROZEN_SCORE_VARIANT")
    return panel.drop(columns=["score"], errors="ignore").rename(
        columns={"score_" + variant: "score"}
    )


def main():
    root = Path(__file__).resolve().parents[1]
    refit_path = root / "state/baseline-v2-release/corrected-refit.json"
    refit = read_json(refit_path)
    if refit["status"] != "COMPLETED_RESEARCH_NOT_PUBLISHED":
        raise ValueError("COMPLETED_CORRECTED_REFIT_REQUIRED")
    refit_dir = Path(refit["directory"])
    weights_path = refit_dir / "state/baseline-v2-validation/weight-validation.json"
    weights = read_json(weights_path)
    panel_path = refit_dir / "state/baseline-v2-validation/monthly-panel.parquet"
    if digest(panel_path) != weights["panel_sha256"]:
        raise ValueError("REFIT_TRAINING_PANEL_CHANGED")
    overlay_report = read_json(root / "state/baseline-v2-release/adjustment-overlay.json")
    overlay_path = Path(overlay_report["output_directory"]) / "native-with-derived-factors.parquet"
    if digest(overlay_path) != refit["overlay_sha256"]:
        raise ValueError("REPLAY_MUST_USE_EXACT_REFIT_OVERLAY")
    source_report_path = root / "state/baseline-v2-validation/daily-inputs/daily-replay.json"
    source_report = read_json(source_report_path)
    if not source_report["two_year_window_complete"]:
        raise ValueError("COMPLETE_REFERENCE_EXECUTION_REQUIRED")
    source_dir = Path(source_report["output_directory"])
    source = read_json(source_dir / "frozen-protocol.json")
    # Reuse verified native execution evidence, never relabel old signals as new ones.
    hashes = dict(source["input_hashes"])
    for extension in source["execution_extensions"]:
        for name, checksum in extension["artifact_hashes"].items():
            hashes[str(Path(extension["output_directory"]) / name)] = checksum
    for name, checksum in hashes.items():
        if digest(Path(name)) != checksum:
            raise ValueError("REFERENCE_EXECUTION_INPUT_CHANGED:" + name)
    frozen = source["scoring"]
    policy = frozen["policy"]
    if policy["factors"] != weights["fixed_weights"]:
        raise ValueError("REFIT_AND_REPLAY_POLICY_MISMATCH")
    pool_path = (
        Path(frozen["identity_audit"]["output_directory"]) / "identity-top1000-preview.parquet"
    )
    if digest(pool_path) != frozen["pool_sha256"]:
        raise ValueError("POOL_HASH_MISMATCH")
    pool = pd.read_parquet(pool_path)
    if not pool.groupby("signal_date").security_id.nunique().eq(1000).all():
        raise ValueError("COMPLETE_DAILY_POOL_REQUIRED")
    directory = (
        root
        / "state/baseline-v2-release/corrected-daily-replays"
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    )
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "SCORING",
        "output_directory": str(directory),
        "runs": [],
        "real_orders_enabled": False,
        "production_changed": False,
        "release_authorized": False,
        "overlay_scope": refit["overlay_scope"],
        "limitations": [
            "PARTIAL_FACTOR_ECONOMICS_VERIFICATION",
            "VENDOR_FINANCIAL_REVISIONS_NOT_ORIGINAL_PIT",
            "REFERENCE_OPEN_NOT_REAL_PRINTS",
            "DESCRIPTIVE_REPLAY_NOT_ROBUSTNESS_PASS",
        ],
    }
    pointer = root / "state/baseline-v2-release/corrected-daily-replay.json"
    write_diagnostic(pointer, report)
    try:
        store = ParquetStore(root / "data")
        calendar = trading_dates(store, "20261028")
        signals = sorted(pool.signal_date.unique())
        dates = [
            d
            for d in calendar
            if frozen["coverage_audit"]["identity_scope_start"] <= d <= signals[-1]
        ]
        overlay = pd.read_parquet(overlay_path)
        overlay["price"] = overlay.close * overlay.derived_adj_factor
        prices = overlay.pivot(index="trade_date", columns="ts_code", values="price").reindex(
            index=dates
        )
        prices = prices.where(np.isfinite(prices) & prices.gt(0))
        core = corrected_features(prices)
        continuous = prices.notna().rolling(253, min_periods=253).sum().eq(253)
        identity = SecurityIdentity(frozen["coverage_audit"]["identity_policy"])
        finances = identity.market_rows(
            research_financial_frame(store),
            "ann_date",
            ["roe", "q_sales_yoy"],
            record_keys=["ann_date", "end_date", "growth_source"],
        )
        industry = identity.industry_rows(store.read("baseline_v2_industry_membership"))
        finances.to_parquet(directory / "financial-input.parquet", index=False)
        industry.to_parquet(directory / "industry-input.parquet", index=False)
        frames, exclusions = [], []
        for number, signal in enumerate(signals, 1):
            raw = pool.loc[pool.signal_date.eq(signal)].rename(columns={"ts_code": "symbol"}).copy()
            if not raw.trade_date.eq(calendar[calendar.index(signal) - 1]).all():
                raise ValueError("POOL_NOT_PRIOR_SESSION")
            raw["date"] = pd.Timestamp(signal)
            raw["earnings_yield"] = 1 / pd.to_numeric(raw.pe_ttm, errors="coerce").where(
                raw.pe_ttm.gt(0)
            )
            raw["book_to_price"] = 1 / pd.to_numeric(raw.pb, errors="coerce").replace(0, np.nan)
            raw["dividend_yield_ttm"] = pd.to_numeric(raw.dv_ttm, errors="coerce") / 100
            for field, matrix in core.items():
                raw[field] = raw.symbol.map(matrix.loc[signal])
            finance = common_period_features(finances, signal).set_index("ts_code")
            for field, name in [("roe", "roe"), ("q_sales_yoy", "quarterly_revenue_growth")]:
                valid = finance[field + "_end_date"].astype(str).eq(required_period(signal))
                raw[name] = raw.symbol.map(finance[field].where(valid))
            raw["industry"] = raw.symbol.map(industry_as_of(industry, signal).l1_name)
            eligible = raw[policy["required_factors"]].notna().all(axis=1)
            eligible &= raw.symbol.map(continuous.loc[signal]).fillna(False) & raw.industry.notna()
            exclusions.append(raw.loc[~eligible, ["date", "symbol"]])
            frames.append(score_pair(raw.loc[eligible], policy, weights))
            if number % 50 == 0:
                print(
                    json.dumps({"status": "SCORING", "days": number, "required": len(signals)}),
                    flush=True,
                )
        panel = pd.concat(frames, ignore_index=True)
        audit_scores(panel, policy, weights)
        panel.to_parquet(directory / "daily-scores.parquet", index=False)
        pd.concat(exclusions).to_parquet(directory / "excluded-pool-members.parquet", index=False)
        prices.index = pd.to_datetime(prices.index)
        protocol = source["protocol"]
        if protocol["real_orders_enabled"] or protocol["automatic_promotion"]:
            raise ValueError("RESEARCH_ONLY_PROTOCOL_REQUIRED")
        config = BacktestConfig.model_validate(
            {k: protocol[k] for k in BacktestConfig.model_fields}
        )
        bars = pd.read_parquet(source_dir / "execution-bars.parquet")
        actions = pd.read_parquet(source_dir / "execution-actions.parquet")
        # Freeze the independent market support actually reused, including its known bounds.
        report["input_hashes"] = {
            str(p): digest(p)
            for p in [
                refit_path,
                weights_path,
                overlay_path,
                pool_path,
                source_dir / "frozen-protocol.json",
                source_dir / "execution-bars.parquet",
                source_dir / "execution-actions.parquet",
                directory / "financial-input.parquet",
                directory / "industry-input.parquet",
                directory / "daily-scores.parquet",
            ]
        }
        report.update(
            status="REPLAYING",
            signal_days=len(signals),
            score_rows=len(panel),
            execution_symbols=int(bars.symbol.nunique()),
        )
        write_diagnostic(pointer, report)
        write_diagnostic(
            directory / "frozen-inputs.json",
            {
                "policy": policy,
                "protocol": protocol,
                "weights": weights,
                "input_hashes": report["input_hashes"],
                "execution_source_hashes": hashes,
            },
        )
        expected_dates = [d for d in calendar if "20240131" <= d <= "20251231"]
        for variant in ("fixed", "learned"):
            scores = variant_scores(panel, variant)
            database = directory / (variant + ".sqlite3")
            planner = DailyOpeningPlans(
                scores,
                pool,
                prices,
                policy,
                config,
                protocol,
                calendar,
                "20251231",
                set(bars.symbol),
            )
            try:
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
            except MissingExecutionEvidence as exc:
                report["runs"].append(
                    {
                        "variant": variant,
                        "status": "MISSING_EXECUTION_EVIDENCE",
                        "symbols": exc.symbols,
                        "signal_date": planner.events[-1]["signal_date"],
                    }
                )
            else:
                if replay.equity.trade_date.astype(str).tolist() != expected_dates:
                    raise ValueError("REPLAY_CALENDAR_MISMATCH")
                proof, _, _ = audit_ledger(
                    database, replay.equity, bars, actions, config.initial_cash
                )
                replay.equity.to_parquet(directory / (variant + "-equity.parquet"), index=False)
                pd.DataFrame(replay.fills).to_parquet(
                    directory / (variant + "-fills.parquet"), index=False
                )
                proof.to_parquet(directory / (variant + "-ledger-audit.parquet"), index=False)
                write_diagnostic(
                    directory / (variant + "-opening-audits.json"),
                    {"plans": replay.summary["opening_plan_audits"]},
                )
                report["runs"].append(
                    {
                        "variant": variant,
                        "status": "COMPLETED_RESEARCH_REPLAY",
                        "days": len(replay.equity),
                        "fills": len(replay.fills),
                        "ledger_max_error": float(proof.equity_error.abs().max()),
                        "fill_rules": audit_fill_rules(replay.fills, bars, config),
                        "reconciliation": replay.reconciliation,
                        "performance": diagnostic_performance(
                            replay.equity, replay.fills, config.initial_cash
                        ),
                        "input_hold_days": sum(
                            e["status"] == "BLOCKED_HELD_INPUT_HOLD_AND_ALERT"
                            for e in planner.events
                        ),
                    }
                )
            write_diagnostic(
                directory / (variant + "-closing-decisions.json"), {"events": planner.events}
            )
            write_diagnostic(pointer, report)
        report["status"] = (
            "COMPLETED_RESEARCH_NOT_PUBLISHED"
            if all(r["status"] == "COMPLETED_RESEARCH_REPLAY" for r in report["runs"])
            else "MISSING_EXECUTION_EVIDENCE"
        )
    except Exception as exc:
        report.update(status="FAILED", error=type(exc).__name__ + ":" + str(exc))
        raise
    finally:
        write_diagnostic(directory / "report.json", report)
        write_diagnostic(pointer, report)
    print(json.dumps(report, default=str), flush=True)


if __name__ == "__main__":
    main()
