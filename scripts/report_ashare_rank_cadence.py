"""Immutable, read-only reporting of the frozen raw-rank cadence/drop5 factorial."""

import argparse
import importlib.util
import json
from collections import Counter
from functools import cache
from pathlib import Path

import pandas as pd

from moneymore.research.ashare_mainline import sha256

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "state/ashare-mainline-20261112/n3r-rank-v2"
MODEL = "doubleensemble_full"
COSTS = (10, 20)
SESSIONS = 181
CASE_SPECS = {
    "rank_cap": {"rebalance_sessions": 20, "max_optional_exits": None},
    "rank_cap_buffer": {"rebalance_sessions": 20, "max_optional_exits": 5},
    "rank_cap_fast": {"rebalance_sessions": 10, "max_optional_exits": None},
    "rank_cap_fast_buffer": {"rebalance_sessions": 10, "max_optional_exits": 5},
}
REPORT_SOURCES = (
    "scripts/report_ashare_rank_cadence.py",
    "tests/test_rank_cadence_report.py",
    "scripts/report_ashare_rank_preserving.py",
    "scripts/report_ashare_mainline_replay.py",
    "scripts/finalize_ashare_mainline_n3.py",
)


@cache
def sibling(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def comparison_specs():
    return [
        ("rank_cap_buffer", "rank_cap", "buffer_at_20_sessions"),
        ("rank_cap_fast_buffer", "rank_cap_fast", "buffer_at_10_sessions"),
        ("rank_cap_fast", "rank_cap", "cadence_without_buffer"),
        ("rank_cap_fast_buffer", "rank_cap_buffer", "cadence_with_buffer"),
        ("rank_cap_fast_buffer", "rank_cap", "combined_against_frozen_control"),
    ]


def validate_contract(contract):
    if (
        contract.get("models") != [MODEL]
        or len(contract.get("allocators", [])) != len(CASE_SPECS)
        or set(contract.get("allocators", [])) != set(CASE_SPECS)
        or contract.get("case_specs") != CASE_SPECS
        or contract.get("initial_cash") != 10_000_000
        or contract.get("slippage_bps") != list(COSTS)
        or contract.get("rebalance_sessions") != 20
        or contract.get("n_drop") != 5
        or contract.get("topk") != 50
        or contract.get("stock_exposure") != 0.995
        or contract.get("max_stock_weight") != 0.05
        or contract.get("industry_active_upper_planning") != 0.08
        or contract.get("industry_active_upper_execution") != 0.10
        or contract.get("real_orders_enabled") is not False
        or contract.get("production_enabled") is not False
    ):
        raise ValueError("FROZEN_CADENCE_FACTORIAL_CONTRACT_MISMATCH")


def validate_sources(run):
    contract_path = run / "cost10/contract.json"
    contract = json.loads(contract_path.read_bytes())
    validate_contract(contract)
    manifests, previous_hashes = {}, None
    for bps in COSTS:
        child = run / f"cost{bps}"
        if sha256(child / "contract.json") != sha256(contract_path):
            raise ValueError("MIXED_EXPERIMENT_CONTRACTS")
        runtime = json.loads((child / "runtime.json").read_bytes())
        if runtime["status"] != "COMPLETE" or runtime.get("smoke", False) or runtime["bps"] != bps:
            raise ValueError("FULL_CADENCE_MATRIX_NOT_COMPLETE")
        hashes = runtime.get("source_hashes", {})
        normalized = {key.replace("\\", "/"): value for key, value in hashes.items()}
        if not set(REPORT_SOURCES).issubset(normalized):
            raise ValueError("MISSING_REPORT_OR_TEST_SOURCE_SNAPSHOT")
        if previous_hashes is not None and hashes != previous_hashes:
            raise ValueError("SOURCE_CHANGED_BETWEEN_COSTS")
        previous_hashes = hashes
        for relative, digest in hashes.items():
            if sha256(child / "sources" / relative) != digest:
                raise ValueError("SOURCE_SNAPSHOT_CORRUPTED")
        for relative in REPORT_SOURCES:
            if sha256(ROOT / relative) != normalized[relative]:
                raise ValueError("EXECUTED_REPORT_OR_TEST_DIFFERS_FROM_FROZEN_SOURCE")
        manifests[str(bps)] = {
            "runtime_sha256": sha256(child / "runtime.json"),
            "contract_sha256": sha256(child / "contract.json"),
            "source_hashes": hashes,
        }
    return contract, manifests


def verify_frozen_control(run, reference):
    helper, records = sibling("report_ashare_rank_preserving"), []
    for bps in COSTS:
        case = MODEL + "--rank_cap"
        for name in ("equity.parquet", "fills.parquet"):
            current = run / f"cost{bps}/bps{bps}" / case / name
            frozen = reference / f"cost{bps}/bps{bps}" / case / name
            left, right = pd.read_parquet(current), pd.read_parquet(frozen)
            helper.assert_exact_replay_equal(left, right, f"{bps}:{case}:{name}")
            records.append(
                {
                    "bps": bps,
                    "artifact": name,
                    "rows": len(left),
                    "current_sha256": sha256(current),
                    "frozen_sha256": sha256(frozen),
                    "frozen_path": str(frozen.resolve()),
                    "binary_identical": sha256(current) == sha256(frozen),
                }
            )
    return {
        "passed": True,
        "numeric_atol": 0,
        "numeric_rtol": 0,
        "schema_keys_order_and_values_exactly_equal": True,
        "all_artifacts_binary_identical": all(row["binary_identical"] for row in records),
        "artifacts": records,
    }


def audit_decision(
    decision, day, fills, allocator, index, *, previous_retry=False, held_before=True
):
    """Only the risk-audit alias changes; all four actual strategies retain raw scores."""
    if allocator not in CASE_SPECS:
        raise ValueError("UNKNOWN_CADENCE_ALLOCATOR")
    policy = CASE_SPECS[allocator]
    scheduled = index % policy["rebalance_sessions"] == 0
    if (
        decision.get("cadence_policy") != policy
        or type(decision.get("execution_index")) is not int
        or decision["execution_index"] != index
        or type(decision.get("scheduled_rebalance")) is not bool
        or decision["scheduled_rebalance"] != scheduled
    ):
        raise ValueError("CADENCE_DECISION_METADATA_MISMATCH")
    status = decision["status"]
    permitted = scheduled or previous_retry or not held_before
    if status in ("NOT_SCHEDULED", "MANDATORY_EXITS_ONLY") and permitted:
        raise ValueError("SCHEDULED_OR_RETRY_SESSION_SKIPPED")
    if status not in ("NOT_SCHEDULED", "MANDATORY_EXITS_ONLY") and not permitted:
        raise ValueError("OFF_SCHEDULE_OPTIONAL_PLAN_WITHOUT_RETRY_OR_EMPTY_BOOK")
    if status == "NOT_SCHEDULED" and len(fills):
        raise ValueError("FILL_ON_NOT_SCHEDULED_DAY")
    if status == "BLOCKED_OPTIONAL_TRADES_MANDATORY_EXITS_CHECKED" and (
        not fills.side.eq("SELL").all()
        or not set(fills.symbol).issubset(decision.get("exit_symbols", []))
    ):
        raise ValueError("OPTIONAL_FILL_ON_PROTECTED_DAY")
    allocation = decision.get("allocation")
    if allocation is not None:
        if allocation.get("max_optional_exits") != policy["max_optional_exits"]:
            raise ValueError("OPTIONAL_EXIT_BUDGET_DIFFERS_FROM_FROZEN_CASE")
        forced = set(allocation.get("risk_forced_exits", [])) | set(
            allocation.get("slot_capacity_forced_exits", [])
        )
        if forced & set(allocation.get("optional_exits", [])):
            raise ValueError("FORCED_AND_OPTIONAL_MEMBERSHIP_EXITS_OVERLAP")
    helper = sibling("report_ashare_rank_preserving")
    # This is strictly audit dispatch, never the runner's score or allocator selection.
    alias = "sector_rank_buffer" if policy["max_optional_exits"] == 5 else "rank_cap"
    if allocation is not None:
        helper.audit_allocation(allocation, alias)
    result = helper.audit_decision(decision, day, fills, alias)
    return {
        **result,
        "execution_index": index,
        "scheduled_rebalance": scheduled,
        "rebalance_sessions": policy["rebalance_sessions"],
        "max_optional_exits": policy["max_optional_exits"],
        "off_schedule_retry": not scheduled and previous_retry,
        "off_schedule_empty_book": not scheduled and not held_before,
    }


def audit_decisions(run):
    records, calendar = [], None
    for bps in COSTS:
        folder = run / f"cost{bps}/bps{bps}"
        holdings = pd.read_parquet(folder / "holdings.parquet")
        for allocator in CASE_SPECS:
            case = MODEL + "--" + allocator
            equity = pd.read_parquet(folder / case / "equity.parquet")
            fills = pd.read_parquet(folder / case / "fills.parquet")
            paths = sorted((run / f"cost{bps}/decisions" / case).glob("*.json"))
            days = equity.date.tolist()
            if len(days) != SESSIONS or len(paths) != SESSIONS or [p.stem for p in paths] != days:
                raise ValueError("INCOMPLETE_DECISION_CALENDAR")
            if calendar is not None and days != calendar:
                raise ValueError("MIXED_CADENCE_EQUITY_CALENDARS")
            calendar = days
            held_days = set(holdings.loc[holdings.kind.eq(case), "date"])
            previous_retry = False
            for index, path in enumerate(paths):
                decision = json.loads(path.read_bytes())
                row = audit_decision(
                    decision,
                    path.stem,
                    fills.loc[fills.date.eq(path.stem)],
                    allocator,
                    index,
                    previous_retry=previous_retry,
                    held_before=index > 0 and days[index - 1] in held_days,
                )
                records.append({"bps": bps, "case": case, **row})
                previous_retry = bool(decision.get("retry_next_session", False))
    expected = len(COSTS) * len(CASE_SPECS) * SESSIONS
    if len(records) != expected:
        raise ValueError("INCOMPLETE_CADENCE_DECISION_MATRIX")
    return records, {
        "passed": True,
        "decisions": len(records),
        "verified_open_integer_plans": sum(r["open_postcheck_verified"] for r in records),
        "verified_allocation_caps": sum(r["allocation_caps_verified"] for r in records),
        "scheduled_flags_are_calendar_due_not_trade_occurrence": True,
        "unknown_allocators_fail_closed": True,
        "audit_aliases_do_not_transform_raw_scores": True,
    }


def markdown_report(payload):
    lines = [
        "# 原始排序：调仓周期 × drop5 有限机制研究",
        "",
        "4种组合 × 10/20bps × 181交易日；1000万元。固定DoubleEnsemble全特征预测与原始排序，不重新训练。",
        "共同约束：50等权槽位、目标仓位不高于99.5%、单股5%、行业主动上限计划+8个百分点／执行+10个百分点。",
        "仅比较20/10交易日周期及无缓冲/drop5两因素；不含行业内分位变换、交易带或保留旧权重实验。",
        "rank_cap两档成本逐日净值与逐笔成交严格复现已冻结n3r-rank-v2（零数值容差）。",
        "",
    ]
    for bps in COSTS:
        lines += [
            f"## {bps}bps 全矩阵",
            "",
            "|组合|净收益|夏普|最大回撤|年化波动|平均实际仓位|最小／最大仓位|年化换手¹|平均最大行业|最高行业权重|",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        rows = [row for row in payload["cases"] if row["bps"] == bps]
        for row in rows:
            lines.append(
                f"|{row['case'].split('--')[1]}|{row['net_return']:.2%}|{row['sharpe']:.3f}|"
                f"{row['maximum_drawdown']:.2%}|{row['annualized_volatility']:.2%}|"
                f"{row['mean_actual_stock_exposure']:.2%}|{row['minimum_actual_stock_exposure']:.2%}／"
                f"{row['maximum_actual_stock_exposure']:.2%}|{row['annualized_half_gross_turnover']:.2f}|"
                f"{row['mean_largest_industry_weight']:.2%}|{row['maximum_industry_weight']:.2%}|"
            )
        lines += [
            "",
            "|组合|Q1|Q2|Q3|手续费元|滑点元|成交笔数|买入股数|卖出股数|成交日数|平均持股数|",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            q = [row["quarters"][f"2026Q{i}"]["net_return"] for i in (1, 2, 3)]
            lines.append(
                f"|{row['case'].split('--')[1]}|{q[0]:.2%}|{q[1]:.2%}|{q[2]:.2%}|{row['fees']:,.2f}|"
                f"{row['slippage']:,.2f}|{row['fills']}|{row['buy_shares']:,.0f}|{row['sell_shares']:,.0f}|"
                f"{row['trading_days']}|{row['mean_held_names']:.1f}|"
            )
        lines.append("")
    lines += [
        "## 因子配对与组合效应",
        "",
        "|成本|候选 − 对照|收益差|夏普差|回撤减少|仓位差|换手差|执行成本差元|收益差95%区间|夏普差95%区间|",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in payload["paired_comparisons"]:
        lo, hi = row["net_return_difference_ci95"]
        slo, shi = row["sharpe_difference_ci95"]
        lines.append(
            f"|{row['bps']}|{row['candidate'].split('--')[1]} − {row['reference'].split('--')[1]}|"
            f"{row['net_return_difference']:+.2%}|{row['sharpe_difference']:+.3f}|{row['drawdown_reduction']:+.2%}|"
            f"{row['mean_stock_exposure_difference']:+.2%}|{row['turnover_difference']:+.2f}|"
            f"{row['execution_cost_difference']:+,.2f}|[{lo:.2%}, {hi:.2%}]|[{slo:.3f}, {shi:.3f}]|"
        )
    lines += [
        "",
        "## 保护、计划周期与成员调整",
        "",
        "|成本|组合|保护记录|周期到期日|验证通过计划|非周期重试|冻结日|待退出日|主动换出|行业强制换出|容量强制换出|",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["protection_counts_by_case"]:
        lines.append(
            f"|{row['bps']}|{row['case'].split('--')[1]}|{sum(row['reason_counts'].values())}|"
            f"{row['scheduled_sessions']}|{row['verified_plans']}|{row['off_schedule_retries']}|"
            f"{row['frozen_position_sessions']}|{row['pending_exit_sessions']}|{row['optional_membership_exits']}|"
            f"{row['industry_risk_forced_membership_exits']}|{row['slot_capacity_forced_membership_exits']}|"
        )
    lines += [
        "",
        "## 审计与边界",
        "",
        "- 8账户账本、现金、证券数量及导出成交核验；1448条决策的时间线、周期元数据、行业限额、整数股计划独立检查通过。",
        "- 周期到期不等于实际成交；空仓和前日保护后的重试单列。成分退出、行业/容量强制修复不占主动drop5预算。",
        "- drop5是每次分配的主动成员换出预算，不是成交次数上限；10日周期意味着更多潜在调整机会。",
        "- 成员换出计数取自分配计划，可能包含后续保护未执行的计划；实际交易另看成交笔数、股数与费用，不能混同。",
        "- ¹年化换手＝每日买卖名义额／前日净值／2之和×252/181，含初始建仓；股数变化不含送转权益。",
        "- 配对区间为同日20日环形区块、2000次重采样、种子42，未校正多重比较。该历史已多次查看，不是新留出集。",
        "- 真实仓位、行业集中与执行成本均单列；较低回撤或波动不等于预测提升。收盘漂移不自动视为计划限额违规。",
        "- 全矩阵如实报告，无收益承诺，无参数追加搜索；研究完成未发布，未改实盘。源代码与测试均冻结并校验哈希。",
        "",
    ]
    return "\n".join(lines)


def main(run, output, reference=REFERENCE):
    run, output, reference = Path(run), Path(output), Path(reference)
    if output.exists():
        raise FileExistsError("IMMUTABLE_CADENCE_REPORT")
    contract, sources = validate_sources(run)
    control = verify_frozen_control(run, reference)
    decisions, decision_audit = audit_decisions(run)
    helper = sibling("report_ashare_rank_preserving")
    sibling("report_ashare_mainline_replay").main(
        [run / f"cost{bps}" for bps in COSTS], output / "report"
    )
    ledger_path = output / "report/results.json"
    ledger = json.loads(ledger_path.read_bytes())
    rows = ledger["cases"]
    expected = {(bps, MODEL + "--" + name) for bps in COSTS for name in CASE_SPECS}
    if (
        ledger["status"] != "FULL_REPLAY_LEDGER_VERIFIED"
        or len(rows) != len(expected)
        or {(r["bps"], r["case"]) for r in rows} != expected
        or any(r["sessions"] != SESSIONS for r in rows)
    ):
        raise ValueError("CADENCE_MATRIX_NOT_8_BY_181")
    metrics, equities, changes, protection_counts = {}, {}, [], []
    for row in rows:
        bps, case = row["bps"], row["case"]
        folder = run / f"cost{bps}/bps{bps}"
        equity = pd.read_parquet(folder / case / "equity.parquet")
        fills = pd.read_parquet(folder / case / "fills.parquet")
        holdings = pd.read_parquet(folder / "holdings.parquet")
        counts = (
            holdings.loc[holdings.kind.eq(case)]
            .groupby("date")
            .symbol.nunique()
            .reindex(equity.date, fill_value=0)
        )
        row.update(
            gross_shares_traded=int(fills.quantity.sum()),
            buy_shares=int(fills.loc[fills.side.eq("BUY"), "quantity"].sum()),
            sell_shares=int(fills.loc[fills.side.eq("SELL"), "quantity"].sum()),
            trading_days=int(fills.date.nunique()),
            mean_held_names=float(counts.mean()),
            minimum_held_names=int(counts.min()),
            maximum_held_names=int(counts.max()),
            total_execution_cost=float(fills.fee.sum() + fills.slippage.sum()),
        )
        metrics[bps, case], equities[bps, case] = row, equity
        changes.append(helper.executed_share_changes(fills).assign(bps=bps, case=case))
        group = [d for d in decisions if d["bps"] == bps and d["case"] == case]
        protection_counts.append(
            {
                "bps": bps,
                "case": case,
                "reason_counts": dict(Counter(d["reason"] for d in group if d["reason"])),
                "status_counts": dict(Counter(d["status"] for d in group)),
                "scheduled_sessions": sum(d["scheduled_rebalance"] for d in group),
                "verified_plans": sum(d["status"] == "OPEN_PLAN_VERIFIED" for d in group),
                "off_schedule_retries": sum(d["off_schedule_retry"] for d in group),
                "off_schedule_empty_book_sessions": sum(
                    d["off_schedule_empty_book"] for d in group
                ),
                "frozen_position_sessions": sum(d["frozen_count"] > 0 for d in group),
                "pending_exit_sessions": sum(d["pending_exit_count"] > 0 for d in group),
                **{
                    key: sum(d[key] for d in group)
                    for key in (
                        "optional_membership_exits",
                        "industry_risk_forced_membership_exits",
                        "slot_capacity_forced_membership_exits",
                        "new_membership_entries",
                    )
                },
            }
        )
    comparisons = []
    for bps in COSTS:
        for candidate, baseline, role in comparison_specs():
            candidate, baseline = MODEL + "--" + candidate, MODEL + "--" + baseline
            comparisons.append(
                {
                    "bps": bps,
                    "candidate": candidate,
                    "reference": baseline,
                    "comparison_role": role,
                    **helper.compare_cases(
                        metrics[bps, candidate],
                        metrics[bps, baseline],
                        equities[bps, candidate],
                        equities[bps, baseline],
                        contract["initial_cash"],
                    ),
                }
            )
    payload = {
        "status": "FULL_CADENCE_MATRIX_COMPLETE_NOT_RELEASED",
        "cases": rows,
        "case_specs": CASE_SPECS,
        "paired_comparisons": comparisons,
        "frozen_rank_cap_control_audit": control,
        "independent_decision_audit": decision_audit,
        "sources": sources,
        "reporter_sha256": sha256(Path(__file__)),
        "ledger_report_sha256": sha256(ledger_path),
        "protection_counts_by_case": protection_counts,
        "protection_events": [d for d in decisions if d["reason"]],
        "model_retrained": False,
        "production_modified": False,
        "retrospective_not_pristine_holdout": True,
        "exposure_differences_are_not_alpha": True,
        "paired_intervals_are_not_multiple_comparison_adjusted": True,
        "bounded_two_by_two_no_parameter_expansion": True,
        "trade_band_or_retained_weight_experiments": False,
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    frame = pd.DataFrame(rows)
    for column in ("industry_mean_weights", "decision_reasons", "ledger", "quarters"):
        frame[column] = frame[column].map(json.dumps)
    frame.to_parquet(output / "metrics.parquet", index=False)
    pd.DataFrame(comparisons).assign(
        industry_mean_weight_differences=lambda x: x.industry_mean_weight_differences.map(
            json.dumps
        )
    ).to_parquet(output / "paired-comparisons.parquet", index=False)
    pd.DataFrame(decisions).to_parquet(output / "decision-audit.parquet", index=False)
    pd.concat(changes, ignore_index=True).to_parquet(
        output / "executed-share-changes.parquet", index=False
    )
    (output / "report.md").write_text(markdown_report(payload), encoding="utf-8")
    print(
        json.dumps({"status": payload["status"], "cases": len(rows), "output": str(output)}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    args = parser.parse_args()
    main(args.run, args.output, args.reference)
