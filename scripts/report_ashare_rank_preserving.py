"""Read-only verification of the fixed-prediction, rank-preserving research matrix.

Only the new report directory is written. No model, source run, or live account is changed.
"""

import argparse
import importlib.util
import json
from collections import Counter
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from moneymore.research.ashare_mainline import sha256

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "state/ashare-mainline-20261112"
REFERENCE = BASE / "n3-full-matrix-v4"
MODEL = "doubleensemble_full"
ALLOCATORS = (
    "topk",
    "sector_topk",
    "rank_equal",
    "rank_cap",
    "sector_rank_cap",
    "sector_rank_buffer",
)
COSTS = (10, 20)
SESSIONS = 181


@cache
def sibling(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def daily_returns(frame, initial_cash):
    """Include the first day's return from initial capital, without filling missing days."""
    if (
        len(frame) < 2
        or frame.date.duplicated().any()
        or not frame.date.is_monotonic_increasing
        or not np.isfinite(frame.equity).all()
        or not frame.equity.gt(0).all()
        or not np.isfinite(initial_cash)
        or initial_cash <= 0
    ):
        raise ValueError("INVALID_COMPARISON_EQUITY")
    values = frame.equity.to_numpy(dtype=float)
    return values / np.r_[initial_cash, values[:-1]] - 1


def compare_cases(candidate, reference, candidate_equity, reference_equity, initial_cash):
    """Pure, same-calendar comparison; positive drawdown reduction means less drawdown."""
    if candidate_equity.date.tolist() != reference_equity.date.tolist():
        raise ValueError("PAIRED_CALENDAR_MISMATCH")
    left = daily_returns(candidate_equity, initial_cash)
    right = daily_returns(reference_equity, initial_cash)
    differences = {
        "net_return_difference": "net_return",
        "sharpe_difference": "sharpe",
        "annualized_volatility_difference": "annualized_volatility",
        "mean_stock_exposure_difference": "mean_actual_stock_exposure",
        "minimum_stock_exposure_difference": "minimum_actual_stock_exposure",
        "maximum_stock_exposure_difference": "maximum_actual_stock_exposure",
        "turnover_difference": "annualized_half_gross_turnover",
        "fees_difference": "fees",
        "slippage_difference": "slippage",
        "fills_difference": "fills",
        "mean_largest_industry_weight_difference": "mean_largest_industry_weight",
        "maximum_industry_weight_difference": "maximum_industry_weight",
        "mean_growth_board_weight_difference": "mean_growth_board_weight",
        "gross_shares_traded_difference": "gross_shares_traded",
        "buy_shares_difference": "buy_shares",
        "sell_shares_difference": "sell_shares",
        "trading_days_difference": "trading_days",
        "mean_held_names_difference": "mean_held_names",
    }
    result = {key: float(candidate[field] - reference[field]) for key, field in differences.items()}
    result["drawdown_reduction"] = float(
        reference["maximum_drawdown"] - candidate["maximum_drawdown"]
    )
    result["execution_cost_difference"] = result["fees_difference"] + result["slippage_difference"]
    result["point_estimate_dominates_return_sharpe_drawdown"] = bool(
        candidate["net_return"] >= reference["net_return"]
        and candidate["sharpe"] >= reference["sharpe"]
        and candidate["maximum_drawdown"] <= reference["maximum_drawdown"]
    )
    result["industry_mean_weight_differences"] = {
        industry: float(
            candidate["industry_mean_weights"].get(industry, 0)
            - reference["industry_mean_weights"].get(industry, 0)
        )
        for industry in sorted(
            set(candidate["industry_mean_weights"]) | set(reference["industry_mean_weights"])
        )
    }
    result.update(sibling("finalize_ashare_mainline_n3").paired_intervals(left, right))
    return result


def assert_replay_equal(actual, expected, label):
    """Protect the frozen control including row order, fill keys, and all numeric fields."""
    if list(actual.columns) != list(expected.columns) or actual.shape != expected.shape:
        raise ValueError("FROZEN_TOPK_SCHEMA_OR_COUNT_CHANGED:" + label)
    for column in expected:
        a, b = actual[column], expected[column]
        if pd.api.types.is_numeric_dtype(b):
            equal = (
                pd.api.types.is_numeric_dtype(a)
                and np.isfinite(a).all()
                and np.isfinite(b).all()
                and np.allclose(a, b, atol=1e-8, rtol=1e-10)
            )
        else:
            equal = a.tolist() == b.tolist()
        if not equal:
            raise ValueError("FROZEN_TOPK_REPLAY_CHANGED:" + label + ":" + column)


def verify_frozen_controls(run, reference):
    records = []
    for bps in COSTS:
        case = MODEL + "--topk"
        for name in ("equity.parquet", "fills.parquet"):
            current = run / f"cost{bps}/bps{bps}" / case / name
            frozen = reference / f"cost{bps}/bps{bps}" / case / name
            assert_replay_equal(pd.read_parquet(current), pd.read_parquet(frozen), f"{bps}:{name}")
            records.append(
                {
                    "bps": bps,
                    "artifact": name,
                    "current_sha256": sha256(current),
                    "frozen_path": str(frozen.resolve()),
                    "frozen_sha256": sha256(frozen),
                }
            )
    return {"passed": True, "numeric_atol": 1e-8, "numeric_rtol": 1e-10, "artifacts": records}


def assert_exact_replay_equal(actual, expected, label):
    """Stricter explanatory-control regression check: no numeric tolerance at all."""
    try:
        pd.testing.assert_frame_equal(actual, expected, check_exact=True, rtol=0, atol=0)
    except AssertionError as error:
        raise ValueError("ORIGINAL_CASE_NOT_EXACTLY_REPRODUCED:" + label) from error
    for column in actual.select_dtypes(include="number"):
        if not np.isfinite(actual[column]).all():
            raise ValueError("NONFINITE_ORIGINAL_CASE_ARTIFACT:" + label)


def audit_original_cases(run, prior):
    """After both matrices complete, write only an immutable audit under the new run."""
    run, prior = Path(run), Path(prior)
    output = run / "original-five-v1-equivalence.json"
    if output.exists():
        raise FileExistsError("IMMUTABLE_ORIGINAL_CASE_EQUIVALENCE_AUDIT")
    runtimes = {}
    for label, root in (("current", run), ("prior", prior)):
        for bps in COSTS:
            path = root / f"cost{bps}/runtime.json"
            runtime = json.loads(path.read_bytes())
            if (
                runtime["status"] != "COMPLETE"
                or runtime["bps"] != bps
                or runtime.get("smoke", False)
            ):
                raise ValueError("BOTH_FULL_MATRICES_MUST_BE_COMPLETE_BEFORE_COMPARISON")
            runtimes[f"{label}_cost{bps}"] = sha256(path)
    records = []
    for bps in COSTS:
        for allocator in (
            "topk",
            "sector_topk",
            "rank_cap",
            "sector_rank_cap",
            "sector_rank_buffer",
        ):
            case = MODEL + "--" + allocator
            for name in ("equity.parquet", "fills.parquet"):
                current_path = run / f"cost{bps}/bps{bps}" / case / name
                prior_path = prior / f"cost{bps}/bps{bps}" / case / name
                current, previous = pd.read_parquet(current_path), pd.read_parquet(prior_path)
                assert_exact_replay_equal(current, previous, f"{bps}:{case}:{name}")
                if name == "equity.parquet" and (
                    len(current) != SESSIONS
                    or current.date.duplicated().any()
                    or not current.date.is_monotonic_increasing
                ):
                    raise ValueError("INCOMPLETE_ORIGINAL_CASE_EQUITY")
                current_hash, prior_hash = sha256(current_path), sha256(prior_path)
                records.append(
                    {
                        "bps": bps,
                        "case": case,
                        "artifact": name,
                        "rows": len(current),
                        "columns": current.columns.tolist(),
                        "date_start": current.date.min() if len(current) else None,
                        "date_end": current.date.max() if len(current) else None,
                        "current_path": str(current_path.resolve()),
                        "prior_path": str(prior_path.resolve()),
                        "current_sha256": current_hash,
                        "prior_sha256": prior_hash,
                        "binary_identical": current_hash == prior_hash,
                        "schema_keys_order_and_values_exactly_equal": True,
                    }
                )
    payload = {
        "status": "ORIGINAL_FIVE_CASES_EXACTLY_REPRODUCED",
        "passed": True,
        "accounts": len(COSTS) * 5,
        "artifacts": records,
        "artifact_count": len(records),
        "equity_rows": sum(r["rows"] for r in records if r["artifact"] == "equity.parquet"),
        "fill_rows": sum(r["rows"] for r in records if r["artifact"] == "fills.parquet"),
        "numeric_atol": 0,
        "numeric_rtol": 0,
        "all_artifacts_binary_identical": all(r["binary_identical"] for r in records),
        "runtime_sha256": runtimes,
        "audit_script_sha256": sha256(Path(__file__)),
        "new_rank_equal_case_not_part_of_regression_comparison": True,
        "source_runs_modified": False,
        "production_modified": False,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def audit_allocation(allocation, allocator):
    """Recompute caps, sector sums and slot accounting independently of reported status."""
    if allocation.get("status") not in ("OK", "INSUFFICIENT_CAPACITY"):
        raise ValueError("INVALID_ALLOCATION_AUDIT_STATUS")
    fields = (
        "benchmark_industry_weights",
        "frozen_industry_weights",
        "selected_industry_weights",
        "total_industry_weights",
        "industry_weight_caps",
    )
    for field in fields:
        if any(not np.isfinite(v) or v < 0 for v in allocation[field].values()):
            raise ValueError("INVALID_ALLOCATION_WEIGHT")
    sectors = set().union(*(allocation[field] for field in fields))
    tol = 1e-8
    active_limit = 1.0 if allocator == "rank_equal" else 0.08
    if not np.isclose(allocation["industry_active_limit"], active_limit, atol=tol, rtol=0):
        raise ValueError("ALLOCATION_CAP_CONTRACT_CHANGED")
    for sector in sectors:
        benchmark = allocation["benchmark_industry_weights"].get(sector, 0)
        frozen = allocation["frozen_industry_weights"].get(sector, 0)
        selected = allocation["selected_industry_weights"].get(sector, 0)
        total = allocation["total_industry_weights"].get(sector, 0)
        cap = allocation["industry_weight_caps"].get(sector, np.nan)
        if (
            not np.isfinite(cap)
            or abs(cap - benchmark - active_limit) > tol
            or abs(total - selected - frozen) > tol
            or total > cap + tol
        ):
            raise ValueError("INDEPENDENT_ALLOCATION_INDUSTRY_CAP_FAILED")
    selected = sum(allocation["selected_industry_weights"].values())
    frozen = sum(allocation["frozen_industry_weights"].values())
    new, retained = set(allocation["new_symbols"]), set(allocation["retained_symbols"])
    if (
        new & retained
        or len(new | retained) != allocation["selected_count"]
        or abs(selected - allocation["selected_count"] * allocation["slot_weight"]) > tol
        or abs(selected - allocation["selected_weight"]) > tol
        or abs(frozen - allocation["frozen_weight"]) > tol
        or abs(selected + frozen - allocation["total_weight"]) > tol
        or selected + frozen > allocation["target_exposure"] + tol
        or not np.isfinite(allocation["target_exposure"])
        or not 0 <= allocation["target_exposure"] <= 0.995 + tol
        or not np.isfinite(allocation["slot_weight"])
        or allocation["slot_weight"] <= 0
        or allocation["requested_slots"] != 50
        or abs(
            allocation["requested_slots"] * allocation["slot_weight"]
            - allocation["target_exposure"]
        )
        > tol
    ):
        raise ValueError("INDEPENDENT_ALLOCATION_BUDGET_FAILED")
    if allocator == "sector_rank_buffer" and (
        allocation["max_optional_exits"] != 5 or len(allocation["optional_exits"]) > 5
    ):
        raise ValueError("OPTIONAL_MEMBERSHIP_EXIT_BUFFER_FAILED")


def audit_decision(decision, day, day_fills, allocator):
    """Independent checks of causal timestamps, protected holdings and exit-only days."""
    if decision["signal_date"] >= day or decision["execution_date"] != day:
        raise ValueError("DECISION_TIMELINE_MISMATCH")
    if decision.get("next_open_used_for_selection", False):
        raise ValueError("NEXT_OPEN_USED_FOR_SELECTION")
    frozen = set(decision.get("frozen_symbols", []))
    if decision["status"] == "OPEN_PLAN_VERIFIED" and set(day_fills.symbol) & frozen:
        raise ValueError("FROZEN_POSITION_WAS_TRADED")
    if decision["status"] == "MANDATORY_EXITS_ONLY" and (
        not day_fills.side.eq("SELL").all()
        or not set(day_fills.symbol).issubset(decision["exit_symbols"])
    ):
        raise ValueError("OPTIONAL_TRADE_ON_EXIT_ONLY_DAY")
    constrained = allocator in ("rank_equal", "rank_cap", "sector_rank_cap", "sector_rank_buffer")
    if decision["status"] == "OPEN_PLAN_VERIFIED":
        if decision["risk_audit"]["risk_end"] != decision["signal_date"]:
            raise ValueError("RISK_TIMELINE_MISMATCH")
        postcheck = decision.get("open_postcheck")
        if constrained:
            if not isinstance(postcheck, dict) or postcheck.get("passed") is not True:
                raise ValueError("OPEN_INTEGER_RISK_POSTCHECK_MISSING_OR_FAILED")
            values = [
                postcheck[field]
                for field in (
                    "actual_stock_exposure",
                    "maximum_stock_weight",
                    "maximum_industry_violation",
                    "stock_limit",
                )
            ]
            if (
                not np.isfinite(values).all()
                or not 0 <= postcheck["actual_stock_exposure"] <= 1 + 1e-8
                or not 0 <= postcheck["maximum_stock_weight"] <= 0.05 + 1e-8
                or postcheck["maximum_industry_violation"] > 1e-8
                or abs(postcheck["stock_limit"] - 0.05) > 1e-8
            ):
                raise ValueError("INDEPENDENT_OPEN_INTEGER_RISK_POSTCHECK_FAILED")
            audit_allocation(decision["allocation"], allocator)
            if "industry_weights" in postcheck or "industry_limits" in postcheck:
                weights, limits = postcheck["industry_weights"], postcheck["industry_limits"]
                benchmark = decision["allocation"]["benchmark_industry_weights"]
                sectors = set(weights) | set(limits) | set(benchmark)
                active_limit = 1.0 if allocator == "rank_equal" else 0.10
                if (
                    not all(np.isfinite(w) and w >= 0 for w in weights.values())
                    or abs(sum(weights.values()) - postcheck["actual_stock_exposure"]) > 1e-8
                    or any(
                        not np.isfinite(limits.get(sector, np.nan))
                        or abs(limits[sector] - benchmark.get(sector, 0) - active_limit) > 1e-8
                        or weights.get(sector, 0) > limits[sector] + 1e-8
                        for sector in sectors
                    )
                ):
                    raise ValueError("INDEPENDENT_OPEN_INDUSTRY_WEIGHTS_FAILED")
        for field in ("target_quantities", "executable_quantities"):
            quantities = decision.get(field, {})
            if any(not np.isfinite(q) or q < 0 or int(q) != q for q in quantities.values()):
                raise ValueError("NONINTEGER_OR_NEGATIVE_SHARE_PLAN")
    allocation = decision.get("allocation") or {}
    return {
        "date": day,
        "status": decision["status"],
        "reason": decision.get("reason"),
        "frozen_count": len(frozen),
        "pending_exit_count": len(decision.get("pending_exit_symbols", [])),
        "missing_symbols": decision.get("missing_symbols", []),
        "open_postcheck_verified": constrained and decision["status"] == "OPEN_PLAN_VERIFIED",
        "allocation_caps_verified": constrained and decision["status"] == "OPEN_PLAN_VERIFIED",
        "allocation_status": allocation.get("status"),
        "optional_membership_exits": len(allocation.get("optional_exits", [])),
        "industry_risk_forced_membership_exits": len(allocation.get("risk_forced_exits", [])),
        "slot_capacity_forced_membership_exits": len(
            allocation.get("slot_capacity_forced_exits", [])
        ),
        "new_membership_entries": len(allocation.get("new_symbols", [])),
    }


def audit_decisions(run, contract):
    records = []
    for bps in COSTS:
        for allocator in contract["allocators"]:
            case = MODEL + "--" + allocator
            folder = run / f"cost{bps}/bps{bps}" / case
            equity = pd.read_parquet(folder / "equity.parquet")
            fills = pd.read_parquet(folder / "fills.parquet")
            paths = sorted((run / f"cost{bps}/decisions" / case).glob("*.json"))
            if len(paths) != SESSIONS or [p.stem for p in paths] != equity.date.tolist():
                raise ValueError("INCOMPLETE_DECISION_CALENDAR")
            for path in paths:
                decision = json.loads(path.read_bytes())
                item = audit_decision(
                    decision, path.stem, fills.loc[fills.date.eq(path.stem)], allocator
                )
                records.append({"bps": bps, "case": case, **item})
    if len(records) != len(COSTS) * len(ALLOCATORS) * SESSIONS:
        raise ValueError("INCOMPLETE_DECISION_MATRIX")
    return records, {
        "passed": True,
        "decisions": len(records),
        "verified_plan_frozen_positions_never_traded": True,
        "exit_only_sessions_have_only_mandatory_sells": True,
        "verified_open_integer_plans": sum(r["open_postcheck_verified"] for r in records),
        "verified_allocation_industry_caps": sum(r["allocation_caps_verified"] for r in records),
    }


def executed_share_changes(fills):
    """Executed changes only: corporate-action share entitlements are not trades."""
    values = fills.copy()
    if not values.side.isin(["BUY", "SELL"]).all() or not values.quantity.gt(0).all():
        raise ValueError("INVALID_EXECUTED_SHARE_CHANGE")
    values["buy_shares"] = values.quantity.where(values.side.eq("BUY"), 0)
    values["sell_shares"] = values.quantity.where(values.side.eq("SELL"), 0)
    values["signed_executed_shares"] = values.buy_shares - values.sell_shares
    return values.groupby(["date", "symbol"], as_index=False)[
        ["buy_shares", "sell_shares", "signed_executed_shares", "fee", "slippage", "notional"]
    ].sum()


def validate_sources(run):
    contract = json.loads((run / "cost10/contract.json").read_bytes())
    if (
        contract["models"] != [MODEL]
        or len(contract["allocators"]) != len(ALLOCATORS)
        or set(contract["allocators"]) != set(ALLOCATORS)
        or contract["initial_cash"] != 10_000_000
    ):
        raise ValueError("RANK_PRESERVING_CONTRACT_MISMATCH")
    manifests, hashes = {}, None
    for bps in COSTS:
        child = run / f"cost{bps}"
        if sha256(child / "contract.json") != sha256(run / "cost10/contract.json"):
            raise ValueError("MIXED_EXPERIMENT_CONTRACTS")
        runtime = json.loads((child / "runtime.json").read_bytes())
        if runtime["status"] != "COMPLETE" or runtime.get("smoke", False) or runtime["bps"] != bps:
            raise ValueError("FULL_RANK_PRESERVING_RUN_NOT_COMPLETE")
        if not runtime.get("source_hashes"):
            raise ValueError("MISSING_SOURCE_SNAPSHOT_HASHES")
        if hashes is not None and hashes != runtime["source_hashes"]:
            raise ValueError("SOURCE_CHANGED_BETWEEN_COSTS")
        hashes = runtime["source_hashes"]
        for relative, digest in hashes.items():
            if sha256(child / "sources" / relative) != digest:
                raise ValueError("SOURCE_SNAPSHOT_CORRUPTED")
        manifests[str(bps)] = {
            "runtime_sha256": sha256(child / "runtime.json"),
            "contract_sha256": sha256(child / "contract.json"),
            "source_hashes": runtime.get("source_hashes", {}),
        }
    return contract, manifests


def markdown_report(payload):
    lines = [
        "# 保留排序信号的组合研究：完整回放，未发布",
        "",
        "固定 DoubleEnsemble 全特征预测；6种组合器 × 2档成本 × 181交易日。初始资金1000万元。",
        "单边滑点10/20bps，手续费另计。无风险利率0，年化252日。未重新训练、未修改实盘。",
        "原始 TopK 与已冻结 N3 对照的逐日净值、逐笔成交在10/20bps均复现一致。",
        "新增rank_equal仅作机制解释对照：相同原始排序、50槽位与等权全量刷新，行业主动上限设为1.0而不构成实际限制。",
        "rank_equal−TopK反映刷新与权重重置的组合影响；rank_cap−rank_equal才隔离行业限额增量，不能将前者误归因于行业约束。",
        "这是发现归因混杂后补充的解释性对照，不是预先未见数据的检验，也不是收益驱动的参数搜索。",
        "",
    ]
    for bps in COSTS:
        lines += [
            f"## {bps}bps 完整结果",
            "",
            "|组合器|净收益|夏普|最大回撤|年化波动|平均实际仓位|最小／最大仓位|年化换手¹|平均最大行业权重|最高行业权重|",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        rows = [r for r in payload["cases"] if r["bps"] == bps]
        for row in rows:
            allocator = row["case"].split("--", 1)[1]
            lines.append(
                f"|{allocator}|{row['net_return']:.2%}|{row['sharpe']:.3f}|"
                f"{row['maximum_drawdown']:.2%}|{row['annualized_volatility']:.2%}|"
                f"{row['mean_actual_stock_exposure']:.2%}|"
                f"{row['minimum_actual_stock_exposure']:.2%}／{row['maximum_actual_stock_exposure']:.2%}|"
                f"{row['annualized_half_gross_turnover']:.2f}|"
                f"{row['mean_largest_industry_weight']:.2%}|{row['maximum_industry_weight']:.2%}|"
            )
        lines += [
            "",
            "|组合器|一季度|二季度|三季度|手续费元|滑点损耗元|成交笔数|买入股数|卖出股数|有成交日数|平均持股数|",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            quarter = [row["quarters"][f"2026Q{i}"]["net_return"] for i in (1, 2, 3)]
            lines.append(
                f"|{row['case'].split('--', 1)[1]}|{quarter[0]:.2%}|{quarter[1]:.2%}|"
                f"{quarter[2]:.2%}|{row['fees']:,.2f}|{row['slippage']:,.2f}|{row['fills']}|"
                f"{row['buy_shares']:,.0f}|{row['sell_shares']:,.0f}|{row['trading_days']}|"
                f"{row['mean_held_names']:.1f}|"
            )
        lines.append("")
    for category, title in (("relative_to_topk", "相对原始TopK"), ("stepwise", "分步机制对照")):
        lines += [
            "## " + title,
            "",
            "|成本|候选 − 对照|收益差|夏普差|回撤减少|平均仓位差|年化换手差|执行成本差元|收益差95%区间|夏普差95%区间|",
            "|---:|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
        for row in payload["paired_comparisons"]:
            if category not in row["comparison_roles"]:
                continue
            lo, hi = row["net_return_difference_ci95"]
            slo, shi = row["sharpe_difference_ci95"]
            candidate, reference = (row[k].split("--", 1)[1] for k in ("candidate", "reference"))
            lines.append(
                f"|{row['bps']}|{candidate} − {reference}|{row['net_return_difference']:+.2%}|"
                f"{row['sharpe_difference']:+.3f}|{row['drawdown_reduction']:+.2%}|"
                f"{row['mean_stock_exposure_difference']:+.2%}|{row['turnover_difference']:+.2f}|"
                f"{row['execution_cost_difference']:+,.2f}|[{lo:.2%}, {hi:.2%}]|[{slo:.3f}, {shi:.3f}]|"
            )
        lines.append("")
    lines += [
        "## 保护与成员调整计数",
        "",
        "|成本|组合器|保护记录|冻结持仓日|待退出日|主动换出成员|行业修复强制换出|容量强制换出|",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["protection_counts_by_case"]:
        lines.append(
            f"|{row['bps']}|{row['case'].split('--', 1)[1]}|{sum(row['reason_counts'].values())}|"
            f"{row['frozen_position_sessions']}|{row['pending_exit_sessions']}|"
            f"{row['optional_membership_exits']}|{row['industry_risk_forced_membership_exits']}|"
            f"{row['slot_capacity_forced_membership_exits']}|"
        )
    lines += [
        "",
        (
            "成员调整来自行业限额分配器的审计：TopK两组未生成这项分配审计，其0值不代表无换股。"
            "成员换出数量不等同于成交笔数或实际卖出股数，强制风险修复也不计入主动drop5预算。"
        ),
        "",
    ]
    lines += [
        "## 审计与解释边界",
        "",
        "- 12账户独立账本、现金、逐证券数量、订单与成交导出核验通过；2172条决策时间线核验通过。",
        (
            f"- 保护记录 {len(payload['protection_events'])} 条；涉及冻结持仓的决策 "
            f"{payload['isolated_frozen_position_sessions']} 条；只做强制退出 "
            f"{payload['mandatory_exit_only_sessions']} 条；尚不可交易的退出 "
            f"{payload['pending_untradable_exit_sessions']} 条。"
        ),
        "- 各组合器保护原因、行业均值权重、分季度指标、逐日逐股成交股数变化均另存机器可读结果；成交股数不含送转股权益。",
        "- ¹年化换手＝每日买卖成交名义额／前日净值／2之和×252/181，包含初始建仓。不同价格股票的股数不能直接解释成资金规模。",
        "- 区间采用同日配对20日环形区块、2000次重采样、固定种子42；未做多重比较校正。历史已反复查看，不是全新留出集。",
        "- 行业使用截至当日可见的分类；平均最大行业权重与最大单日行业权重分别列出。收盘价格漂移不等于分配时违反行业约束。",
        "- 实际仓位、整手残留现金和成交成本均可能不同；较低回撤、波动或换手不能自动解释为排序信号提升。",
        "- 点估计和区间均如实报告，不承诺改进，不据此发布或替换现有模型；未触发新的训练、参数搜索或实盘操作。",
        "",
    ]
    return "\n".join(lines)


def main(run, output, reference=REFERENCE):
    run, output, reference = Path(run), Path(output), Path(reference)
    if output.exists():
        raise FileExistsError("IMMUTABLE_RANK_PRESERVING_REPORT")
    contract, manifests = validate_sources(run)
    control_audit = verify_frozen_controls(run, reference)
    decisions, decision_audit = audit_decisions(run, contract)
    sibling("report_ashare_mainline_replay").main(
        [run / f"cost{bps}" for bps in COSTS], output / "report"
    )
    baseline_path = output / "report/results.json"
    baseline = json.loads(baseline_path.read_bytes())
    rows = baseline["cases"]
    if (
        baseline["status"] != "FULL_REPLAY_LEDGER_VERIFIED"
        or len(rows) != len(COSTS) * len(ALLOCATORS)
        or any(r["sessions"] != SESSIONS for r in rows)
    ):
        raise ValueError("RANK_PRESERVING_MATRIX_NOT_12_BY_181")
    metrics, equity_frames, share_changes = {}, {}, []
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
        changes = executed_share_changes(fills)
        share_changes.append(changes.assign(bps=bps, case=case))
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
        metrics[bps, case] = row
        equity_frames[bps, case] = equity
    comparison_pairs = {
        (allocator, "topk"): ["relative_to_topk"] for allocator in ALLOCATORS if allocator != "topk"
    }
    for pair in (
        ("sector_topk", "topk"),
        ("rank_equal", "topk"),
        ("rank_cap", "rank_equal"),
        ("sector_rank_cap", "rank_cap"),
        ("sector_rank_buffer", "sector_rank_cap"),
    ):
        comparison_pairs.setdefault(pair, []).append("stepwise")
    comparisons = []
    for bps in COSTS:
        for (candidate, control), roles in comparison_pairs.items():
            candidate, control = MODEL + "--" + candidate, MODEL + "--" + control
            comparisons.append(
                {
                    "bps": bps,
                    "candidate": candidate,
                    "reference": control,
                    "comparison_roles": roles,
                    **compare_cases(
                        metrics[bps, candidate],
                        metrics[bps, control],
                        equity_frames[bps, candidate],
                        equity_frames[bps, control],
                        contract["initial_cash"],
                    ),
                }
            )
    protection_counts = []
    for bps in COSTS:
        for allocator in ALLOCATORS:
            case = MODEL + "--" + allocator
            group = [r for r in decisions if r["bps"] == bps and r["case"] == case]
            protection_counts.append(
                {
                    "bps": bps,
                    "case": case,
                    "reason_counts": dict(Counter(r["reason"] for r in group if r["reason"])),
                    "status_counts": dict(Counter(r["status"] for r in group)),
                    "frozen_position_sessions": sum(r["frozen_count"] > 0 for r in group),
                    "pending_exit_sessions": sum(r["pending_exit_count"] > 0 for r in group),
                    **{
                        key: sum(r[key] for r in group)
                        for key in (
                            "optional_membership_exits",
                            "industry_risk_forced_membership_exits",
                            "slot_capacity_forced_membership_exits",
                            "new_membership_entries",
                        )
                    },
                }
            )
    payload = {
        "status": "FULL_MATRIX_COMPLETE_NOT_RELEASED",
        "reporter_sha256": sha256(Path(__file__)),
        "ledger_report_sha256": sha256(baseline_path),
        "cases": rows,
        "paired_comparisons": comparisons,
        "frozen_n3_topk_control_audit": control_audit,
        "independent_decision_audit": decision_audit,
        "sources": manifests,
        "protection_events": [d for d in decisions if d["reason"]],
        "protection_counts_by_case": protection_counts,
        "isolated_frozen_position_sessions": sum(d["frozen_count"] > 0 for d in decisions),
        "mandatory_exit_only_sessions": sum(
            d["status"] == "MANDATORY_EXITS_ONLY" for d in decisions
        ),
        "pending_untradable_exit_sessions": sum(d["pending_exit_count"] > 0 for d in decisions),
        "model_retrained": False,
        "production_modified": False,
        "retrospective_not_pristine_holdout": True,
        "exposure_differences_are_not_alpha": True,
        "paired_intervals_are_not_multiple_comparison_adjusted": True,
        "executed_share_changes_exclude_corporate_actions": True,
        "rank_equal_added_to_separate_refresh_from_industry_cap": True,
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    metrics_frame = pd.DataFrame(rows)
    for column in ("industry_mean_weights", "decision_reasons", "ledger", "quarters"):
        metrics_frame[column] = metrics_frame[column].map(json.dumps)
    metrics_frame.to_parquet(output / "metrics.parquet", index=False)
    pd.DataFrame(comparisons).assign(
        industry_mean_weight_differences=lambda x: x.industry_mean_weight_differences.map(
            json.dumps
        )
    ).to_parquet(output / "paired-comparisons.parquet", index=False)
    pd.DataFrame(decisions).to_parquet(output / "decision-audit.parquet", index=False)
    pd.concat(share_changes, ignore_index=True).to_parquet(
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
    arguments = parser.parse_args()
    main(arguments.run, arguments.output, arguments.reference)
