"""Close only N3: fixed-prediction execution comparisons, not live promotion."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from moneymore.research.ashare_mainline import sha256


def paired_intervals(left, right, block=20, samples=2000):
    """Paired circular blocks; uncertainty diagnosis, not an untouched holdout test."""
    left, right = np.asarray(left), np.asarray(right)
    if len(left) != len(right) or not np.isfinite([left, right]).all():
        raise ValueError("INVALID_PAIRED_RETURNS")
    rng = np.random.default_rng(42)
    n = len(left)
    starts = rng.integers(n, size=(samples, (n + block - 1) // block))
    indices = ((starts[..., None] + np.arange(block)) % n).reshape(samples, -1)[:, :n]
    a, b = left[indices], right[indices]
    difference = np.prod(1 + a, axis=1) - np.prod(1 + b, axis=1)
    sharpe_a = a.mean(axis=1) / np.maximum(a.std(axis=1, ddof=1), 1e-15) * np.sqrt(252)
    sharpe_b = b.mean(axis=1) / np.maximum(b.std(axis=1, ddof=1), 1e-15) * np.sqrt(252)
    return {
        "net_return_difference_ci95": np.quantile(difference, [0.025, 0.975]).tolist(),
        "sharpe_difference_ci95": np.quantile(sharpe_a - sharpe_b, [0.025, 0.975]).tolist(),
        "block_sessions": block,
        "bootstrap_samples": samples,
        "seed": 42,
    }


def calibration_diagnostics(records):
    frame = pd.DataFrame(records)
    if (
        len(frame) != 36
        or frame.duplicated(["case", "month"]).any()
        or (frame.last_label_end >= frame.cutoff_exclusive).any()
        or not np.allclose(
            frame.beta_conservative,
            np.maximum(0, frame.beta - frame.standard_error_block20),
            atol=1e-12,
            rtol=0,
        )
    ):
        raise ValueError("CALIBRATION_DIAGNOSTIC_CONTRACT_FAILED")
    return {
        "months_total": len(frame),
        "zero_expected_return_months": int(frame.beta_conservative.eq(0).sum()),
        "by_model": {
            model: {
                "zero_months": g.loc[g.beta_conservative.eq(0), "month"].tolist(),
                "nonzero_months": g.loc[
                    g.beta_conservative.gt(0), ["month", "beta_conservative"]
                ].to_dict("records"),
                "negative_raw_beta_months": int(g.beta.lt(0).sum()),
            }
            for model, g in frame.groupby("case")
        },
        "interpretation": (
            "Zero mu means risk/cost allocation, not model-informed stock selection; "
            "positive RankIC does not establish a positive linear return calibration."
        ),
    }


def audit_decisions(run, contract, bps_values=(10, 20)):
    rows, native_count, cache_count = [], 0, 0
    for bps in bps_values:
        for model in contract["models"]:
            for allocator in contract["allocators"]:
                case = model + "--" + allocator
                fills = pd.read_parquet(run / f"cost{bps}/bps{bps}" / case / "fills.parquet")
                filled = set(zip(fills.date, fills.symbol))
                for path in sorted((run / f"cost{bps}/decisions" / case).glob("*.json")):
                    d = json.loads(path.read_bytes())
                    if d["signal_date"] >= path.stem or d["execution_date"] != path.stem:
                        raise ValueError("DECISION_TIMELINE_MISMATCH")
                    frozen = d.get("frozen_symbols", [])
                    if d["status"] == "MANDATORY_EXITS_ONLY":
                        day_fills = fills.loc[fills.date.eq(path.stem)]
                        if not day_fills.side.eq("SELL").all() or not set(
                            day_fills.symbol
                        ).issubset(d["exit_symbols"]):
                            raise ValueError("OPTIONAL_TRADE_ON_EXIT_ONLY_DAY")
                    if d["status"] == "OPEN_PLAN_VERIFIED":
                        if any((path.stem, symbol) in filled for symbol in frozen):
                            raise ValueError("FROZEN_POSITION_WAS_TRADED")
                        if d["risk_audit"]["risk_end"] != d["signal_date"]:
                            raise ValueError("RISK_TIMELINE_MISMATCH")
                        if allocator != "topk":
                            native = run / f"cost{bps}/native" / case / path.stem
                            result = d["continuous"]
                            if result.get("cache_hit"):
                                cache_decision = (
                                    Path(result["cache_source"]).parents[2]
                                    / "decisions"
                                    / case
                                    / (path.stem + ".json")
                                )
                                if sha256(cache_decision) != result["cache_source_decision_sha256"]:
                                    raise ValueError("CACHE_SOURCE_AUDIT_CHANGED")
                                cache_count += 1
                            if (
                                sha256(native / "request.json") != result["input_sha256"]
                                or sha256(native / "request.npz") != result["covariance_sha256"]
                            ):
                                raise ValueError("NATIVE_INPUT_ARTIFACT_MISMATCH")
                            request = json.loads((native / "request.json").read_bytes())
                            symbols = request["symbols"]
                            initial = np.asarray(request["weights"][:-1])
                            weights = np.array([result["weights"][s] for s in symbols])
                            exposure = np.asarray(request["industry_exposure"])
                            if (
                                abs(weights.sum() - request["stock_exposure"]) > 1e-5
                                or (weights < np.asarray(request["lower_weights"]) - 1e-5).any()
                                or (weights > np.asarray(request["upper_weights"]) + 1e-5).any()
                                or (
                                    exposure.T @ weights
                                    > np.asarray(request["industry_limits"]) + 1e-5
                                ).any()
                                or np.abs(weights - initial).sum() / 2
                                > request["turnover_limit"] + 1e-5
                            ):
                                raise ValueError("INDEPENDENT_CONTINUOUS_CONSTRAINT_CHECK_FAILED")
                            for symbol in frozen:
                                i = symbols.index(symbol)
                                if (
                                    abs(request["lower_weights"][i] - initial[i]) > 1e-8
                                    or abs(request["upper_weights"][i] - initial[i]) > 1e-8
                                ):
                                    raise ValueError("FROZEN_RISK_CONSTRAINT_NOT_PRESERVED")
                            native_count += 1
                    rows.append(
                        {
                            "bps": bps,
                            "case": case,
                            "date": path.stem,
                            "status": d["status"],
                            "reason": d.get("reason"),
                            "frozen_count": len(frozen),
                            "pending_exit_count": len(d.get("pending_exit_symbols", [])),
                            "missing_symbols": d.get("missing_symbols", []),
                        }
                    )
    if len(rows) != len(bps_values) * len(contract["models"]) * len(contract["allocators"]) * 181:
        raise ValueError("INCOMPLETE_DECISION_MATRIX")
    return rows, {
        "passed": True,
        "decisions": len(rows),
        "native_constraints_verified": native_count,
        "exact_input_cache_reuses": cache_count,
        "frozen_positions_never_traded": True,
        "native_input_hashes_verified": True,
    }


def main(run, output):
    if output.exists():
        raise FileExistsError("IMMUTABLE_N3_FINAL")
    progress = json.loads((run / "progress.json").read_bytes())
    report_path = run / "report/results.json"
    report = json.loads(report_path.read_bytes())
    if progress["status"] != "COMPLETE" or report["status"] != "FULL_REPLAY_LEDGER_VERIFIED":
        raise ValueError("N3_MATRIX_NOT_COMPLETE")
    metrics = pd.DataFrame(report["cases"]).set_index(["bps", "case"])
    if len(metrics) != 24 or not metrics.sessions.eq(181).all():
        raise ValueError("N3_MATRIX_NOT_24_BY_181")
    contract = json.loads((run / "cost10/contract.json").read_bytes())
    n2 = run.parent / "n2-summary-v1"
    n2_manifest = json.loads((n2 / "results.json").read_bytes())
    if sha256(n2 / "calibration.json") != n2_manifest["hashes"]["calibration.json"]:
        raise ValueError("CALIBRATION_SOURCE_CHANGED")
    calibration = calibration_diagnostics(json.loads((n2 / "calibration.json").read_bytes()))
    source_hashes = None
    for bps in (10, 20):
        provenance = json.loads((run / f"cost{bps}/provenance.json").read_bytes())
        if provenance["n2_results_sha256"] != sha256(n2 / "results.json"):
            raise ValueError("N2_SOURCE_CHANGED_BETWEEN_COSTS")
        runtime = json.loads((run / f"cost{bps}/runtime.json").read_bytes())
        if runtime["status"] != "COMPLETE":
            raise ValueError("N3_CHILD_NOT_COMPLETE")
        if source_hashes is not None and runtime["source_hashes"] != source_hashes:
            raise ValueError("SOURCE_CHANGED_BETWEEN_COSTS")
        source_hashes = runtime["source_hashes"]
        for relative, digest in source_hashes.items():
            if sha256(run / f"cost{bps}/sources" / relative) != digest:
                raise ValueError("SOURCE_SNAPSHOT_CORRUPTED")
    comparisons = []
    for bps in (10, 20):
        for model in contract["models"]:
            for reference in ("topk", "cvx_risk"):
                candidate, control = model + "--cvx_cost", model + "--" + reference
                a, b = metrics.loc[bps, candidate], metrics.loc[bps, control]
                series = []
                for case in (candidate, control):
                    frame = pd.read_parquet(run / f"cost{bps}/bps{bps}" / case / "equity.parquet")
                    value = frame.equity.to_numpy()
                    series.append(value / np.r_[contract["initial_cash"], value[:-1]] - 1)
                comparisons.append(
                    {
                        "bps": bps,
                        "candidate": candidate,
                        "reference": control,
                        "net_return_difference": float(a.net_return - b.net_return),
                        "sharpe_difference": float(a.sharpe - b.sharpe),
                        "drawdown_reduction": float(b.maximum_drawdown - a.maximum_drawdown),
                        "mean_stock_exposure_difference": float(
                            a.mean_actual_stock_exposure - b.mean_actual_stock_exposure
                        ),
                        "turnover_difference": float(
                            a.annualized_half_gross_turnover - b.annualized_half_gross_turnover
                        ),
                        "point_estimate_dominates_return_sharpe_drawdown": bool(
                            a.net_return >= b.net_return
                            and a.sharpe >= b.sharpe
                            and a.maximum_drawdown <= b.maximum_drawdown
                        ),
                        **paired_intervals(*series),
                    }
                )
    pareto = {}
    for bps in (10, 20):
        frame = metrics.loc[bps]

        def dominates(a, b):
            x = np.array(
                [a.net_return, a.sharpe, -a.maximum_drawdown, -a.annualized_half_gross_turnover]
            )
            y = np.array(
                [b.net_return, b.sharpe, -b.maximum_drawdown, -b.annualized_half_gross_turnover]
            )
            return (x >= y).all() and (x > y).any()

        pareto[str(bps)] = [
            name
            for name, row in frame.iterrows()
            if not any(
                dominates(other, row)
                for other_name, other in frame.iterrows()
                if name != other_name
            )
        ]
    decisions, decision_audit = audit_decisions(run, contract)
    output.mkdir(parents=True)
    payload = {
        "status": "N3_RESEARCH_COMPLETE_NOT_RELEASED",
        "finalizer_sha256": sha256(Path(__file__)),
        "report_sha256": sha256(report_path),
        "cases": report["cases"],
        "independent_decision_audit": decision_audit,
        "paired_comparisons": comparisons,
        "calibration_diagnostics": calibration,
        "pareto_four_objectives": pareto,
        "model_retrained": False,
        "production_modified": False,
        "retrospective_not_pristine_holdout": True,
        "exposure_differences_are_not_alpha": True,
        "N4_N5_N6_completed": False,
        "protection_events": [d for d in decisions if d["reason"]],
        "isolated_frozen_position_sessions": sum(d["frozen_count"] > 0 for d in decisions),
        "mandatory_exit_only_sessions": sum(d["status"] == "MANDATORY_EXITS_ONLY" for d in decisions),
        "pending_untradable_exit_sessions": sum(d["pending_exit_count"] > 0 for d in decisions),
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(comparisons).to_parquet(output / "paired-comparisons.parquet", index=False)
    lines = [
        "# N3 完整回放结果",
        "",
        "24组×181日；2026年前九个月、1000万元。单边滑点10/20bps，另计佣金、印花税和过户费。无风险利率0，年化252日。",
        "",
        f"N3研究完成不等于模型发布；本历史已多次查看。旧v1/v2/v3只作诊断，以下仅使用{run.name}。",
        "",
    ]
    for bps in (10, 20):
        lines += [
            f"## {bps}bps",
            "",
            "|模型／组合器|净收益|夏普|最大回撤|平均实际仓位|年化换手¹|最大行业平均权重|",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for case, row in metrics.loc[bps].iterrows():
            lines.append(
                f"|{case}|{row.net_return:.2%}|{row.sharpe:.3f}|{row.maximum_drawdown:.2%}|{row.mean_actual_stock_exposure:.2%}|{row.annualized_half_gross_turnover:.2f}|{row.mean_largest_industry_weight:.2%}|"
            )
        lines.append("")
    lines += [
        "¹全年化换手为每日买卖名义额之和÷前日净值÷2，再乘252/181；包含初始建仓，并非纯卖出周转。",
        "",
        "## 10bps分季度与执行成本",
        "",
        "|模型／组合器|一季度|二季度|三季度|年化波动|手续费元|滑点损耗元|成交笔数|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case, row in metrics.loc[10].iterrows():
        q = [row.quarters[f"2026Q{i}"]["net_return"] for i in (1, 2, 3)]
        lines.append(
            f"|{case}|{q[0]:.2%}|{q[1]:.2%}|{q[2]:.2%}|{row.annualized_volatility:.2%}|{row.fees:,.2f}|{row.slippage:,.2f}|{row.fills}|"
        )
    lines += [
        "",
        "## 同预测的成本感知组合相对TopK",
        "",
        "|模型|成本|收益差|夏普差|回撤减少|平均仓位差|收益差95%区间|",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in comparisons:
        if not row["reference"].endswith("--topk"):
            continue
        lo, hi = row["net_return_difference_ci95"]
        lines.append(
            f"|{row['candidate']}|{row['bps']}|{row['net_return_difference']:.2%}|{row['sharpe_difference']:.3f}|{row['drawdown_reduction']:.2%}|{row['mean_stock_exposure_difference']:.2%}|[{lo:.2%}, {hi:.2%}]|"
        )
    lines += [
        "",
        "区间采用同日配对20日环形区块、2000次重采样；未校正多重比较，不能证明未来有效。风险厌恶系数/阈值未按本轮收益寻优。",
        "",
        "## 执行与限制",
        "",
        "- 独立现金、逐证券数量、成交导出、订单关联核验24账户全部通过；每日账本对账通过。",
        f"- 剩余保护记录 {len(payload['protection_events'])} 条；只冻结个股且其余可调整的记录 {payload['isolated_frozen_position_sessions']} 条，详见results.json。",
        "- 整手向下取整导致不同组合现金残留不同；实际仓位已单列，低回撤/低波动不自动证明预测变好。",
        f"- 收益校准36个模型月份中，{calibration['zero_expected_return_months']}个预期收益向量为零；这些优化决策主要由风险与成本驱动，不能称预测模型已充分作用于持仓。",
        "- 行业来自截至当日可见的历史分类；创业板/科创板与科技行业不是同一指标。",
        "- 模型、实际模拟账户和历史成交均未改；N4、N5、N6不在本次完成范围。",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "cases": len(metrics),
                "protections": len(payload["protection_events"]),
                "output": str(output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.run, args.output)
