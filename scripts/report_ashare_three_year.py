"""Independent cash/share, chronology, style and paired 36-month performance audit."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter

import numpy as np
import pandas as pd
from diagnose_ashare_rank_turnover import json_safe, position_trade_partition
from prepare_ashare_three_year import BASE, CONFIG, ROOT
from prepare_four_strategy_2026 import save
from report_ashare_mainline_replay import audit_ledger

from moneymore.research.ashare_mainline import sha256
from moneymore.research.paired_bootstrap import paired_block_intervals
from moneymore.research.period_statistics import period_statistics

NAMES = {
    "rank_cap": "原版 rank_cap（历史滚动重训）",
    "financial_rank_cap": "rank_cap＋财务增强",
    "rnn_rank_cap": "单骨干 RNN 对照",
    "tra_rank_cap": "TRA 三专家（LR 路由）",
    "rank_cap_vol": "rank_cap＋波动仓位",
    "rank_cap_static": "等平均仓位静态对照（事后）",
}


def historical_industry(frame, panel):
    left = frame.copy()
    left["_row"] = np.arange(len(left))
    # No forward/back fill from a future membership observation.
    right = panel.loc[panel.industry.notna(), ["date", "symbol", "industry"]].copy()
    left["_time"] = pd.to_datetime(left.date)
    right["_time"] = pd.to_datetime(right.pop("date"))
    got = pd.merge_asof(
        left.sort_values(["_time", "symbol"]),
        right.sort_values(["_time", "symbol"]),
        on="_time",
        by="symbol",
        direction="backward",
    )
    if got.industry.isna().any():
        raise ValueError("MISSING_HISTORICAL_INDUSTRY")
    return got.sort_values("_row").drop(columns=["_row", "_time"])


def drawdown_interval(equity, capital, previous_date):
    nav = pd.concat([pd.Series([capital], index=[previous_date]), equity.set_index("date").equity])
    peaks = nav.cummax()
    trough = (nav / peaks - 1).idxmin()
    peak_value = float(peaks.loc[trough])
    peak = nav.loc[:trough].loc[lambda s: s.eq(peak_value)].index[-1]
    recovery = nav.loc[nav.index > trough].loc[lambda s: s >= peak_value]
    return {
        "peak_date": peak,
        "trough_date": trough,
        "peak_nav": peak_value,
        "trough_nav": float(nav.loc[trough]),
        "net_pnl": float(nav.loc[trough] - peak_value),
        "recovery_date": recovery.index[0] if len(recovery) else None,
    }


def attribution(folder, equity, fills, holdings, panel, capital, previous_date):
    with sqlite3.connect(
        (folder / "paper.sqlite3").resolve().as_uri() + "?mode=ro", uri=True
    ) as db:
        ledger = pd.read_sql_query("SELECT * FROM corporate_action_ledger", db)
    partition, _ = position_trade_partition(fills, ledger, holdings, equity.date.tolist())
    h = holdings.merge(equity[["date", "equity"]], on="date", validate="many_to_one")
    h["value"] = h.weight * h.equity
    values = h.pivot(index="date", columns="symbol", values="value").reindex(equity.date).fillna(0)
    symbols = sorted(set(values.columns) | set(fills.symbol) | set(ledger.symbol))
    values = values.reindex(columns=symbols, fill_value=0)
    flows = pd.DataFrame(0.0, index=values.index, columns=symbols)
    for r in fills.itertuples():
        flows.loc[r.date, r.symbol] += r.notional * (1 if r.side == "SELL" else -1) - r.fee
    for r in ledger.itertuples():
        if r.trade_date in flows.index:
            flows.loc[r.trade_date, r.symbol] += r.cash_amount
    pnl = values - values.shift(1, fill_value=0) + flows
    daily = equity.set_index("date")
    change = daily.equity - daily.equity.shift(1).fillna(capital)
    residual = change - pnl.sum(axis=1)
    # The engine NAV additionally includes unsettled stock/cash entitlements.
    rights = daily.equity - daily.cash - values.sum(axis=1)
    right_change = rights - rights.shift(1, fill_value=0)
    if (residual - right_change).abs().max() > 0.01:
        raise ValueError("DAILY_PNL_ATTRIBUTION_NOT_RECONCILED")
    long = (
        pnl.where(pnl.abs().gt(1e-7))
        .stack()
        .rename("net_pnl")
        .rename_axis(["date", "symbol"])
        .reset_index()
    )
    long = historical_industry(long, panel)
    dd = drawdown_interval(equity, capital, previous_date)
    loss = long.loc[long.date.gt(dd["peak_date"]) & long.date.le(dd["trough_date"])]
    cost = fills.loc[fills.date.gt(dd["peak_date"]) & fills.date.le(dd["trough_date"])]
    dd.update(
        {
            "industry_net_pnl": loss.groupby("industry").net_pnl.sum().sort_values().to_dict(),
            "bottom10_stock_pnl": loss.groupby("symbol")
            .net_pnl.sum()
            .sort_values()
            .head(10)
            .to_dict(),
            "fees_plus_slippage": float(cost.fee.sum() + cost.slippage.sum()),
            "unsettled_rights_change": float(
                residual.loc[
                    (residual.index > dd["peak_date"]) & (residual.index <= dd["trough_date"])
                ].sum()
            ),
        }
    )
    return {
        "drawdown": dd,
        "industry_net_pnl": long.groupby("industry").net_pnl.sum().sort_values().to_dict(),
        "trade_categories": partition.groupby("trade_category")
        .agg(
            fills=("quantity", "count"),
            notional=("notional", "sum"),
            fees=("fee", "sum"),
            slippage=("slippage", "sum"),
        )
        .to_dict("index"),
        "maximum_daily_reconciliation_error": float((residual - right_change).abs().max()),
        "unsettled_rights_total_pnl": float(residual.sum()),
        "cash_dividends_received_gross": float(
            ledger.loc[ledger.action_type.eq("CASH_DIVIDEND"), "cash_amount"].sum()
        ),
        "cash_dividend_tax_20pct_direct_amount_sensitivity": float(
            ledger.loc[ledger.action_type.eq("CASH_DIVIDEND"), "cash_amount"].sum() * 0.2
        ),
        "tax_sensitivity_is_not_compounded_replay_or_holding_period_tax": True,
        "daily_quantities_reconstructed": True,
    }


def subperiods(equity, capital, frequency):
    previous = capital
    result = {}
    for key, group in equity.groupby(
        pd.to_datetime(equity.date).dt.to_period(frequency), sort=True
    ):
        result[str(key)] = {
            "first": group.date.iloc[0],
            "last": group.date.iloc[-1],
            **period_statistics(group.equity, previous),
        }
        previous = float(group.equity.iloc[-1])
    return result


def main():
    cfg = json.loads(CONFIG.read_bytes())
    out = BASE / "final-v1"
    if (out / "results.json").exists():
        raise FileExistsError("IMMUTABLE_COMPLETED_REPORT")
    out.mkdir(exist_ok=True)
    calendar = json.loads((BASE / "data-v1/calendar.json").read_bytes())["days"]
    days = [d for d in calendar if cfg["evaluation_start"] <= d <= cfg["evaluation_end"]]
    previous_date = calendar[calendar.index(days[0]) - 1]
    panel = pd.read_parquet(
        BASE / "panel-v1/core-panel.parquet", columns=["date", "symbol", "industry"]
    )
    capital = cfg["initial_cash"]
    expected = [c + "--rank_cap" for c in cfg["cases"]]
    rows, equities, evidence = [], {}, {}
    data_audit = json.loads((BASE / "data-audit-v1.json").read_bytes())
    if data_audit["status"] != "COMPLETE_WITH_DISCLOSED_LIMITATIONS":
        raise ValueError("DATA_AUDIT_NOT_COMPLETE")
    for bps in cfg["slippage_bps"]:
        run = BASE / "replay-v1" / f"cost{bps}"
        complete = json.loads((run / "completed.json").read_bytes())
        if complete["smoke"]:
            raise ValueError("SMOKE_NOT_FORMAL")
        if complete["execution_repair_manifest_sha256"] != sha256(
            BASE / "execution-repair-v1/results.json"
        ):
            raise ValueError("EXECUTION_REPAIR_MANIFEST_CHANGED")
        for path, digest in complete["source_hashes"].items():
            if sha256(run / "sources" / path) != digest:
                raise ValueError("FROZEN_SOURCE_CHANGED")
        folder = run / f"bps{bps}"
        engine = json.loads((folder / "results.json").read_bytes())
        if (
            engine["smoke"]
            or not engine["all_daily_reconciliations_pass"]
            or {r["strategy"] for r in engine["cases"]} != set(expected)
        ):
            raise ValueError("INCOMPLETE_FORMAL_MATRIX")
        allhold = pd.read_parquet(folder / "holdings.parquet")
        for case in expected:
            equity = pd.read_parquet(folder / case / "equity.parquet")
            fills = pd.read_parquet(folder / case / "fills.parquet")
            if equity.date.tolist() != days:
                raise ValueError("NONEXACT_THREE_YEAR_CALENDAR")
            audit = audit_ledger(folder / case / "paper.sqlite3", fills)
            statistics = period_statistics(equity.equity, capital)
            engine_row = next(r for r in engine["cases"] if r["strategy"] == case)
            for metric in ["net_return", "sharpe", "maximum_drawdown", "annualized_volatility"]:
                if not np.isclose(statistics[metric], engine_row[metric], atol=1e-10):
                    raise ValueError("INDEPENDENT_METRIC_MISMATCH")
            decisions = [
                json.loads(p.read_bytes())
                for p in sorted((run / "decisions" / case).glob("*.json"))
            ]
            if [d["execution_date"] for d in decisions] != days:
                raise ValueError("INCOMPLETE_DECISION_CALENDAR")
            for decision in decisions:
                if decision["signal_date"] >= decision["execution_date"]:
                    raise ValueError("DECISION_LOOKAHEAD")
                if decision["status"] == "OPEN_PLAN_VERIFIED":
                    post = decision["open_postcheck"]
                    if not post["passed"] or decision["next_open_used_for_selection"]:
                        raise ValueError("OPEN_PLAN_AUDIT_FAILURE")
                    if post["maximum_stock_weight"] > cfg["max_stock_weight"] + 1e-5 or any(
                        w > post["industry_limits"][sector] + 1e-5
                        for sector, w in post["industry_weights"].items()
                    ):
                        raise ValueError("INDEPENDENT_OPEN_CONSTRAINT_VIOLATION")
                    if not 0 <= decision["overlay_scale"] <= 1:
                        raise ValueError("LEVERAGED_OVERLAY")
            h = allhold.loc[allhold.kind.eq(case)]
            styled = historical_industry(h, panel)
            sectors = styled.groupby(["date", "industry"]).weight.sum()
            nav = equity.set_index("date").equity
            turnover = (
                fills.groupby("date").notional.sum().reindex(nav.index, fill_value=0)
                / nav.shift(1).fillna(capital)
                / 2
            )
            rolling = []
            for end in range(251, len(equity)):
                start = end - 251
                base = capital if start == 0 else equity.equity.iloc[start - 1]
                rolling.append(
                    {
                        "start": days[start],
                        "end": days[end],
                        **period_statistics(equity.equity.iloc[start : end + 1], base),
                    }
                )
            pd.DataFrame(rolling).to_parquet(out / f"rolling-{bps}-{case}.parquet", index=False)
            row = {
                "case": case.split("--")[0],
                "name": NAMES[case.split("--")[0]],
                "bps": bps,
                **statistics,
                "ledger": audit,
                "fills": len(fills),
                "total_cost": float(fills.fee.sum() + fills.slippage.sum()),
                "fees": float(fills.fee.sum()),
                "slippage": float(fills.slippage.sum()),
                "turnover_annual_half_gross": float(turnover.sum() * 252 / len(days)),
                "mean_exposure": float(equity.exposure.mean()),
                "minimum_exposure": float(equity.exposure.min()),
                "mean_names": float(h.groupby("date").size().reindex(days, fill_value=0).mean()),
                "minimum_names": int(h.groupby("date").size().reindex(days, fill_value=0).min()),
                "maximum_single_weight_after_price_moves": float(h.weight.max()),
                "growth_board_mean_weight": float(
                    h.loc[h.symbol.str.startswith(("30", "68"))].weight.sum() / len(days)
                ),
                "mean_largest_industry": float(
                    sectors.groupby(level=0).max().reindex(days, fill_value=0).mean()
                ),
                "industry_mean_weights": (
                    styled.groupby("industry").weight.sum() / len(days)
                ).to_dict(),
                "years": subperiods(equity, capital, "Y"),
                "quarters": subperiods(equity, capital, "Q"),
                "decision_states": dict(Counter(d["status"] for d in decisions)),
                "block_reasons": dict(Counter(d["reason"] for d in decisions if d.get("reason"))),
                "attribution": attribution(
                    folder / case, equity, fills, h, panel, capital, previous_date
                ),
            }
            rows.append(row)
            equities[(bps, row["case"])] = nav / nav.shift(1).fillna(capital) - 1
            evidence[f"{bps}:{case}"] = {
                "equity_sha256": sha256(folder / case / "equity.parquet"),
                "fills_sha256": sha256(folder / case / "fills.parquet"),
            }
    comparisons = []
    for bps in cfg["slippage_bps"]:
        pairs = [(c, "rank_cap") for c in cfg["cases"] if c != "rank_cap"] + [
            ("tra_rank_cap", "rnn_rank_cap"),
            ("rank_cap_vol", "rank_cap_static"),
        ]
        for candidate, reference in pairs:
            a = next(r for r in rows if r["case"] == candidate and r["bps"] == bps)
            b = next(r for r in rows if r["case"] == reference and r["bps"] == bps)
            comparisons.append(
                {
                    "candidate": candidate,
                    "reference": reference,
                    "bps": bps,
                    "net_return_difference_pp": (a["net_return"] - b["net_return"]) * 100,
                    "sharpe_difference": a["sharpe"] - b["sharpe"],
                    "drawdown_difference_pp": (a["maximum_drawdown"] - b["maximum_drawdown"]) * 100,
                    "cost_difference": a["total_cost"] - b["total_cost"],
                    "paired_intervals": paired_block_intervals(
                        equities[(bps, candidate)], equities[(bps, reference)]
                    ),
                }
            )
    trained = []
    frozen_model_sources = None
    for path in sorted((BASE / "training-v1").glob("*/*/results.json")):
        result = json.loads(path.read_bytes())
        if frozen_model_sources is None:
            frozen_model_sources = result["source_hashes"]
        if result["source_hashes"] != frozen_model_sources:
            raise ValueError("MODEL_SOURCE_MIXED_DURING_MATRIX")
        for source, digest in result["source_hashes"].items():
            if sha256(ROOT / source) != digest:
                raise ValueError("MODEL_SOURCE_CHANGED_AFTER_TRAINING:" + source)
        contract = json.loads((path.parent / "time-contract.json").read_bytes())
        if (
            result["smoke"]
            or result["reload_error"] > 1e-6
            or not contract["train"]["last_label_end"] < contract["valid"]["first"]
            or not contract["valid"]["last_label_end"] < contract["test"]["first"]
        ):
            raise ValueError("MODEL_TIME_OR_RELOAD_CHECK")
        if result["predictions_sha256"] != sha256(path.parent / "predictions.parquet"):
            raise ValueError("PREDICTIONS_CHANGED")
        diag = pd.read_parquet(path.parent / "daily-diagnostics.parquet")
        routing = path.parent / "routing-diagnostics.json"
        financial_gain = None
        importance_path = path.parent / "feature-importance.parquet"
        if result["case"] == "financial_rank_cap":
            importance = pd.read_parquet(importance_path).gain
            names = cfg["financial_features"]
            financial_gain = {
                "new_feature_gain": importance.reindex(names, fill_value=0).to_dict(),
                "new_missing_indicator_gain": importance.reindex(
                    ["missing__" + n for n in names], fill_value=0
                ).to_dict(),
                "new_feature_and_missing_gain_fraction": float(
                    importance.reindex(names + ["missing__" + n for n in names], fill_value=0).sum()
                    / max(float(importance.sum()), 1e-12)
                ),
                "interpretation": "training gain, not causal realized-return contribution",
            }
        trained.append(
            {
                "fold": result["fold"],
                "case": result["case"],
                "mature_days": int(diag.rank_ic.notna().sum()),
                "rank_ic": float(diag.rank_ic.mean()),
                "minimum_distinct_scores": result["minimum_distinct_scores"],
                "routing": json.loads(routing.read_bytes()) if routing.exists() else None,
                "financial_gain": financial_gain,
            }
        )
    if len(trained) != 52 or len(rows) != 12:
        raise ValueError("INCOMPLETE_52_MODEL_12_ACCOUNT_MATRIX")
    payload = json_safe(
        {
            "status": "COMPLETE_ISOLATED_THREE_YEAR_RESEARCH",
            "window": [days[0], days[-1]],
            "sessions": len(days),
            "configuration": cfg,
            "configuration_sha256": sha256(CONFIG),
            "authoritative_replay_protocol": "cost*/runtime.json; legacy engine protocol.json retains an older study link and diagnostic field names, not the active planner settings",
            "cases": rows,
            "comparisons": comparisons,
            "trained_models": trained,
            "evidence": evidence,
            "data_audit_sha256": sha256(BASE / "data-audit-v1.json"),
            "execution_quote_repair": json.loads(
                (BASE / "execution-repair-v1/results.json").read_bytes()
            ),
            "production_modified": False,
            "limitations": [
                "回顾性三年并非所有周期；包含此前已多次检视的2026前九月",
                "财报公告可用日期过滤不能证明供应商原始历史版本",
                "TRA仅LR路由消融，不是完整TPE复现",
                "波动组不加杠杆并继承20日换仓；静态平均仓位使用事后信息，只作诊断",
                "不自动发布或替换原生产策略",
                "固定单一训练种子；配对区块区间未校正多重比较，不代替冻结后的前瞻验证",
                "原模型面板有286条2026年9—10月候选股日行情缺口（726000候选股日的约0.039%）；正式价格补数仅用于撮合与持仓估值，冻结预测特征与选股风险资格未据此重算，三年检验并非无缺失数据实验",
                "公司行为范围与未支持事项见数据审计；沿用既有税前现金分红，未模拟按持有期限补扣的个人红利税，另列现金红利20%直接金额敏感性（不是完整复利回放）",
            ],
        }
    )
    save(out / "results.json", payload)
    lines = [
        "# rank_cap 三年升级实测",
        "",
        f"2023-11-01—2026-10-30，{len(days)}个交易日，1000万元。各组同历史池、滚动训练边界、整数手、T+1、停牌/涨跌停和公司行为处理。收益为佣金、印花税、过户费及滑点后，个人红利税前；夏普为无风险利率0、252日年化。",
        "",
        "| 策略 | 滑点 | 三年净收益 | 年化收益 | 夏普 | 年化波动 | 最大回撤 | 总成本万元 | 双创平均仓位 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['bps']}bps | {r['net_return']:.2%} | {r['annualized_return']:.2%} | {r['sharpe']:.3f} | {r['annualized_volatility']:.2%} | {r['maximum_drawdown']:.2%} | {r['total_cost'] / 10000:.2f} | {r['growth_board_mean_weight']:.2%} |"
        )
    lines += [
        "",
        "## 分年度收益（首尾年度不是全年）",
        "",
        "| 策略（10bps） | 2023年11—12月 | 2024年 | 2025年 | 2026年1—10月 |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r["bps"] == 10:
            lines.append(
                "| "
                + r["name"]
                + " | "
                + " | ".join(f"{r['years'][str(y)]['net_return']:.2%}" for y in range(2023, 2027))
                + " |"
            )
    lines += [
        "",
        "## 限制",
        "",
        *["- " + s for s in payload["limitations"]],
        "",
        "独立重建现金/股份及每日归因；完整季度、滚动一年、区块置信区间、专家使用比例、交易受阻原因见同目录 results.json。",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "accounts": len(rows),
                "sessions": len(days),
                "report": str(out / "RESULTS.md"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
