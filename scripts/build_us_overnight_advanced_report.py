"""Build the complete human-readable advanced overnight search report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "state" / "us-overnight-advanced-search"


def fmt(value: object, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def main() -> int:
    rows = [
        json.loads(line)
        for line in (STATE / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    champion = json.loads((STATE / "champion.json").read_text(encoding="utf-8"))
    baseline = next(row for row in rows if row["experiment_id"] == "baseline")
    final_id = champion["winner"]
    final = next(row for row in rows if row["experiment_id"] == final_id)
    training_rows = [row for row in rows if "metrics" in row]

    flat = []
    for number, row in enumerate(training_rows, start=1):
        metrics = row["metrics"]
        decision = "REJECTED"
        if row["experiment_id"] == "baseline":
            decision = "BASELINE"
        elif row["experiment_id"] == final_id:
            decision = "ADOPTED"
        elif row["experiment_id"] == "top1_weight-1p5":
            decision = "RISK_REVIEW_REJECTED"
        flat.append(
            {
                "round": number,
                "experiment_id": row["experiment_id"],
                "phase": row["phase"],
                "value": row["value"],
                "decision": decision,
                "reason": row.get("reason", ""),
                **metrics,
            }
        )
    detail = pd.DataFrame(flat)
    detail.to_csv(STATE / "complete_round_report.csv", index=False, encoding="utf-8-sig")

    b = baseline["metrics"]
    f = final["metrics"]
    lines = [
        "# 美股隔夜高级参数搜索完整报告",
        "",
        "## 总结",
        "",
        "- 历史面板：772个交易日、约74.9万条股票日样本。",
        "- 新协议：520个滚动验证日、按63日重训和分段，共约8个历史验证段。",
        f"- 完整训练轮次：{len(training_rows)}（含1轮基线、{len(training_rows)-1}轮挑战者）。",
        "- 最终采纳轮次：1。",
        "- 风险复核撤销：1（Top1加权1.5因最大回撤恶化）。",
        "- 真实交易：始终关闭。",
        "",
        "## 最终冠军",
        "",
        f"- 采纳实验：`{final_id}`",
        f"- 唯一新增参数：`min_child_weight={champion['parameters']['min_child_weight']}`",
        f"- Top1日均净收益：{fmt(f['top1_mean_bps'])} bp（基线{fmt(b['top1_mean_bps'])} bp）",
        f"- Top1 Sharpe：{fmt(f['top1_sharpe'], 3)}（基线{fmt(b['top1_sharpe'], 3)}）",
        f"- Top1最大回撤：{fmt(f['top1_drawdown']*100)}%（基线{fmt(b['top1_drawdown']*100)}%）",
        f"- Top10日均净收益：{fmt(f['top10_mean_bps'])} bp（基线{fmt(b['top10_mean_bps'])} bp）",
        f"- Top10 Sharpe：{fmt(f['top10_sharpe'], 3)}（基线{fmt(b['top10_sharpe'], 3)}）",
        f"- 排序违例：{f['violations']}（基线{b['violations']}）",
        f"- 正收益历史分段：{f['positive_top1_folds']}（基线{b['positive_top1_folds']}）",
        f"- 最差分段Top1：{fmt(f['worst_fold_top1_bps'])} bp（基线{fmt(b['worst_fold_top1_bps'])} bp）",
        "",
        "## 全部训练轮次",
        "",
        "|轮次|实验|参数阶段|取值|结论|Top1 bp|Top1 Sharpe|Top1回撤|Top10 bp|违例|正收益段|理由|",
        "|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in flat:
        cells = row | {
            "top1": fmt(row.get("top1_mean_bps")),
            "sharpe": fmt(row.get("top1_sharpe"), 3),
            "drawdown": fmt(float(row.get("top1_drawdown", 0)) * 100),
            "top10": fmt(row.get("top10_mean_bps")),
            "violations": row.get("violations", "—"),
            "folds": row.get("positive_top1_folds", "—"),
        }
        lines.append(
            "|{round}|`{experiment_id}`|{phase}|{value}|{decision}|{top1}|{sharpe}|{drawdown}%|{top10}|{violations}|{folds}|{reason}|".format(
                **cells
            )
        )
    lines.extend(
        [
            "",
            "## 风险说明",
            "",
            "累计收益在长周期逐日复利下会被少数极端隔夜收益放大，因此晋升主要依据日均净收益、Sharpe、最大回撤、分段稳定性和排序违例，而不是累计收益绝对值。",
            "",
            "Top1加权1.5虽然把Top1日均收益提高到226.61 bp，但最大回撤由-32.29%恶化到-65.28%，已在风险复核中撤销，不进入正式模型。",
        ]
    )
    (STATE / "complete_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rounds": len(training_rows), "adopted": 1, "champion": final_id}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
