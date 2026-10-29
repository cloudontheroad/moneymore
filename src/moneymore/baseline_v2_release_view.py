"""Small read-only release dashboard payload; never starts research or orders."""

import json
from pathlib import Path


def release_view(root):
    base = (root / "state/baseline-v2-release").resolve()
    latest = base / "latest.json"
    if not latest.exists():
        return {"status": "NOT_PREPARED", "stages": [], "real_orders_enabled": False}
    pointer = json.loads(latest.read_text(encoding="utf-8"))
    directory = Path(pointer["directory"]).resolve()
    if not directory.is_relative_to(base / "runs"):
        raise ValueError("RELEASE_VIEW_OUTSIDE_RESEARCH_DIRECTORY")

    def load(name):
        path = directory / (name + ".json")
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    status = load("release-status")
    financial_stage = (
        "financial_controls"
        if "financial_controls" in load("manifest").get("protocol", {}).get("required_stages", [])
        else "financial_vintages"
    )
    labels = {
        "attribution": "逐年持仓与风格归因",
        "full_union_economics": "全历史联合池经济性",
        financial_stage: "财务供应商数据与时点控制"
        if financial_stage == "financial_controls"
        else "财务原始版本",
        "robustness": "权重与滚动稳健性",
        "execution_stress": "全链路执行压力",
        "parity": "研究与模拟路径一致",
        "forward_observation": "新版本前瞻观察",
    }
    stages = [
        {
            "id": name,
            "label": label,
            "status": status.get("stage_statuses", {}).get(name, "NOT_COMPLETED"),
        }
        for name, label in labels.items()
    ]
    accounting = load("attribution")
    rows = []
    for run in accounting.get("runs", []):
        for annual in run.get("annual", []):
            rows.append(
                {
                    "strategy": "固定权重对照" if run["variant"] == "fixed" else "学习权重",
                    "year": annual["year"],
                    "net_return": annual["net_return"],
                    "growth_contribution": annual["board_return_contributions"].get(
                        "GROWTH_STAR", 0
                    ),
                    "main_contribution": annual["board_return_contributions"].get("MAIN_BOARD", 0),
                    "fees": annual["fees"],
                }
            )
    source_progress = base / "source-cache/progress.json"
    progress = (
        json.loads(source_progress.read_text(encoding="utf-8")) if source_progress.exists() else {}
    )
    return {
        "status": status.get("status", "BLOCKED_NOT_FINAL_V2"),
        "version": directory.name,
        "stages": stages,
        "annual_accounting": rows,
        "mature_prospective_days": load("forward_observation").get("mature_prospective_days", 0),
        "return_robustness": load("robustness").get("continuous_return_status", "NOT_COMPLETED"),
        "source_progress": progress,
        "real_orders_enabled": False,
        "baseline_replaced": False,
        "paper_activation_authorized": status.get("paper_release_authorized", False),
        "observation_activation_authorized": status.get("observation_activation_authorized", False),
        "promotion_authorized": status.get("promotion_authorized", False),
        "observation_blocked_stages": status.get("observation_blocked_stages", []),
        "execution_stress": load("execution_stress").get("runs", []),
        "notes": [
            "行业/板块数字是实际损益记账归属，不是因果风格Alpha。",
            "历史回放和事后补齐不计入新的前瞻观察日。",
            "原基线和旧候选历史不改，未通过门禁不启用最终V2。",
        ],
    }
