type Candidate = Record<string, unknown>;

export function costExecutionLabel(execution?: Candidate): string {
  if (!execution?.status) return "尚未进入首次独立执行";
  const plan = execution.plan as Candidate | undefined;
  const orders = Array.isArray(plan?.orders) ? plan.orders.length : 0;
  const labels: Record<string, string> = {
    PLAN_BLOCKED: "执行受阻", QUOTE_QUALITY_BLOCKED: "行情质量阻断",
    COST_EXECUTION_COMPENSATION_COOLDOWN: "等待有限补偿",
    COST_EXECUTION_BUDGET_EXHAUSTED: "补偿预算用尽，未成交", PREPARED: "已准备，待成交对账",
    EXECUTION_MISMATCH: "成交对账异常", ACCOUNT_RISK_BLOCKED: "账户风控阻断",
  };
  const status = execution.status === "COMPLETED" ? (orders ? `已模拟成交 ${orders} 笔` : "无需交易，已完成核验")
    : labels[String(execution.status)] ?? String(execution.status);
  const reason = plan?.reason ?? execution.reason;
  return `${execution.trade_date ?? ""} · ${status}${reason ? ` · ${reason}` : ""}`;
}

export function visibleStrategyAccounts(
  active: Candidate[], archived: Candidate[], permanent: string[], showArchived: boolean,
): Set<string> {
  const excluded = new Set(["EARLY_REJECTED", "REJECTED", "GATE_REJECTED", "REVIEW_EXPIRED", "ELIMINATED", "RETIRED", "OUTDATED"]);
  const rows = [...active, ...archived];
  return new Set([...permanent, ...rows.filter(row => showArchived || (
    !row.archived && !row.validation_hold && Boolean(row.research_gate_passed)
    && !excluded.has(String(row.review_stage ?? "").toUpperCase())
    && !excluded.has(String(row.latest_status ?? "").toUpperCase())
  )).map(row => String(row.account_id))]);
}

const gateNames: Record<string, string> = {
  COST_ADJUSTED_EXCESS_RETURN: "成本后Top-K超额未达标",
  RANK_IC: "Rank IC未达标",
  RANK_IC_IR: "ICIR未达标",
  SEED_STABILITY: "随机种子稳定性未达标",
  SAMPLE_COUNT: "样本数量不足",
  UNIVERSE_SIZE: "候选池覆盖不足",
};

export function candidateObservationLabel(row: Candidate): string {
  const days = Number(row.common_observation_days ?? row.observation_days ?? 0);
  if (row.review_stage === "GATE_REJECTED") {
    return days > 0 ? `历史观察${days}日` : "尚未开始观察";
  }
  return `${days}日`;
}

export function candidateOptionLabel(row: Candidate, statusName: string): string {
  const neverStarted = row.review_stage === "GATE_REJECTED"
    && Number(row.observation_days ?? row.common_observation_days ?? 0) === 0;
  const status = neverStarted ? "入场验收未通过" : statusName;
  return `${row.archived ? "[已归档] " : ""}${row.candidate_tag}${row.display_name ? ` ${row.display_name}` : ""} · ${status} · ${candidateObservationLabel(row)}`;
}

export function candidateReviewReason(row: Candidate): string {
  if (row.review_stage !== "GATE_REJECTED") return String(row.review_reason ?? "等待评审");
  const lifecycle = row.lifecycle as Candidate | undefined;
  const failures = lifecycle?.gate_failures ?? row.research_gate_failures;
  const reasons = (Array.isArray(failures) ? failures : []).map((code: unknown) => {
    if (code === "COST_ADJUSTED_EXCESS_RETURN") {
      const value = row.cost_adjusted_top_k_excess_return;
      const metric = value != null && Number.isFinite(Number(value))
        ? ` ${(Number(value) * 100).toFixed(4)}%` : "未取得有效值";
      return `离线成本后Top-K超额${metric}，要求 > 0%（非模拟盘累计收益）`;
    }
    return gateNames[String(code)] ?? String(code);
  });
  return `${candidateObservationLabel(row)}；${reasons.join("；") || "研究验收未通过"}。不属于观察期绩效淘汰。`;
}
