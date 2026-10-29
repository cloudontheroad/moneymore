"use client";

import { useEffect, useState } from "react";

type Stage = { id: string; label: string; status: string };
type Annual = { strategy: string; year: string; net_return: number; growth_contribution: number; main_contribution: number; fees: number };
type Stress = { scenario: string; variant: string; days: number; input_hold_days: number; blocked_solver_plans: number; performance: { cumulative_return: number; maximum_drawdown: number; mean_gross_exposure: number } };
type Evidence = { version: string; status: string; stages: Stage[]; annual_accounting: Annual[]; mature_prospective_days: number; source_progress?: { pending_after?: number }; notes?: string[]; observation_activation_authorized?: boolean; promotion_authorized?: boolean; execution_stress?: Stress[]; style_experiment?: {status:string;display_name:string;candidate_tag:string;signal_date:string;planned_trade_date?:string;first_live_paper_fill_verified:boolean} };

const scenarios: Record<string, string> = { slippage_5bps: "5bps滑点", slippage_10bps: "10bps滑点", combined_10bps_double_commission_tax20: "10bps＋双倍佣金＋股息扣20%", platform_risk_hold: "旧风险入口：非正常状态全冻结", platform_risk_reductions: "修正风险执行：减仓/只卖，显式恢复" };

function label(status: string) {
  if (status === "PASS") return "已通过";
  if (status.startsWith("ACCOUNTING_PASS")) return "记账通过 · 风格待核验";
  if (status.startsWith("FAIL")) return "稳健性未通过";
  if (status.startsWith("WAITING")) return "等待合格版本与真实观察";
  if (status.startsWith("BLOCKED")) return "已阻断 · 证据待补齐";
  if (status.startsWith("INVALID")) return "证据失效 · 需重新核验";
  if (status.startsWith("PARTIAL_ENDOGENOUS")) return "回放证据已接入 · 容量与路径验收待完成";
  return "尚未完成";
}

export default function BaselineV2Release() {
  const [data, setData] = useState<Evidence | null>(null);
  const [error, setError] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    const refresh = async () => {
      try {
        const response = await fetch("/api/baseline-v2-release", { cache: "no-store", signal: controller.signal });
        if (!response.ok) throw new Error("release evidence unavailable");
        setData(await response.json() as Evidence); setError(false);
      } catch { if (!controller.signal.aborted) setError(true); }
    };
    void refresh(); const timer = setInterval(() => { void refresh(); }, 60000);
    return () => { controller.abort(); clearInterval(timer); };
  }, []);
  const percent = (value: number) => `${(value * 100).toFixed(2)}%`;
  return <section className="panel">
    <header><div><h3>基线 V2 · 发布验收</h3><p>独立风格修正实验，不覆盖原基线；实验启用与正式晋级分开。</p></div><span>{data?.style_experiment ? "V2 独立模拟已启用" : data?.status === "ELIGIBLE_FOR_INDEPENDENT_PAPER_RELEASE" ? "具备独立模拟发布资格" : "最终版未启用"}</span></header>
    {error && <p role="status">发布证据暂时无法读取，不能据此视为已通过。</p>}
    {!data && !error && <p className="empty">正在读取发布证据…</p>}
    {data?.style_experiment&&<><p>{data.style_experiment.display_name} · 版本 {data.style_experiment.candidate_tag} · 信号 {data.style_experiment.signal_date} · 下一计划交易日 {data.style_experiment.planned_trade_date??"待交易日历确认"}</p><p>已接入每日候选计划与开盘模拟执行。{data.style_experiment.first_live_paper_fill_verified?"首轮实时模拟成交已核验。":"尚未验收首笔启用后的实时模拟成交；收盘后只生成计划，不补造历史成交。"} 在候选策略下拉框选择本版查看风格修正、排名、委托和持仓。</p></>}
    {data && <details open={!data.style_experiment}><summary>历史研究与正式晋级证据（不等于独立实验启用状态）</summary>
      <p>版本：{data.version} · 可信新前瞻日：{data.mature_prospective_days} · {data.source_progress?.pending_after !== undefined ? `待补官方历史响应 ${data.source_progress.pending_after} 只` : "数据核验中"}</p>
      <p>观察版启用：{data.observation_activation_authorized ? "硬门禁通过，具备启用资格（不代表账户已启动）" : "数据、模型或执行硬门禁尚未通过"} · 正式晋级：{data.promotion_authorized ? "具备评审资格" : "未通过"}。20/60日前瞻计数从观察版实际启用后开始，不作为首次启用的前置条件。</p>
      <div className="table-wrap"><table><thead><tr><th>发布环节</th><th>核验状态</th></tr></thead><tbody>{data.stages.map(stage => <tr key={stage.id}><td>{stage.label}</td><td>{label(stage.status)}</td></tr>)}</tbody></table></div>
      {!!data.execution_stress?.length && <details><summary>完整两年压力与风险合同（不同风险合同不可混作同一收益）</summary><div className="table-wrap"><table><thead><tr><th>场景</th><th>研究对照</th><th>账本日</th><th>累计收益</th><th>最大回撤</th><th>平均股票仓位</th><th>数据暂停 / 求解阻断日</th></tr></thead><tbody>{data.execution_stress.map(row => <tr key={`${row.scenario}-${row.variant}`}><td>{scenarios[row.scenario] ?? row.scenario}</td><td>{row.variant === "fixed" ? "固定权重" : "学习权重"}</td><td>{row.days}</td><td>{percent(row.performance.cumulative_return)}</td><td>{percent(row.performance.maximum_drawdown)}</td><td>{percent(row.performance.mean_gross_exposure)}</td><td>{row.input_hold_days} / {row.blocked_solver_plans}</td></tr>)}</tbody></table></div><p>股息扣20%仅是压力假设。风险修正版没有自动恢复买入；长期现金会显著改变收益。此表不代表新版本已获发布许可。</p></details>}
      <details><summary>展开逐年实际损益归属（不是因果归因）</summary><div className="table-wrap"><table><thead><tr><th>研究对照</th><th>区间</th><th>净收益</th><th>创业 / 科创贡献</th><th>主板贡献</th><th>显式费用</th></tr></thead><tbody>{data.annual_accounting.map(row => <tr key={`${row.strategy}-${row.year}`}><td>{row.strategy}</td><td>{row.year === "2024" ? "2024-01-31起" : row.year}</td><td>{percent(row.net_return)}</td><td>{percent(row.growth_contribution)}</td><td>{percent(row.main_contribution)}</td><td>¥{row.fees.toLocaleString("zh-CN", { maximumFractionDigits: 0 })}</td></tr>)}</tbody></table></div></details>
      {(data.notes ?? []).map(note => <p key={note}>{note}</p>)}
    </details>}
  </section>;
}
