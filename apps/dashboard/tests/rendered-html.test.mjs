import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const app = new URL("../app/", import.meta.url);

test("dashboard presents the complete multi-sector portfolio", async () => {
  const source = await readFile(new URL("Dashboard.tsx", app), "utf8");
  for (const label of ["全局横截面组合", "整体配置全貌", "行业与个股", "策略研究", "运行与对账", "银行", "红利", "工业有色", "芯片", "创业板成长", "基线日内智能执行"]) {
    assert.match(source, new RegExp(label));
  }
  assert.match(source, /\/api\/sector-portfolio/);
  assert.match(source, /\/api\/bank-dashboard/);
  assert.match(source, /\/api\/multi-sector-execution/);
  assert.match(source, /目标与实际仓位偏差/);
  assert.match(source, /行业收益与持仓归因/);
  assert.match(source, /风险监控/);
  assert.match(source, /数据健康/);
  assert.match(source, /\/api\/data-quality/);
  assert.match(source, /\/api\/intraday-execution/);
  assert.match(source, /DATA QUALITY CENTER/);
  assert.match(source, /账户风险状态机/);
  assert.match(source, /REDUCE_ONLY/);
  assert.match(source, /SELL_ONLY/);
  assert.match(source, /\/api\/risk-state\/recover/);
  assert.match(source, /symbol_names/);
  assert.match(source, /security\(v,names\)/);
  assert.match(source, /names\[s\]/);
  assert.doesNotMatch(source, /MA120|MA250|招商银行|长江电力/);
});

test("metadata describes the multi-sector console", async () => {
  const layout = await readFile(new URL("layout.tsx", app), "utf8");
  assert.match(layout, /MoneyMore · 多行业量化组合/);
  assert.match(layout, /银行、红利、工业有色、芯片与创业板成长/);
});

test("V2 release panel is read-only and does not present partial evidence as approval", async () => {
  const source = await readFile(new URL("BaselineV2Release.tsx", app), "utf8");
  for (const label of ["基线 V2 · 发布验收", "最终版未启用", "可信新前瞻日", "不是因果归因", "/api/baseline-v2-release"]) {
    assert.ok(source.includes(label));
  }
  assert.ok(source.includes('status === "PASS"'));
  assert.ok(!source.includes('method: "POST"'));
});

test("Qlib operating page does not render research and training process panels", async () => {
  const source = await readFile(new URL("Dashboard.tsx", app), "utf8");
  const page = source.slice(source.indexOf("function ChallengerPage("), source.indexOf("function AccountColumn("));
  for (const panel of ["<BaselineV2Release", "<FactorBaselineV2Info", "<ChallengerDiagnostics", "<CandidateObservation", "统一样本外模型竞赛", "固定预算在首次实际模拟建仓时冻结"]) {
    assert.ok(!page.includes(panel), panel);
  }
  for (const panel of ["<CandidateSwitcher", "<AccountColumn", "<PerformanceComparisonCharts", "<TradeCycleComparison", "candidateRunStatus"]) {
    assert.ok(page.includes(panel), panel);
  }
  const route = source.split("\n").find(line => line.includes('page === "challenger"'));
  for (const panel of ["<ChallengerEvidence", "<PointInTimeEvidence", "<GovernanceEvidence", "<LongTermReview"]) assert.ok(!route.includes(panel), panel);
});

test("Qlib third column is fixed V2 while the fourth remains selectable", async () => {
  const source = await readFile(new URL("Dashboard.tsx", app), "utf8");
  const page = source.slice(source.indexOf("function ChallengerPage("), source.indexOf("function AccountColumn("));
  assert.ok(page.includes('AccountColumn title="基线V2"'));
  assert.ok(page.includes('accountId="qlib_candidate_20261029"'));
  assert.ok(page.includes('selectedCandidate?.account_id'));
  assert.ok(!page.includes('AccountColumn title="挑战者"'));
  assert.ok(!page.includes('GRU最新全局排名'));
  assert.ok(source.includes('第四列候选策略'));
});
