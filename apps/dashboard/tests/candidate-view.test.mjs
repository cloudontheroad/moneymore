import assert from "node:assert/strict";
import test from "node:test";
import {candidateOptionLabel,candidateObservationLabel,candidateReviewReason,visibleStrategyAccounts} from "../app/candidate-view.ts";

test("failed entry validation is not a zero-day performance elimination",()=>{
  const row={candidate_tag:"20261024",archived:true,review_stage:"GATE_REJECTED",observation_days:0,common_observation_days:0,research_gate_failures:["COST_ADJUSTED_EXCESS_RETURN"],cost_adjusted_top_k_excess_return:-0.0007748461325810382};
  assert.equal(candidateOptionLabel(row,"研究验收未通过"),"[已归档] 20261024 · 入场验收未通过 · 尚未开始观察");
  assert.match(candidateReviewReason(row),/-0.0775%/);
  assert.match(candidateReviewReason(row),/非模拟盘累计收益/);
  assert.doesNotMatch(candidateOptionLabel(row,"研究验收未通过"),/0日|淘汰/);
});
test("failed research gate does not erase existing observation history",()=>{
  const row={review_stage:"GATE_REJECTED",observation_days:47,common_observation_days:47,lifecycle:{gate_failures:["RANK_IC"]}};
  assert.equal(candidateObservationLabel(row),"历史观察47日");
  assert.match(candidateReviewReason(row),/Rank IC未达标/);
});
test("performance elimination retains its distinct status and actual days",()=>{
  const row={candidate_tag:"test",archived:true,review_stage:"EARLY_REJECTED",common_observation_days:40,review_reason:"显著落后两个对照"};
  assert.match(candidateOptionLabel(row,"提前淘汰"),/提前淘汰 · 40日/);
  assert.equal(candidateReviewReason(row),"显著落后两个对照");
});

test("one archive switch controls chart and comparison accounts without mutating histories",()=>{
  const active=[{account_id:"valid",research_gate_passed:true},{account_id:"stale",research_gate_passed:true,latest_status:"OUTDATED"}];
  const archived=[{account_id:"rejected",archived:true,research_gate_passed:true,review_stage:"EARLY_REJECTED"},{account_id:"gate",archived:true,research_gate_passed:false,review_stage:"GATE_REJECTED"}];
  const before=JSON.stringify([active,archived]);
  assert.deepEqual([...visibleStrategyAccounts(active,archived,["baseline"],false)],["baseline","valid"]);
  assert.deepEqual([...visibleStrategyAccounts(active,archived,["baseline"],true)],["baseline","valid","stale","rejected","gate"]);
  assert.equal(JSON.stringify([active,archived]),before);
});
