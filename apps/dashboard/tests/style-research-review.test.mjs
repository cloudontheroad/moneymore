import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import vm from "node:vm";
import test from "node:test";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ts from "typescript";

const source = readFileSync(new URL("../app/StyleResearchReview.tsx", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } });
const exported = {};
vm.runInNewContext(compiled.outputText, { exports: exported, require: createRequire(import.meta.url) });
const render = review => renderToStaticMarkup(React.createElement(exported.default, { review }));

test("review is absent until evidence exists, never creates a strategy column", () => {
  assert.equal(render(undefined), "");
  assert.equal(render(null), "");
});

test("actual rendered review distinguishes missing metrics from zero and escapes evidence text", () => {
  const html = render({title: "无合格新模型", reason: "<script>unsafe</script>", gates: [
    {stage: "预测", status: "未通过", reason: "区间跨0"}, {stage: "前瞻", status: "条件未触发", reason: "历史不计入"}],
    comparison: [{name: "Ridge", return_10bps: null, return_2bps: 0, pure_cost_delta: 0.002, board_weight: null, matched_windows: "0/4"}]
  });
  for (const label of ["无合格新模型", "条件未触发", "未完成", "0.00%", "0.20%", "0/4", "非连续账户收益", "不自动替换现有账户"]) assert.ok(html.includes(label), label);
  assert.ok(html.includes("&lt;script&gt;unsafe&lt;/script&gt;"));
  assert.ok(!html.includes("<script>"));
  assert.ok(!html.includes("<details open"));
});
