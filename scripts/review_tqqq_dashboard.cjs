// Invoke the pinned upstream engine without writing to its checkout.
const fs = require("node:fs");
const path = require("node:path");
const { createRequire } = require("node:module");

const root = path.resolve(process.argv[2]);
const source = path.join(root, "api", "_lib.js");
const upstream = { exports: {} };
const exposeAction = "\nmodule.exports.applySignalAction = applySignalAction;";
new Function("require", "module", "exports", "__filename", "__dirname",
  fs.readFileSync(source, "utf8") + exposeAction)(
  createRequire(source), upstream, upstream.exports, source, path.dirname(source));
const engine = upstream.exports;
const input = JSON.parse(fs.readFileSync(0, "utf8"));
const data = engine.loadSnapshotData();
data.prices = engine.buildPrices(data.nasdaq, data.qqq, data.tqqq, data.rates);
const states = engine.buildStates(data.prices, data.nasdaq, data.vix,
  engine.capeSeriesForBacktest(data.capeLatestFirst).reverse());
const native = engine.runBacktest(data, input.start, 1000, undefined, states, input.end, 5);
const nativeResult = Object.fromEntries(["signal", "tqqq"].map((key) => {
  const book = native.portfolios[key];
  return [key, {
    finalValue: engine.valueOf(book, native.finalPrices),
    maxDrawdown: book.maxDrawdown,
    points: book.points.map((point) => [point.date, point.nav]),
  }];
}));

function singleDeposit(bps) {
  const book = engine.emptyPortfolio(bps);
  const machine = engine.createSignalMachine();
  const rows = [];
  let fills = 0;
  for (let i = 30; i < data.prices.length; i += 1) {
    const date = data.prices[i].date;
    if (date < input.start || date > input.end) continue;
    const { qqqOpen, tqqqOpen, qqqClose, tqqqClose } = input.prices[date];
    const open = { date, qqq: qqqOpen, tqqq: tqqqOpen };
    const close = { date, qqq: qqqClose, tqqq: tqqqClose };
    if (!rows.length) engine.addContribution(book, 100000, open);
    const priorWeight = book.tqqq * tqqqOpen / engine.valueOf(book, open);
    const order = engine.signalMachineDue(machine, date);
    let turnover = 0;
    if (order) {
      const before = [book.qqq, book.tqqq];
      // Upstream uses contribution size as a spending limit. Here the limit
      // comes from existing cash, with an initial capital / 12 monthly budget.
      engine.applySignalAction(book, order.decision, open, 100000 / 12);
      const after = [book.qqq, book.tqqq];
      turnover = Math.abs(after[0] - before[0]) * qqqOpen
        + Math.abs(after[1] - before[1]) * tqqqOpen;
      fills += after.filter((shares, j) => Math.abs(shares - before[j]) > 1e-9).length;
    }
    const openNav = engine.valueOf(book, open);
    engine.signalMachineClose(machine, states[i], date);
    const weight = book.tqqq * tqqqOpen / openNav;
    rows.push({date, nav: engine.valueOf(book, close) / 100000,
      weight, weight_change: weight - priorWeight, turnover: turnover / openNav,
      action: order?.decision.key ?? null});
  }
  if (rows.length !== Object.keys(input.prices).length) throw Error("Upstream market sessions differ");
  return { rows, fills };
}

console.log(JSON.stringify({
  snapshot: engine.buildDataSnapshotId(data),
  native: nativeResult,
  nativeActions: native.actionCounts,
  runs: Object.fromEntries(input.costs.map((bps) => [bps, singleDeposit(bps)])),
}));
