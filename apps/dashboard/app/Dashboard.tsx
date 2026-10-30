"use client";

import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import DesktopExecution from "./DesktopExecution";
import { candidateObservationLabel, candidateOptionLabel, candidateReviewReason, visibleStrategyAccounts } from "./candidate-view";

type Row = Record<string, unknown>;
type Page = "overview" | "sectors" | "research" | "risk" | "leveraged" | "challenger" | "data" | "operations" | "desktop";
type Bank = {
  latest_date: string; latest_scores: Row[]; latest_holdings: Row[];
  scheduler: { enabled: boolean; time: string; timezone: string; next_run?: string };
  recent_runs: { id: number; trade_date: string; source: string; status: string }[];
  timing: { risk_degree: number; active_strategy: string; evidence_note: string };
  shadow: { portfolio: { equity: number; cash: number; positions: Row[] } };
  symbol_names: Record<string, string>;
};
type Universe = {
  sector: string; name: string; fund_code: string; style: string;
  factor_weights: Record<string, number>; ranking: Row[];
};
type Sectors = {
  latest_date: string; disclosure_date: string; evidence_status: string; warning: string;
  allocation_method: string; allocation: Row[]; report: Row[]; universes: Universe[];
  portfolio_policy?: Row; risk_metrics?: Row;
  symbol_names: Record<string, string>;
  strategy_universe?: Row; industry_catalog?: Row[]; board_catalog?: Row[];
  strategy_constituents?: Row[];
  factor_data_quality?: {live?:Row;repair?:Row;comparison?:Row;score_archive?:Row};
};
type Execution = {
  account_id: string; status: string; trade_date?: string;
  target_weights: Record<string, number>; symbol_sectors: Record<string, string>;
  theoretical_target_weights?: Record<string, number>; target_adjustments?: Row[];
  orders: Row[]; fills: Row[]; positions: Row[];
  portfolio: Bank["shadow"]["portfolio"] | null; reconciliation: Row;
  attribution: Row[]; symbol_names: Record<string, string>;
  return_attribution: Row[]; risk_attribution: Row[];
  execution_attribution: Row[]; attribution_reconciliation: Row;
  metrics: Row; history: Row[]; deviations: Row[]; risk_alerts: Row[];
  corporate_actions_today: { registered: Row[]; settled: Row[] };
  corporate_action_ledger: Row[];
  risk_state: Row; risk_transitions: Row[];
};
type DataQuality = {
  latest: Row; checks: Row[]; history: Row[]; symbol_names: Record<string, string>;
};
type OperationsCenter = {
  task_runs: Row[]; task_steps: Row[];
  notifications: (Row & { id: number; acknowledged: boolean })[];
  pending_orders: Row[]; deferred_attempts: Row[]; recent_attempts: Row[];
  scheduler: Row; preview: Row & { steps?: Row[]; expected_effects?: string[]; warnings?: string[] };
  readiness: Row & { checks?: Row[] };
  open_execution?: Row & { accounts?: Row[]; fills?: Row[]; pending_orders?: Row[]; account_ids?: string[] };
};
type IntradayExecution = {
  policy_id: string; activation_date: string; account_id: string;
  pending_orders: Row[]; fills: Row[]; account_ticks: Row[];
  latest: Row & { decisions?: Row[]; executions?: Row[] };
};
type LiveAccount = {
  portfolio: Bank["shadow"]["portfolio"] & { market_value?:number };
  positions: Row[]; orders: Row[]; fills: Row[];
  today_fills: Row[]; pending_orders: Row[];
};
type LiveAccounts = {
  observed_at:string; quote_source:string; symbol_count:number;
  accounts:Record<string,LiveAccount>;
};
type SystemHealth = {
  observed_at:string; system_status:"HEALTHY"|"DEGRADED"|"FROZEN";
  trading_gate:"OPEN"|"FROZEN"; blocking_count:number; warning_count:number;
  checks:Row[];
};
type MarketRisk = {
  status:string; as_of_date:string; effective_for:string; universe_version:string;
  universe_size:number; coverage:number;
  exposure_league:Row & { contestants:Row[]; audit_contestants:Row[]; history:Row[] };
  paper_accounts?:Row;
};
type LeveragedEtfPaper = {
  account_id:string; strategy_id:string; initial_cash:number; currency:string; status:string;
  cash:number; market_value:number; equity:number; total_return:number; drawdown:number;
  positions:Row[]; latest_signal:(Row & { indicators?:Row })|null;
  orders:Row[]; fills:Row[]; history:Row[]; benchmark_history?:Row[]; data_source:string; execution_policy:string;
  scheduler?:Row;
};
type SecurityHistory = {
  symbol:string; name:string; bars:Row[]; fills:Row[];
  coverage: { start_date:string; end_date:string; trading_days:number; fill_count:number };
};
type ModelRegistry = { versions: Row[]; artifacts: Row[]; bindings: Row[] };
type MonthlyAcceptance = {
  cycle_id: string; start_date: string; observation_date: string; status: string;
  progress: number; expected_earliest_completion: string; checks: Row[];
  evidence: Row; report_path?: string | null;
};
type Challenger = {
  diagnostics?: { as_of_date?:string; forward?:{ accounts?:Row[] }; style_attribution?:Row & { accounts?:(Row & { contributions?:Row; skipped_days?:Row[] })[] }; recommendations?:Row[] };
  account_id: string; research: Row & { metrics?: Row[] };
  next_trade_date?: string | null;
  forward: Row & { daily?: Row[] };
  comparison: Row & { metrics?: Row[]; histories?: Record<string, Row[]>; independent_histories?: Record<string, Row[]> };
  latest: Row & { selected?: string[]; scores?: Row[]; portfolio?: Bank["shadow"]["portfolio"] };
  orders: Row[]; fills: Row[]; reconciliation: Row;
  baseline: { account_id: string; latest: Row; orders?: Row[]; fills?: Row[] };
  intraday_baseline?: { account_id:string; status:string; portfolio?:Row; orders:Row[]; fills:Row[] };
  historical_execution: Row & { strategies?: (Row & { summary?: Row })[] };
  point_in_time: Row & { historical_replay_gate?: Row };
  governance: Row & { releases?: Row[]; transitions?: Row[] };
  drift: Row & { breaches?: string[] };
  long_term_review: Row & { evaluation?: Row; criteria?: Record<string, boolean>; cost_ratios?: Row };
  trade_cycle_analysis?: Record<string, Row & { summary?: Row; by_symbol?: Row[]; closed_cycles?: Row[]; open_cycles?: Row[] }>;
  candidate_observation?: {
    status: string; candidate_tag?: string; account_id?: string;
    research_gate_passed?: boolean; latest_trade_date?: string;
    leaderboard: Row[]; latest: Row & { selected?: string[]; portfolio?: Bank["shadow"]["portfolio"] };
    orders: Row[]; fills: Row[]; reconciliation?: Row;
    comparison: Row & { metrics?: Row[]; histories?: Record<string, Row[]> };
    candidates?: (Row & { candidate_tag:string; account_id:string; latest:Row & { portfolio?:Bank["shadow"]["portfolio"] }; orders:Row[]; fills:Row[]; comparison:Challenger["comparison"] })[];
    archived_candidates?: NonNullable<Challenger["candidate_observation"]>["candidates"];
    archived_leaderboard?: Row[];
    global_strategy_view?: Challenger["comparison"] & { strategy_count?:number };
  };
};

const SecurityNavigationContext=createContext<(symbol:string,accountId?:string)=>void>(()=>{});
const SecurityAccountContext=createContext<string>("multi_sector_shadow");

const nav: { id: Page; label: string; note: string }[] = [
  { id: "overview", label: "组合总览", note: "PORTFOLIO" },
  { id: "sectors", label: "行业与个股", note: "SLEEVES" },
  { id: "research", label: "策略研究", note: "MODELS" },
  { id: "risk", label: "市场与仓位", note: "RISK" },
  { id: "leveraged", label: "TQQQ / SQQQ", note: "US ETF PAPER" },
  { id: "challenger", label: "Qlib挑战者", note: "AI LAB" },
  { id: "data", label: "数据健康", note: "QUALITY" },
  { id: "operations", label: "运行与对账", note: "PAPER" },
  { id: "desktop", label: "券商工作台", note: "BROKER WORKSPACE" },
];
const meta: Record<string, { label: string; color: string; thesis: string }> = {
  bank: { label: "银行", color: "#a8e063", thesis: "估值 · 防御 · 动量" },
  dividend: { label: "红利", color: "#f1be62", thesis: "股息 · 低波 · 价值" },
  metals: { label: "工业有色", color: "#db806e", thesis: "周期 · 动量 · 估值" },
  chip: { label: "芯片", color: "#7e91e8", thesis: "成长 · 景气 · 趋势" },
  science50: { label: "科创50", color: "#a07ce5", thesis: "硬科技 · 成长 · 动量" },
  consumer: { label: "主要消费", color: "#e89a5b", thesis: "品牌 · 质量 · 防御" },
  growth: { label: "创业板成长", color: "#53bfb0", thesis: "营收增长 · ROE · 动量" },
};
const industryColors = ["#a8e063","#f1be62","#db806e","#7e91e8","#a07ce5","#e89a5b","#53bfb0","#6fa0a8","#c08366","#809a5d"];
const sectorMeta = (value: unknown) => {
  const key = text(value);
  if (meta[key]) return meta[key];
  const hash = [...key].reduce((sum, char) => (sum * 31 + char.charCodeAt(0)) >>> 0, 0);
  return { label:key, color:industryColors[hash % industryColors.length], thesis:"全局选股后的行业归因 · 不设行业预算" };
};
const factorLabels: Record<string, string> = {
  momentum_252_21: "12-1月动量", return_120: "120日收益", quarterly_revenue_growth: "季度营收增长",
  roe: "ROE", volatility_60: "60日低波", earnings_yield: "盈利收益率", book_to_price: "账面市值比",
  drawdown_120: "120日回撤", dividend_yield_ttm: "股息率", price_to_ma_60: "60日趋势",
};
const pct = (v: unknown) => `${(Number(v ?? 0) * 100).toFixed(1)}%`;
const num = (v: unknown) => Number(v ?? 0).toFixed(2);
const modelScore = (v: unknown) => Number(v ?? 0).toFixed(6);
const text = (v: unknown) => v == null ? "—" : String(v);
const reviewStatusName = (value: unknown) => ({
  OBSERVING:"观察积累中",
  PRELIMINARY_REVIEW:"初步评审中",
  PRELIMINARY_REVIEW_DUE:"等待初步评审",
  CONTINUE_TO_60D:"继续观察至60日",
  EARLY_REJECTED:"提前淘汰",
  FORMAL_REVIEW_READY:"等待正式评审",
  FORMAL_REVIEW_DUE:"等待正式评审",
  PROMOTED:"已晋级",
  REJECTED:"已淘汰",
  GATE_REJECTED:"研究验收未通过",
  REVIEW_EXPIRED:"评审到期归档",
  REVIEW_HOLD:"证据不足·暂停",
  WEIGHT_VALIDATION_HOLD:"权重验证中·暂停交易",
  EXPERIMENTAL_PAPER:"模拟观察中",
  AWAITING_OBSERVATION:"等待首次观察",
}[text(value)]??text(value));
const security = (v: unknown, names: Record<string, string>) => {
  const code = text(v);
  return names[code] ? `${names[code]} · ${code}` : code;
};
const money = (v: unknown) => `¥${Number(v ?? 0).toLocaleString("zh-CN", { maximumFractionDigits: 0 })}`;
const usd = (v: unknown) => `$${Number(v ?? 0).toLocaleString("en-US", { maximumFractionDigits: 2 })}`;
const duration = (v: unknown) => {
  const seconds = Math.max(0, Number(v ?? 0));
  return `${Math.floor(seconds / 3600)}小时${Math.floor((seconds % 3600) / 60)}分`;
};
async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, { cache: "no-store", signal:AbortSignal.timeout(45000), ...init });
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  return response.json() as Promise<T>;
}

export default function Dashboard() {
  const [page, setPage] = useState<Page>("overview");
  const [showArchivedCandidates,setShowArchivedCandidates]=useState(false);
  const [bank, setBank] = useState<Bank | null>(null);
  const [sectors, setSectors] = useState<Sectors | null>(null);
  const [execution, setExecution] = useState<Execution | null>(null);
  const [quality, setQuality] = useState<DataQuality | null>(null);
  const [operations, setOperations] = useState<OperationsCenter | null>(null);
  const [models, setModels] = useState<ModelRegistry | null>(null);
  const [acceptance, setAcceptance] = useState<MonthlyAcceptance | null>(null);
  const [challenger, setChallenger] = useState<Challenger | null>(null);
  const [intraday, setIntraday] = useState<IntradayExecution | null>(null);
  const [live, setLive] = useState<LiveAccounts | null>(null);
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [marketRisk, setMarketRisk] = useState<MarketRisk | null>(null);
  const [leveragedEtf,setLeveragedEtf]=useState<LeveragedEtfPaper|null>(null);
  const [securityDetail,setSecurityDetail]=useState<SecurityHistory|null>(null);
  const [securityDetailAccount,setSecurityDetailAccount]=useState("");
  const [securityDetailLoading,setSecurityDetailLoading]=useState(false);
  const [securityDetailError,setSecurityDetailError]=useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const refreshing = useRef(false);
  const refreshingLive = useRef(false);
  const refresh = useCallback(async () => {
    if(refreshing.current)return;
    refreshing.current=true;
    try {
      const [b, s, e, q, o, m, a, c, i, r, l] = await Promise.all([
        json<Bank>("/api/bank-dashboard"), json<Sectors>("/api/sector-portfolio"), json<Execution>("/api/multi-sector-execution"), json<DataQuality>("/api/data-quality"), json<OperationsCenter>("/api/operations-center"), json<ModelRegistry>("/api/model-registry"), json<MonthlyAcceptance>("/api/monthly-acceptance"), json<Challenger>("/api/qlib-challenger"),
        json<IntradayExecution>("/api/intraday-execution"),
        json<MarketRisk>("/api/market-risk"),
        json<LeveragedEtfPaper>("/api/leveraged-etf-paper"),
      ]);
      setBank(b); setSectors(s); setExecution(e); setQuality(q); setOperations(o); setModels(m); setAcceptance(a); setChallenger(c); setIntraday(i); setMarketRisk(r); setLeveragedEtf(l); setError("");
    } catch (reason) { setError(reason instanceof Error ? reason.message : "服务暂不可用"); }
    finally { refreshing.current=false; }
  }, []);
  const refreshLive = useCallback(async () => {
    if(refreshingLive.current)return;
    refreshingLive.current=true;
    try {
      const [accounts,health]=await Promise.all([
        json<LiveAccounts>("/api/live-accounts"),
        json<SystemHealth>("/api/system-health"),
      ]);
      setLive(accounts); setSystemHealth(health);
    }
    catch { /* retain the last valid live snapshot */ }
    finally { refreshingLive.current=false; }
  }, []);
  useEffect(() => {
    const initial = window.setTimeout(refresh, 0);
    const id = window.setInterval(refresh, 30_000);
    return () => { clearTimeout(initial); clearInterval(id); };
  }, [refresh]);
  useEffect(() => {
    const initial = window.setTimeout(refreshLive, 250);
    const id = window.setInterval(refreshLive, 5_000);
    return () => { clearTimeout(initial); clearInterval(id); };
  }, [refreshLive]);
  const run = async () => {
    setBusy(true);
    try { await json("/api/tasks/daily-run", { method: "POST" }); window.setTimeout(refresh, 1200); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "任务启动失败"); }
    finally { setBusy(false); }
  };
  const openSecurity=useCallback(async(symbol:string,accountId?:string)=>{
    setSecurityDetailLoading(true); setSecurityDetailError("");
    setSecurityDetailAccount(accountId??"");
    try { setSecurityDetail(await json<SecurityHistory>(`http://127.0.0.1:8788/api/security-history?symbol=${encodeURIComponent(symbol)}`)); }
    catch(reason) { setSecurityDetailError(reason instanceof Error?reason.message:"证券历史加载失败"); }
    finally { setSecurityDetailLoading(false); }
  },[]);
  useEffect(() => {
    const symbol=securityDetail?.symbol;
    if(!symbol)return;
    const id=window.setInterval(async()=>{
      try { setSecurityDetail(await json<SecurityHistory>(`http://127.0.0.1:8788/api/security-history?symbol=${encodeURIComponent(symbol)}`)); }
      catch { /* keep the last valid candle while QMT reconnects */ }
    },10_000);
    return ()=>clearInterval(id);
  },[securityDetail?.symbol]);
  if (!bank || !sectors || !execution || !quality || !operations || !models || !acceptance || !challenger || !intraday || !marketRisk || !leveragedEtf) return <main className="loading"><span>M</span><b>MoneyMore</b><p>{error || "正在装载综合组合…"}</p><button onClick={() => void refresh()}>重新连接</button><a href="/desktop-execution">独立打开桌面执行（不等待报表）</a></main>;
  const current = nav.find((item) => item.id === page)!;
  const liveBaseline=live?.accounts[execution.account_id];
  const displayedExecution=liveBaseline?{...execution,portfolio:liveBaseline.portfolio,positions:liveBaseline.positions,orders:liveBaseline.orders,fills:liveBaseline.fills}:execution;
  return <SecurityNavigationContext.Provider value={(symbol,accountId)=>void openSecurity(symbol,accountId)}><div className="app-shell">
    <aside>
      <div className="brand"><span>M</span><div><b>MoneyMore</b><small>MULTI-SECTOR QUANT</small></div></div>
      <div className="system-card"><i/><small>ACTIVE SYSTEM</small><b>全局横截面组合</b><span>统一候选池 · 行业仅作归因</span></div>
      <nav>{nav.map((item, index) => <button key={item.id} className={page === item.id ? "active" : ""} onClick={() => setPage(item.id)}><em>0{index + 1}</em><span>{item.label}<small>{item.note}</small></span></button>)}</nav>
      <div className="aside-foot"><i/> PAPER ONLY<br/><small>{bank.scheduler.time} · {bank.scheduler.timezone}</small></div>
    </aside>
    <main className="workspace">
      <header><div><small>MONEYMORE / {current.note}</small><h1>{current.label}</h1></div><div className="header-actions"><span><i/> {live&&["QMT_REALTIME","ALPHAFEED_REALTIME"].includes(live.quote_source)?(live.quote_source==="ALPHAFEED_REALTIME"?"AlphaFeed":"QMT")+" · "+live.observed_at.slice(11,19):"数据日 "+sectors.latest_date}</span><button onClick={() => { void refresh(); void refreshLive(); }}>刷新</button><button className="primary" disabled={busy} onClick={() => void run()}>{busy ? "恢复中…" : "恢复并重算"}</button></div></header>
      {systemHealth&&<SystemHealthBar health={systemHealth}/>}
      {error && <div className="error">{error}</div>}
      {page === "overview" && <SecurityAccountContext.Provider value="multi_sector_shadow"><Overview sectors={sectors} execution={displayedExecution}/></SecurityAccountContext.Provider>}
      {page === "sectors" && <SecurityAccountContext.Provider value="multi_sector_shadow"><SectorPage sectors={sectors}/></SecurityAccountContext.Provider>}
      {page === "research" && <SecurityAccountContext.Provider value="multi_sector_shadow"><Research sectors={sectors} models={models}/></SecurityAccountContext.Provider>}
      {page === "risk" && <MarketRiskPage risk={marketRisk} live={live??undefined} names={sectors.symbol_names}/>}
      {page === "leveraged" && <LeveragedEtfPage account={leveragedEtf}/>}
      {page === "challenger" && <SecurityAccountContext.Provider value="qlib_gru_shadow"><ChallengerPage challenger={challenger} names={sectors.symbol_names} live={live} showArchived={showArchivedCandidates} onShowArchived={setShowArchivedCandidates}/></SecurityAccountContext.Provider>}
      {page === "data" && <DataHealth quality={quality}/>}
      {page === "desktop" && <DesktopExecution names={sectors.symbol_names}/>}
      {page === "operations" && <SecurityAccountContext.Provider value="multi_sector_shadow"><Operations bank={bank} execution={displayedExecution} operations={operations} acceptance={acceptance} intraday={intraday} live={live} names={sectors.symbol_names} onRefresh={refresh}/></SecurityAccountContext.Provider>}
    </main>
  </div>{(securityDetailLoading||securityDetailError||securityDetail)&&<SecurityHistoryModal detail={securityDetail} initialAccountId={securityDetailAccount} loading={securityDetailLoading} error={securityDetailError} onClose={()=>{setSecurityDetail(null);setSecurityDetailError("");}}/>}</SecurityNavigationContext.Provider>;
}

function LeveragedEtfPage({account}:{account:LeveragedEtfPaper}) {
  const signal=account.latest_signal;
  const indicators=signal?.indicators??{};
  const allocationRows=signal?[{asset:"TQQQ",weight:signal.tqqq},{asset:"SQQQ",weight:signal.sqqq},{asset:"美元现金",weight:signal.cash}]:[];
  const benchmark=account.benchmark_history??[];
  const commonStart=text(benchmark[0]?.trade_date??account.history[0]?.trade_date);
  const strategyAtStart=account.history.find((row)=>text(row.trade_date)===commonStart)??account.history[0];
  const startEquity=Number(strategyAtStart?.equity??account.initial_cash);
  const history=account.history.filter((row)=>text(row.trade_date)>=commonStart).map((row)=>({...row,normalized_nav:Number(row.equity)/startEquity}));
  return <>
    <section className="portfolio-hero leveraged-hero"><div><small>ISOLATED US ETF PAPER ACCOUNT</small><h2>TQQQ Cycle / Cash</h2><p>五年恐慌分位 · 10年期利率动量 · 深度崩盘保护 · 月度滞回再平衡</p></div><div className="exposure"><span>账户净值</span><b>{usd(account.equity)}</b><small>初始资金 {usd(account.initial_cash)} · {account.status}</small></div></section>
    <section className="kpis"><Kpi label="累计收益" value={pct(account.total_return)} note="仅此独立美元账户" accent/><Kpi label="当前回撤" value={pct(account.drawdown)} note="相对账户历史峰值"/><Kpi label="美元现金" value={usd(account.cash)} note={`市值 ${usd(account.market_value)}`}/><Kpi label="最新信号" value={text(signal?.regime??"等待首次运行")} note={`${text(signal?.signal_date)} · 调度${account.scheduler?.running?"正常":"停止"}`}/></section>
    <Panel title="美股独立调度" subtitle="服务启动立即补齐缺失交易日；按 America/New_York 自动处理夏令时，不依赖 A 股流水线"><Table rows={account.scheduler?[account.scheduler]:[]} columns={[["running","运行中"],["last_startup_run","启动补跑"],["open_schedule","开盘执行"],["close_schedule","收盘计算"],["last_open_run","最近开盘任务"],["last_close_run","最近收盘任务"],["last_error","最近错误"]]} format={{running:(value)=>value?"是":"否"}}/></Panel>
    <div className="two-col"><Panel title="最新目标配置" subtitle="收盘生成目标；下一交易日开盘成交，不使用未来数据"><Table rows={allocationRows} columns={[["asset","资产"],["weight","目标权重"]]} format={{weight:pct}}/></Panel><Panel title="周期与仓位诊断" subtitle="五年滚动分位；普通状态只在新月份首个交易日且目标变化达到12%才交易，结构性崩盘立即响应"><Table rows={signal?[{raw_target:indicators.base_tqqq,target_tqqq:signal.tqqq,fear_rank:indicators.fear_rank,rate_rank:indicators.rate_rank,term:indicators.term_structure,rate_change:indicators.rate_change_60d,qqq_close:indicators.qqq_close,ma250:indicators.ma250,distance:indicators.distance_from_ma250,score:signal.score}]:[]} columns={[["raw_target","周期原始仓位"],["target_tqqq","执行目标仓位"],["fear_rank","恐慌分位"],["rate_rank","利率友好分位"],["term","VIX期限比"],["rate_change","利率60日变化"],["qqq_close","QQQ"],["ma250","250日均线"],["distance","距长期均线"],["score","综合分位"]]} format={{raw_target:pct,target_tqqq:pct,fear_rank:num,rate_rank:num,term:num,rate_change:num,qqq_close:num,ma250:num,distance:pct,score:num}}/></Panel></div>
    <div className="two-col"><Panel title="独立持仓" subtitle="不会进入 A 股账户、风控或策略 PK"><Table rows={account.positions} columns={[["symbol","标的"],["quantity","股数"],["avg_cost","均价"],["mark_price","现价"],["market_value","市值"],["unrealized_pnl","浮盈亏"]]} format={{avg_cost:usd,mark_price:usd,market_value:usd,unrealized_pnl:usd}}/></Panel><Panel title="待执行与历史订单" subtitle={account.execution_policy}><Table rows={account.orders.slice(0,30)} columns={[["signal_date","信号日"],["execution_date","执行日"],["symbol","标的"],["side","方向"],["quantity","数量"],["target_weight","目标权重"],["status","状态"],["reason","原因"]]} format={{target_weight:pct}}/></Panel></div>
    <Panel title="美元账户净值" subtitle={account.data_source}>{history.length?<LineComparisonChart title="策略与满仓TQQQ参照" subtitle={`共同起点 ${commonStart} = 100；基准全程满仓持有TQQQ、不调仓`} series={[{accountId:account.account_id,label:"TQQQ Cycle v2",color:"#ef7655",rows:history},{accountId:"tqqq_buy_hold",label:"TQQQ满仓持有",color:"#6478d3",rows:benchmark}]} valueKey="normalized_nav" formatValue={(value)=>`${(value*100).toFixed(2)}`} selectedAccountIds={new Set()} onToggle={()=>{}}/>:<p className="empty">首次运行后开始记录每日净值</p>}</Panel>
    <Panel title="模拟成交" subtitle="5 bps 滑点 + 1 bp 手续费"><Table rows={account.fills} columns={[["trade_date","成交日"],["signal_date","信号日"],["symbol","标的"],["side","方向"],["quantity","数量"],["price","成交价"],["fee","费用"]]} format={{price:usd,fee:usd}}/></Panel>
  </>;
}

function MarketRiskPage({ risk, live, names }: { risk:MarketRisk; live?:LiveAccounts; names:Record<string,string> }) {
  const [highlighted,setHighlighted]=useState<Set<string>>(()=>new Set());
  const toggleHighlighted=(accountId:string)=>setHighlighted((current)=>{
    if(!current.size)return new Set([accountId]);
    const next=new Set(current);
    if(next.has(accountId))next.delete(accountId);else next.add(accountId);
    return next;
  });
  const contestants=risk.exposure_league.contestants??[];
  const byMethod=new Map(contestants.map((row)=>[text(row.method_id),row]));
  const selectedRow=byMethod.get("pysystemtrade_vol_target")??contestants[0];
  const exposureMeta=[
    {id:"pysystemtrade_vol_target",label:"纯波动目标",color:"#7657d5"},
    {id:"pysystemtrade_asymmetric",label:"非对称调仓",color:"#f5a623"},
    {id:"pysystemtrade_trend_cap",label:"趋势风险上限",color:"#19a36a"},
  ];
  const leagueSeries=exposureMeta.map((item)=>({accountId:item.id,label:item.label,color:item.color,rows:(risk.exposure_league.history??[]).filter((row)=>text(row.method_id)===item.id&&Number.isFinite(Number(row.target_exposure)))}));
  const comparison=(risk.exposure_league.strategy_comparison??[]) as Row[];
  const dailyComparison=[...new Map(comparison.map((row)=>[`${text(row.strategy_id)}:${text(row.trade_date)}`,row])).values()];
  const strategySeries=[
    {accountId:"baseline",label:"原基线",color:"#2489e8",rows:dailyComparison.filter((row)=>text(row.strategy_id)==="baseline")},
    {accountId:"baseline_intraday",label:"基线 + 日内实时",color:"#f5a623",rows:dailyComparison.filter((row)=>text(row.strategy_id)==="baseline_intraday")},
    {accountId:"baseline_pysystemtrade",label:"基线 + pysystemtrade",color:"#e84a5f",rows:dailyComparison.filter((row)=>text(row.strategy_id)==="baseline_pysystemtrade")},
    {accountId:"baseline_pysystemtrade_asymmetric",label:"基线 + 非对称调仓",color:"#9b6de3",rows:dailyComparison.filter((row)=>text(row.strategy_id)==="baseline_pysystemtrade_asymmetric")},
    {accountId:"baseline_pysystemtrade_trend_cap",label:"基线 + 趋势上限",color:"#00a6a6",rows:dailyComparison.filter((row)=>text(row.strategy_id)==="baseline_pysystemtrade_trend_cap")},
    {accountId:"baseline_pysystemtrade_intraday",label:"基线 + 仓位控制 + 日内实时",color:"#19a36a",rows:dailyComparison.filter((row)=>text(row.strategy_id)==="baseline_pysystemtrade_intraday")},
  ];
  const paper=(risk.paper_accounts??{}) as Row;
  const baseline=(paper.baseline??{}) as Row;
  const overlay=(paper.baseline_pysystemtrade??{}) as Row;
  const baselineIntraday=(paper.baseline_intraday??{}) as Row;
  const overlayIntraday=(paper.baseline_pysystemtrade_intraday??{}) as Row;
  const asymmetric=(paper.baseline_pysystemtrade_asymmetric??{}) as Row;
  const trendCap=(paper.baseline_pysystemtrade_trend_cap??{}) as Row;
  const baselineId=text(baseline.account_id);
  const overlayId=text(overlay.account_id);
  const baselineIntradayId=text(baselineIntraday.account_id);
  const overlayIntradayId=text(overlayIntraday.account_id);
  const asymmetricId=text(asymmetric.account_id);
  const trendCapId=text(trendCap.account_id);
  const baselineLive=live?.accounts[baselineId];
  const overlayLive=live?.accounts[overlayId];
  const baselineIntradayLive=live?.accounts[baselineIntradayId];
  const overlayIntradayLive=live?.accounts[overlayIntradayId];
  const asymmetricLive=live?.accounts[asymmetricId];
  const trendCapLive=live?.accounts[trendCapId];
  const baselinePortfolio=baselineLive?.portfolio;
  const overlayPortfolio=overlayLive?.portfolio;
  return <>
    <Intro tag="BASELINE × EXPOSURE" title="三路独立总仓位挑战">三条线路共享同一市场代理和基线选股，只比较“总共买多少”：纯波动目标、非对称调仓、趋势风险上限。T日收盘信号从下一交易日生效。</Intro>
    <section className="exposure-methods">{exposureMeta.map((item)=>{const row=byMethod.get(item.id);return row&&<button key={item.id} className="active"><small>{item.label} · 当前建议总仓位</small><b>{pct(row.target_exposure)}</b><span>{text(row.explanation)}</span></button>})}</section>
    <LineComparisonChart title="基线与仓位挑战组合对比" subtitle={`${text(risk.exposure_league.comparison_mode)}；点击图例可单选或多选高亮`} series={strategySeries} valueKey="normalized_nav" formatValue={(value)=>value.toFixed(4)} selectedAccountIds={highlighted} onToggle={toggleHighlighted}/>
    <div className="three-col account-columns">
      <AccountColumn title="纯波动目标" badge="VOL TARGET" accountId={overlayId} status="现役仓位账户" equity={Number(overlayPortfolio?.equity??1_000_000)} portfolio={overlayPortfolio} targetExposure={Number(byMethod.get("pysystemtrade_vol_target")?.target_exposure)} orders={overlayLive?.orders??(overlay.orders as Row[]??[])} fills={overlayLive?.fills??(overlay.fills as Row[]??[])} names={names}/>
      <AccountColumn title="非对称调仓" badge="FAST OFF × SLOW ON" accountId={asymmetricId} status="独立挑战账户" equity={Number(asymmetricLive?.portfolio.equity??1_000_000)} portfolio={asymmetricLive?.portfolio} targetExposure={Number(byMethod.get("pysystemtrade_asymmetric")?.target_exposure)} orders={asymmetricLive?.orders??(asymmetric.orders as Row[]??[])} fills={asymmetricLive?.fills??(asymmetric.fills as Row[]??[])} names={names}/>
      <AccountColumn title="趋势风险上限" badge="VOL × EWMAC CAP" accountId={trendCapId} status="独立挑战账户" equity={Number(trendCapLive?.portfolio.equity??1_000_000)} portfolio={trendCapLive?.portfolio} targetExposure={Number(byMethod.get("pysystemtrade_trend_cap")?.target_exposure)} orders={trendCapLive?.orders??(trendCap.orders as Row[]??[])} fills={trendCapLive?.fills??(trendCap.fills as Row[]??[])} names={names}/>
    </div>
    <div className="four-col account-columns">
      <AccountColumn title="原基线" badge="FACTOR BASELINE" accountId={baselineId} status="正常模拟" equity={Number(baselinePortfolio?.equity??1_000_000)} portfolio={baselinePortfolio} orders={baselineLive?.orders??(baseline.orders as Row[]??[])} fills={baselineLive?.fills??(baseline.fills as Row[]??[])} names={names}/>
      <AccountColumn title="基线 + 日内实时" badge="ADAPTIVE VWAP" accountId={baselineIntradayId} status="日内择价模拟" equity={Number(baselineIntradayLive?.portfolio.equity??1_000_000)} portfolio={baselineIntradayLive?.portfolio} orders={baselineIntradayLive?.orders??(baselineIntraday.orders as Row[]??[])} fills={baselineIntradayLive?.fills??(baselineIntraday.fills as Row[]??[])} names={names}/>
      <AccountColumn title="基线 + pysystemtrade" badge="EXPOSURE OVERLAY" accountId={overlayId} status="独立模拟" equity={Number(overlayPortfolio?.equity??1_000_000)} portfolio={overlayPortfolio} targetExposure={Number(selectedRow?.target_exposure)} orders={overlayLive?.orders??(overlay.orders as Row[]??[])} fills={overlayLive?.fills??(overlay.fills as Row[]??[])} names={names}/>
      <AccountColumn title="仓位控制 + 日内实时" badge="EXPOSURE × VWAP" accountId={overlayIntradayId} status="双层独立模拟" equity={Number(overlayIntradayLive?.portfolio.equity??1_000_000)} portfolio={overlayIntradayLive?.portfolio} targetExposure={Number(selectedRow?.target_exposure)} orders={overlayIntradayLive?.orders??(overlayIntraday.orders as Row[]??[])} fills={overlayIntradayLive?.fills??(overlayIntraday.fills as Row[]??[])} names={names}/>
    </div>
    <LineComparisonChart title="三路策略历史总股票仓位" subtitle={`${text(risk.exposure_league.input_asset)}；${text(risk.exposure_league.history_mode)}`} series={leagueSeries} valueKey="target_exposure" formatValue={pct} yMin={0} yMax={1} selectedAccountIds={new Set()} onToggle={()=>{}}/>
    <Panel title="当前仓位信号" subtitle={`数据日 ${risk.as_of_date} · 覆盖 ${risk.coverage}/${risk.universe_size} 只`}><Table rows={risk.exposure_league.contestants} columns={[["framework","方法"],["status","状态"],["target_exposure","当前总仓位"],["trained_until","截止日期"],["explanation","方法说明"]]} format={{target_exposure:pct}}/></Panel>
  </>;
}

function Overview({ sectors, execution }: { sectors: Sectors; execution: Execution }) {
  const invested = sectors.allocation.reduce((sum, row) => sum + Number(row.target_weight), 0);
  const selected = sectors.allocation.reduce((sum, row) => sum + text(row.selected).split(",").filter(Boolean).length, 0);
  return <>
    <section className="portfolio-hero"><div><span>GLOBAL CROSS-SECTION</span><h2>一个候选池，一套全局排名</h2><p>行业不参与名额和预算。模型全局选股，组合层依据历史价格相关性限制重复风险，行业只在此处汇总展示。</p><div className="hero-tags"><b>全局Top-10</b><b>排名递减权重</b><b>相关风险簇</b><b>T+1 影子执行</b></div></div><div className="exposure"><small>当前股票总仓位</small><b>{pct(invested)}</b><div><i style={{width:`${invested*100}%`}}/></div><span>现金 {pct(1-invested)} · 候选 {selected} 只</span></div></section>
    <section className="kpis"><Kpi label="统一候选池" value={String(sectors.strategy_universe?.size??0)} note={`${text(sectors.strategy_universe?.effective_date)} 起生效 · 市值Top1000`}/><Kpi label="全局持仓" value={String(selected)} note="无行业名额与预算"/><Kpi label="股票目标仓位" value={pct(invested)} note="统一账户风险预算" accent/><Kpi label="覆盖行业" value={String(sectors.industry_catalog?.length??0)} note="选股后反向分类归因"/></section>
    <div className="two-col wide"><Panel title="整体配置全貌" subtitle="Top1000全局选股结果按行业标签汇总，行业不参与策略"><Allocation rows={sectors.allocation}/></Panel><Panel title="当前风险结构" subtitle="相关风险约束优先于普通换手缓冲"><div className="decision"><strong>全局排名 + 动态相关簇约束</strong><p>120日相关系数达到 {num(sectors.portfolio_policy?.cluster_correlation_threshold)} 的股票归入同一风险簇，每簇最多 {text(sectors.portfolio_policy?.maximum_cluster_members)} 只。</p>{sectors.allocation.map((row)=>{const m=sectorMeta(row.sector);return <div key={text(row.sector)}><i style={{background:m.color}}/><span>{m.label}</span><b>{pct(row.target_weight)}</b><small>{text(row.selected).split(",").length}只</small></div>})}</div></Panel></div>
    <Panel title="当前目标持仓的行业归因" subtitle="行业由入选股票反向汇总，不构成行业配额"><SectorCards rows={sectors.allocation} names={sectors.symbol_names}/></Panel>
    <div className="two-col"><Panel title="板块覆盖" subtitle="沪深主板、创业板、科创板统一按总市值排序"><Table rows={sectors.board_catalog??[]} columns={[["board","板块"],["stock_count","股票数"],["market_value","总市值（万元）"]]} format={{market_value:money}}/></Panel><Panel title="候选池行业覆盖 Top 20" subtitle="Tushare当前行业分类，仅作为展示与风险归因"><Table rows={(sectors.industry_catalog??[]).slice(0,20)} columns={[["industry","行业"],["stock_count","股票数"],["market_value","总市值（万元）"]]} format={{market_value:money}}/></Panel></div>
    <Evidence sectors={sectors}/>
  </>;
}

function SectorPage({ sectors }: { sectors: Sectors }) {
  const [industry,setIndustry]=useState("全部行业");
  const constituents=sectors.strategy_constituents??[];
  const visible=industry==="全部行业"?constituents:constituents.filter((row)=>text(row.industry)===industry);
  const selectedCount=constituents.filter((row)=>Boolean(row.selected)).length;
  return <><Intro tag="CURRENT UNIVERSE" title="行业与个股已切换到当前Top1000候选集">沪深主板、创业板和科创板按最新总市值统一入池，再按证券主数据反向归类行业。行业只用于浏览和归因，不拥有预算、名额或独立Top-K。</Intro>
    <section className="kpis"><Kpi label="候选股票" value={String(constituents.length)} note={text(sectors.strategy_universe?.universe_version)}/><Kpi label="反向行业" value={String(sectors.industry_catalog?.length??0)} note="当前证券行业标签"/><Kpi label="全局目标" value={String(selectedCount)} note="基线策略当前入选" accent/><Kpi label="当前筛选" value={String(visible.length)} note={industry}/></section>
    <div className="two-col"><Panel title="行业候选分布" subtitle="按行业总市值排序；目标数来自当前基线全局排名"><Table rows={sectors.industry_catalog??[]} columns={[["industry","行业"],["stock_count","候选数"],["selected_count","当前目标"],["market_value","总市值（万元）"]]} format={{market_value:money}}/></Panel><Panel title="上市板块覆盖" subtitle="四个板块共同参与全局排名"><Table rows={sectors.board_catalog??[]} columns={[["board","板块"],["stock_count","候选数"],["market_value","总市值（万元）"]]} format={{market_value:money}}/></Panel></div>
    <Panel title="当前候选集个股明细" subtitle={`显示 ${visible.length} / ${constituents.length} 只；市值排名为当前候选池排名，因子排名为基线策略全局排名`}><div className="preview-controls"><select value={industry} onChange={(event)=>setIndustry(event.target.value)}><option>全部行业</option>{(sectors.industry_catalog??[]).map((row)=><option key={text(row.industry)}>{text(row.industry)}</option>)}</select><b>{industry}</b></div><Table rows={visible} columns={[["market_cap_rank","市值排名"],["symbol","证券"],["industry","行业"],["board","板块"],["total_mv","总市值（万元）"],["global_rank","因子排名"],["factor_score","因子分"],["selected","当前目标"]]} format={{symbol:(v)=>security(v,sectors.symbol_names),total_mv:money,factor_score:num,selected:(v)=>v?"入选":"—"}}/></Panel>
  </>;
}

function Research({ sectors, models }: { sectors: Sectors; models: ModelRegistry }) {
  const reports=sectors.report.filter((row)=>row.period==="sample_out");
  const constituents=sectors.strategy_constituents??[];
  const factorRanking=[...constituents].sort((a,b)=>Number(a.global_rank??99999)-Number(b.global_rank??99999));
  const selectedCount=constituents.filter((row)=>Boolean(row.selected)).length;
  const quality=sectors.factor_data_quality;
  return <><Intro tag="RESEARCH GOVERNANCE" title="Top1000统一横截面研究体系">因子基线和Qlib模型在统一候选池全局评分，不设行业配额。没有配额不代表风格中性：因子、训练目标及数据缺失仍会影响最终暴露。</Intro>
    <section className="kpis"><Kpi label="统一研究池" value={String(constituents.length)} note="沪深主板 + 创业板 + 科创板"/><Kpi label="模型路线" value="2" note="因子基线 + Qlib GRU"/><Kpi label="当前基线目标" value={String(selectedCount)} note="全局Top-K与风险簇约束" accent/><Kpi label="新口径前瞻起点" value={text(sectors.strategy_universe?.effective_date)} note="此前历史冻结，不回刷"/></section>
    {quality&&<Panel title="因子数据质量与修复进度" subtitle="旧基线保留原口径；全池财务补齐进入隔离研究版本，不自动晋级、不回刷历史"><Table rows={[
      {scope:"现有基线财务",complete:`${text(quality.live?.complete)} / ${text(quality.live?.requested)}`,fresh:text(quality.live?.fresh_complete),state:quality.live?.gate==="PASS"?"完整":"数据不完整（旧版本审计告警）"},
      {scope:"隔离修复版财务",complete:`${text(quality.repair?.complete)} / ${text(quality.repair?.requested)}`,fresh:text(quality.repair?.fresh_complete),state:quality.repair?.status==="PAUSED_ERROR"?"同步暂停，需检查数据源":quality.repair?.gate==="PASS"?"财务覆盖通过，尚未晋级":"限速补齐 / 门禁未通过"},
      {scope:"修复版评分",complete:text(quality.comparison?.complete_factor_rows),fresh:"不生成订单",state:quality.comparison?.status==="RESEARCH_ONLY"?"仅研究，等待前瞻验收":"等待完整数据及一致行情日"}
    ]} columns={[["scope","范围"],["complete","完整数据股票数"],["fresh","新鲜数据数 / 说明"],["state","状态"]]}/><p className="evidence-note">缺失财务不等于基本面差。旧基线仍有按可用因子重新分配权重的历史行为；本轮先补数据并留证，不用行业配额掩盖问题。评分留档：{text(quality.score_archive?.as_of_date)}；市值控制变量：{quality.score_archive?.size_control_present===true?"存在":quality.score_archive?.size_control_present===false?"未实际传入":"等待首次新留档"}。</p></Panel>}
    <Panel title="当前因子基线全局排名" subtitle="全局横截面排名；财务缺失及有效权重差异见数据质量检查，行业列仅用于解释暴露"><Table rows={factorRanking.slice(0,30)} columns={[["global_rank","全局排名"],["symbol","证券"],["industry","行业"],["board","板块"],["factor_score","因子分"],["selected","当前目标"],["target_weight","目标权重"]]} format={{symbol:(v)=>security(v,sectors.symbol_names),factor_score:num,selected:(v)=>v?"入选":"—",target_weight:pct}}/></Panel>
    <Panel title="当前策略架构" subtitle="展示正在运行的研究与组合链路"><div className="architecture">{[["01","版本化候选池","市值Top1000，按月刷新，历史成分不回写"],["02","统一特征层","行情、估值、财务、分红和可交易性按时点对齐"],["03","并行模型层","因子模型与Qlib GRU独立评分、独立账户公平PK"],["04","全局组合层","全局Top-K、退出缓冲、单期替换限制和排名递减权重"],["05","相关风险层","120日收益相关聚类，限制同类风险重复下注"],["06","模拟执行层","T+1撮合、整手、成本、公司行为、对账与收益归因"]].map(([n,t,d])=><div key={n}><b>{n}</b><span><strong>{t}</strong><small>{d}</small></span></div>)}</div></Panel>
    <div className="two-col"><Panel title="候选池行业分布" subtitle="反向分类只用于研究诊断，不形成行业袖套"><Table rows={(sectors.industry_catalog??[]).slice(0,30)} columns={[["industry","行业"],["stock_count","候选数"],["selected_count","当前目标"],["market_value","总市值（万元）"]]} format={{market_value:money}}/></Panel><Panel title="当前组合规则" subtitle="规则来自正在运行的全局组合构建器"><Table rows={Object.entries(sectors.portfolio_policy??{}).map(([rule,value])=>({rule,value:Array.isArray(value)?value.join(", "):value}))} columns={[["rule","规则"],["value","当前值"]]}/></Panel></div>
    <Panel title="模型版本注册表" subtitle="模型版本绑定代码、配置、股票池和数据截止日；任一内容改变都会产生新版本"><Table rows={models.versions} columns={[["version_id","版本"],["lifecycle","生命周期"],["evidence_stage","证据阶段"],["data_cutoff","数据截止"],["universe_hash","股票池指纹"],["code_hash","代码指纹"]]}/></Panel>
    <div className="two-col"><Panel title="证据隔离" subtitle="历史诊断不能自动升级为前瞻证据"><Table rows={models.artifacts.slice(0,20)} columns={[["artifact_type","产物"],["evidence_stage","证据阶段"],["artifact_key","定位"]]}/></Panel><Panel title="信号与订单绑定" subtitle="每个信号和订单都可反查具体模型版本"><Table rows={models.bindings.slice(0,20)} columns={[["trade_date","交易日"],["binding_type","类型"],["symbol","证券"],["version_id","模型版本"]]}/></Panel></div>
    <Panel title="迁移前分行业历史诊断（归档）" subtitle="旧人工候选池研究结果，仅保留审计，不代表当前Top1000策略"><Table rows={reports} columns={[["sector","旧分类"],["cagr","年化"],["volatility","波动率"],["sharpe","夏普"],["max_drawdown","最大回撤"],["fills","成交数"]]} format={{sector:(v)=>meta[text(v)]?.label??text(v),cagr:pct,volatility:pct,sharpe:num,max_drawdown:pct}}/></Panel><Evidence sectors={sectors}/></>;
}

function FactorBaselineV2Info({candidate,names}:{candidate:Row;names:Record<string,string>}) {
  const latest=(candidate.latest??{}) as Row;
  const evidence=(latest.score_evidence??{}) as Row;
  const variants=(evidence.variants??{}) as Record<string,Row>;
  const learned=Boolean((candidate.factor_policy as Row|undefined)?.experimental_linear_model);
  const labels:Record<string,string>={fixed_weight_neutral_missing:learned?"学习权重 · 无中性化":"统一权重与缺失处理",size_comparable:learned?"学习权重 · 市值可比化":"再加市值可比化",industry_size_comparable:learned?"学习权重 · 行业与市值可比化":"再加行业可比化（V2）"};
  const rows=Object.entries(variants).map(([id,v])=>({variant:labels[id]??id,eligible:evidence.eligible,tech:v.top_star_chinext_count,best:v.best_star_chinext_rank,size:v.score_size_correlation}));
  const current=variants.industry_size_comparable??{};
  const validation=(candidate.weight_validation??{}) as Row;
  const styleAudit=(candidate.style_audit??{}) as Row;
  const styleVariants=(styleAudit.variants??{}) as Record<string,Row>;
  const styleNames:Record<string,string>={legacy_current_data_path:"旧基线流程",vendor_repaired_legacy_formula:"修正数据，沿用旧公式",fixed_weight_neutral_missing:"统一缺失处理",size_comparable:"再加市值可比化",industry_size_comparable:"再加行业可比化",learned_industry_size_comparable:"V2 · 冻结学习权重"};
  const styleRows=Object.entries(styleVariants).map(([id,v])=>({step:styleNames[id]??id,eligible:v.eligible,growth:v.growth_board_weight,tech:v.core_information_tech_weight}));
  const scienceRows=(Array.isArray(current.star_chinext_top10)?current.star_chinext_top10 as Row[]:[]).map(row=>({...row,value:Number(row.contribution__earnings_yield??0)+Number(row.contribution__book_to_price??0)+Number(row.contribution__dividend_yield_ttm??0),quality:Number(row.contribution__roe??0),growth:Number(row.contribution__quarterly_revenue_growth??0),trend:Number(row.contribution__momentum_252_21??0)+Number(row.contribution__return_120??0),defense:Number(row.contribution__volatility_60??0)+Number(row.contribution__drawdown_120??0)}));
  return <Panel title={text(candidate.display_name)} subtitle="独立因子基线 V2 · 评分契约验收通过，不代表收益研究通过 · 不自动替换旧基线">
    {styleRows.length>0&&<><h4>风格偏差修正验收</h4><p>候选池 {num(styleAudit.universe_size)} 只 · 有效 {num(styleAudit.eligible)} 只 · 剔除 {num(styleAudit.excluded)} 只。以下为同日、同池、相同组合规则的目标仓位对照，不是实际持仓，也不是收益比较。</p><Table rows={styleRows} columns={[["step","评分处理"],["eligible","有效候选"],["growth","创业板 / 科创板权重"],["tech","电子 / 计算机 / 通信权重"]]} format={{growth:pct,tech:pct}}/><p>板块不等于行业；核心信息科技仅指电子、计算机、通信。没有行业或板块配额。学习系数来自冻结的训练/验证流程，未按近期收益或含科比例重新挑选；仍属独立实验模拟，不自动晋级。</p></>}
    {Array.isArray(candidate.style_exclusions)&&<details><summary>查看本轮数据剔除明细（{candidate.style_exclusions.length}）</summary><p>这些股票未参与本轮评分，不是低分淘汰；不以缺失数据触发持仓卖出。</p><Table rows={candidate.style_exclusions as Row[]} columns={[["symbol","股票"],["industry","行业"],["board","上市板块"],["exclusion_reasons","未评分原因"]]} format={{symbol:v=>security(v,names),exclusion_reasons:v=>text(v).split(";").map(x=>({INCOMPLETE_253_SESSION_HISTORY:"历史行情不足或不连续（253日）",MISSING_CORE_FACTOR:"核心因子缺失",MISSING_OR_STALE_FINANCIALS:"同报告期财务缺失",STALE_MARKET_CROSS_SECTION:"当日截面数据缺失",MISSING_OR_STALE_MARKET_METADATA:"市值/估值元数据缺失",MISSING_INDUSTRY:"行业数据缺失"}[x]??x)).join("；")}}/></details>}
    {Boolean(candidate.validation_hold)&&<><h4>尚非最终 V2：权重验证中，新委托已冻结</h4><p>训练、选参与测试分离，历史样本覆盖和权重增益均需通过；不把未通过验证的系数替换进模拟盘。当前状态：{text(validation.status)}</p><p>固定权重测试 Rank IC：{validation.test_mean_rank_ic_fixed==null?"尚未完成":modelScore(validation.test_mean_rank_ic_fixed)} · 学习权重：{validation.test_mean_rank_ic_learned==null?"尚未完成":modelScore(validation.test_mean_rank_ic_learned)} · 增益置信下界：{validation.rank_ic_improvement_ci_low==null?"尚未完成":modelScore(validation.rank_ic_improvement_ci_low)}。下界未大于零，不认定改善可靠。</p></>}
    <p>估值缺失仅在标准化信号层作中性处理，保留缺失标记，不把原始股息率当零，也不放大其他因子权重。行业与市值用于因子可比性，不规定任何行业或科创/创业板仓位。</p>
    <p>信号数据日：{text(latest.signal_date)} · 计划交易日：{text(latest.planned_trade_date)} · 独立100万元模拟账户 · 不补造上线前成交；模拟成交仅使用启用后的有效实时价格。</p>
    {latest.execution_slippage_bps!=null&&<p>本账户模拟滑点：{num(latest.execution_slippage_bps)} bps（单边，佣金与税费另计）。用于保守评估，不代表实测滑点；旧策略的成本配置与历史收益保持不变。</p>}
    <Table rows={rows} columns={[["variant","处理对照"],["eligible","有效候选"],["tech","前十科创/创业数量"],["best","科创/创业最佳排名"],["size","评分与市值相关"]]} format={{size:num}}/>
    <p>以下是参考评分排名，不是已成交持仓；仍为零含科也不会强制加入。{styleRows.length>0?"本版使用冻结学习权重，尚未证明长期最优；保留未通过的研究诊断，依靠独立前瞻观察评审。":"当前固定因子权重尚未证明最优，后续须用独立前瞻结果和滚动验证判断。"}</p>
    <Table rows={Array.isArray(current.top)?current.top as Row[]:[]} columns={[["symbol","股票"],["industry","行业"],["score","评分"],["observed_weight","已观察因子权重"]]} format={{symbol:v=>security(v,names),score:num,observed_weight:pct}}/>
    <h4>科创 / 创业板候选的评分来源</h4>
    <p>各项为评分贡献，不是收益贡献；没有入选不等于被股票池排除。</p>
    <Table rows={scienceRows} columns={[["symbol","股票"],["rank","全池排名"],["score","总评分"],["value","估值/分红"],["quality","质量"],["growth","增长"],["trend","动量"],["defense","防御"]]} format={{symbol:v=>security(v,names),score:num,value:num,quality:num,growth:num,trend:num,defense:num}}/>
    <h4>待实时重算的参考委托（非已成交）</h4>
    <Table rows={Array.isArray(latest.orders)?latest.orders as Row[]:[]} columns={[["symbol","股票"],["side","方向"],["quantity","参考股数"],["reference_price","收盘参考价"]]} format={{symbol:v=>security(v,names),side:v=>text(v)==="BUY"?"买入":"卖出",reference_price:num}}/>
  </Panel>;
}

function ChallengerPage({ challenger, names, live, showArchived, onShowArchived }: { challenger: Challenger; names: Record<string, string>; live:LiveAccounts|null; showArchived:boolean; onShowArchived:(value:boolean)=>void }) {
  const candidateVersions = [...(challenger.candidate_observation?.candidates??[]),...(showArchived?(challenger.candidate_observation?.archived_candidates??[]):[])];
  const [selectedCandidateTag,setSelectedCandidateTag] = useState("");
  const selectedCandidate = candidateVersions.find((row)=>row.candidate_tag===selectedCandidateTag)??challenger.candidate_observation?.candidates?.find((row)=>Boolean(row.latest_trade_date))??challenger.candidate_observation?.candidates?.[0];
  const selected = new Set(challenger.latest.selected??[]);
  const sectorRanks: Record<string, number> = {};
  const rankedScores = (challenger.latest.scores??[]).map((row) => {
    const sector = text(row.sector);
    sectorRanks[sector] = (sectorRanks[sector]??0) + 1;
    return {...row, sector_rank:sectorRanks[sector], selected:selected.has(text(row.instrument))};
  });
  const challengerLive=live?.accounts[challenger.account_id];
  const baselineLive=live?.accounts[challenger.baseline.account_id];
  const intradayLive=live?.accounts[text(challenger.intraday_baseline?.account_id)];
  const challengerPortfolio = challengerLive?.portfolio??challenger.latest.portfolio;
  const challengerEquity = Number(challengerPortfolio?.equity??1_000_000);
  const baselinePortfolio = (baselineLive?.portfolio??challenger.baseline.latest.portfolio) as Row|undefined;
  const baselineEquity = Number(baselinePortfolio?.equity??1_000_000);
  const observation = challenger.candidate_observation;
  const fixedV2 = [...(observation?.candidates??[]),...(observation?.archived_candidates??[])].find(row=>row.account_id==="qlib_candidate_20261029");
  const fixedV2Live = live?.accounts["qlib_candidate_20261029"];
  const fixedV2Portfolio = fixedV2Live?.portfolio??fixedV2?.latest?.portfolio;
  const candidateLive=live?.accounts[text(selectedCandidate?.account_id??observation?.account_id)];
  const candidatePortfolio = candidateLive?.portfolio??selectedCandidate?.latest?.portfolio??observation?.latest?.portfolio;
  const candidateEquity = Number(candidatePortfolio?.equity??1_000_000);
  const candidateRunStatus=[reviewStatusName(selectedCandidate?.review_stage??observation?.status),selectedCandidate?.latest?.planned_trade_date?`计划交易日 ${text(selectedCandidate.latest.planned_trade_date)}`:"",selectedCandidate?.latest?.execution_slippage_bps!=null?`滑点 ${num(selectedCandidate.latest.execution_slippage_bps)} bps`:""].filter(Boolean).join(" · ");
  const globalView = observation?.global_strategy_view??challenger.comparison;
  return <><Intro tag="QLIB CHALLENGER LAB" title="独立策略，通过公平竞赛验证">原基线保持算法不变，因子V2与深度学习候选使用独立账户、信号、订单和持仓；不自动替换，不重写历史。</Intro>
    <CandidateSwitcher observation={observation} selectedTag={text(selectedCandidate?.candidate_tag)} onSelect={setSelectedCandidateTag} showArchived={showArchived} onShowArchived={onShowArchived}/>
    <div className="four-col account-columns">
      <AccountColumn title="基线" badge="FACTOR BASELINE" accountId={challenger.baseline.account_id} status={text(challenger.baseline.latest.status)} equity={baselineEquity} portfolio={baselinePortfolio} orders={baselineLive?.orders??challenger.baseline.orders??[]} fills={baselineLive?.fills??challenger.baseline.fills??[]} names={names}/>
      <AccountColumn title="基线·日内执行" badge="ADAPTIVE VWAP" accountId={text(challenger.intraday_baseline?.account_id)} status={text(challenger.intraday_baseline?.status)} equity={Number(intradayLive?.portfolio.equity??challenger.intraday_baseline?.portfolio?.equity??0)} portfolio={intradayLive?.portfolio??challenger.intraday_baseline?.portfolio} orders={intradayLive?.orders??challenger.intraday_baseline?.orders??[]} fills={intradayLive?.fills??challenger.intraday_baseline?.fills??[]} names={names}/>
      <AccountColumn title="基线V2" badge="FACTOR V2 · 1029" accountId="qlib_candidate_20261029" status={text(fixedV2?.latest?.status??"等待数据")} equity={Number(fixedV2Portfolio?.equity??1_000_000)} portfolio={fixedV2Portfolio} targetExposure={Number(fixedV2?.latest?.target_gross_exposure)} orders={fixedV2Live?.orders??fixedV2?.orders??[]} fills={fixedV2Live?.fills??fixedV2?.fills??[]} names={names}/>
      <AccountColumn title={text(selectedCandidate?.display_name??"候选者")} badge={text(selectedCandidate?.candidate_tag??observation?.candidate_tag)} accountId={text(selectedCandidate?.account_id??observation?.account_id)} status={candidateRunStatus} equity={candidateEquity} portfolio={candidatePortfolio} targetExposure={Number(selectedCandidate?.latest?.target_gross_exposure??observation?.latest?.target_gross_exposure)} exposurePolicy={(selectedCandidate?.latest?.exposure_policy??observation?.latest?.exposure_policy) as Row|undefined} orders={candidateLive?.orders??selectedCandidate?.orders??observation?.orders??[]} fills={candidateLive?.fills??selectedCandidate?.fills??observation?.fills??[]} names={names} accent={!Boolean(selectedCandidate?.research_gate_passed??observation?.research_gate_passed)}/>
    </div>
    <PerformanceComparisonCharts comparison={globalView} observation={observation} showArchived={showArchived}/>
    <TradeCycleComparison challenger={challenger} names={names} candidateAccountId={selectedCandidate?.account_id}/>
    <p className="muted">原GRU挑战者已淘汰，停止新交易；历史账本保留。第三列固定为基线V2，第四列随顶部选项切换。</p>
  </>;
}

function AccountColumn({ title, badge, accountId, status, equity, portfolio, targetExposure, exposurePolicy, orders, fills, names, accent=false }: { title:string; badge:string; accountId:string; status:string; equity:number; portfolio?:Row; targetExposure?:number; exposurePolicy?:Row; orders:Row[]; fills:Row[]; names:Record<string,string>; accent?:boolean }) {
  const positions = (portfolio?.positions??[]) as Row[];
  const marketValue = Number(portfolio?.market_value??0);
  return <SecurityAccountContext.Provider value={accountId}><article className={`account-column${accent?" account-column-accent":""}`}><header><span>{badge}</span><h3>{title}</h3><small>{accountId}</small></header><div className="account-equity"><b>{money(equity)}</b><span>累计 {pct(equity/1_000_000-1)} · 仓位 {pct(equity?marketValue/equity:0)}{Number.isFinite(targetExposure)?` / 目标 ${pct(targetExposure)}`:""}</span>{exposurePolicy?.realized_volatility!=null&&<small>组合波动 {pct(exposurePolicy.realized_volatility)} · 风险目标 {pct(exposurePolicy.annual_target)}</small>}<i>{status}</i></div><h4>当前持仓 · {positions.length}</h4><Table rows={positions.slice(0,12)} columns={[["symbol","证券"],["quantity","数量"],["avg_cost","成本"],["last","现价"],["market_value","市值"],["unrealized_pnl","浮盈亏"]]} format={{symbol:(v)=>security(v,names),avg_cost:num,last:num,market_value:money,unrealized_pnl:money}}/><h4>最近委托</h4><Table rows={orders.slice(0,8)} columns={[["signal_date","信号日"],["symbol","证券"],["side","方向"],["status","状态"]]} format={{symbol:(v)=>security(v,names)}}/><h4>最近成交</h4><Table rows={fills.slice(-8).reverse()} columns={[["trade_date","成交日"],["symbol","股票"],["side","方向"],["quantity","数量"],["price","成交价"]]} format={{symbol:(v)=>names[text(v)]??"未知证券",price:num}}/></article></SecurityAccountContext.Provider>;
}

function CandidateSwitcher({ observation, selectedTag, onSelect, showArchived, onShowArchived }: { observation:Challenger["candidate_observation"]; selectedTag:string; onSelect:(tag:string)=>void; showArchived:boolean; onShowArchived:(value:boolean)=>void }) {
  if (!observation) return null;
  const active=observation.candidates??[];
  const rows=[...active,...(showArchived?(observation.archived_candidates??[]):[])];
  return <div className="candidate-switcher"><b>第四列候选策略</b><select aria-label="候选策略版本" value={selectedTag} onChange={(event)=>onSelect(event.target.value)}>{rows.length===0&&<option value="">暂无活跃候选</option>}{rows.map((row)=><option key={row.candidate_tag} value={row.candidate_tag}>{candidateOptionLabel(row,reviewStatusName(row.review_stage))}</option>)}</select><button type="button" role="switch" aria-label="查看归档策略" aria-checked={showArchived} className={`candidate-archive-toggle${showArchived?" is-active":""}`} onClick={()=>{const next=!showArchived;onShowArchived(next);if(!next&&!active.some(row=>row.candidate_tag===selectedTag))onSelect(active[0]?.candidate_tag??"");}}><svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6"><path d="M4 8h16v12H4zM3 4h18v4H3zM9 12h6"/></svg><span className="candidate-archive-label">查看归档</span><span className="candidate-archive-count">{observation.archived_candidates?.length??0}</span><span className="candidate-archive-track" aria-hidden="true"><i/></span></button><span>默认仅展示有效候选；20 / 40 / 60共同日评审，退出版本保留收益、委托与成交</span></div>;
}

function ChallengerDiagnostics({challenger,showArchived}:{challenger:Challenger;showArchived:boolean}) {
  const data=challenger.diagnostics;
  if(!data?.forward)return null;
  const observation=challenger.candidate_observation;
  const visible=visibleStrategyAccounts(observation?.candidates??[],observation?.archived_candidates??[],Object.keys(chartPalette),showArchived);
  const recommendations=new Map((data.recommendations??[]).map(row=>[text(row.account_id),row]));
  const forward=(data.forward.accounts??[]).filter(row=>visible.has(text(row.account_id))).map(row=>({...row,recommendation:recommendations.get(text(row.account_id))?.recommendation,review_reason:recommendations.get(text(row.account_id))?.reason}));
  const attribution=(data.style_attribution?.accounts??[]).filter(row=>visible.has(text(row.account_id))).map(row=>({...row,...(row.matched_days?row.contributions:{}),coverage:`${row.matched_days} / ${row.expected_return_days}`}));
  const accountLabel=(value:unknown)=>chartPalette[text(value)]?.label??`候选 ${text(value).replace("qlib_candidate_","")}`;
  const recommendationLabel=(value:unknown)=>({ALREADY_ARCHIVED:"已归档",REPAIR_EVIDENCE_FIRST:"先修复证据",OBSERVE_NO_PROMOTION:"继续观察·不晋级",RESEARCH_REDESIGN:"建议重构研究方案",REVIEW_RISK_EXPOSURE:"优先审查风险暴露",CONTINUE_ISOLATED_REVIEW:"继续隔离评审"}[text(value)]??text(value));
  const nullableNumber=(v:unknown)=>v==null?"—":Number(v).toFixed(4);
  const nullablePercent=(v:unknown)=>v==null?"—":pct(v);
  return <><Panel title="前瞻预测验收与处置建议" subtitle={`${text(data.as_of_date)} · 原预测到5个交易日后才形成标签；事后重建只作诊断，不计入成熟前瞻日。建议不自动修改模型或账户`}><Table rows={forward} columns={[["account_id","策略"],["matured_days","成熟前瞻日"],["pending_days","待标签日"],["incomplete_days","异常日"],["rank_ic","前瞻Rank IC"],["rolling_rank_ic","近20成熟日IC"],["selected_excess_return","平均5日Top-K超额"],["recommendation","处置建议"],["review_reason","理由"]]} format={{account_id:accountLabel,rank_ic:nullableNumber,rolling_rank_ic:nullableNumber,selected_excess_return:nullablePercent,recommendation:recommendationLabel}}/></Panel><Panel title="风格匹配后的收益归因" subtitle={`${text(data.style_attribution?.start_date)} → ${text(data.style_attribution?.end_date)} · 前一日持仓，匹配同板块、市值、20日动量与波动的5只近邻；3/10近邻敏感性另行留档。各项为财富链接贡献，并非因果证明`}><Table rows={attribution} columns={[["account_id","策略"],["coverage","匹配日 / 应有日"],["actual_period_return","同期实际收益"],["market","市场贡献"],["style","风格贡献"],["selection","匹配后选股贡献"],["trading_and_other","交易及其他残差"],["costs","费用贡献"],["reconciliation_error","归因校验差"]]} format={{account_id:accountLabel,actual_period_return:nullablePercent,market:nullablePercent,style:nullablePercent,selection:nullablePercent,trading_and_other:nullablePercent,costs:nullablePercent,reconciliation_error:nullableNumber}}/><p className="empty">使用当时记录的候选池，不用当前行业分类分配预算。交易及其他残差含持仓调整、执行与公司行为差异，不能当作纯择时能力；缺数据日不填零，部分匹配不能冒充完整区间归因。</p></Panel></>;
}

function CandidateObservation({ challenger, selectedTag }: { challenger: Challenger; selectedTag:string }) {
  const observation = challenger.candidate_observation;
  if (!observation) return null;
  const selected = [...(observation.candidates??[]),...(observation.archived_candidates??[])].find((row)=>row.candidate_tag===selectedTag);
  const commonDays = Number(selected?.common_observation_days??observation.comparison?.common_observation_days??0);
  return <>
    <section className="kpis"><Kpi label="有效候选 / 归档" value={`${observation.candidates?.length??0} / ${observation.archived_candidates?.length??0}`} note="退出版本不再生成新委托"/><Kpi label="固定检查点" value="20 / 40 / 60日" note="对两个固定对照，校正多重检验"/><Kpi label="正式评审" value="60共同日" note="冻结新交易；10个市场交易日未评审则归档"/><Kpi label="当前选择" value={selectedTag} note={selected?`${candidateObservationLabel(selected)} · ${candidateReviewReason(selected)}`:`${commonDays} 个共同日`}/></section>
    <Panel title="有效候选模型排行榜" subtitle="固定检查点显著落后两个对照才提前淘汰；研究门禁失败直接退出。新版本加入不重置旧版本计数，也不自动晋级"><Table rows={observation.leaderboard} columns={[["candidate_tag","候选版本"],["data_cutoff","数据截止"],["rank_ic","Rank IC"],["rank_ic_ir","ICIR"],["cost_adjusted_top_k_excess_return","成本后Top-K超额"],["research_gate_passed","研究门禁"],["common_observation_days","运行共同日"],["eligible_observation_days","可信评审日"],["checkpoint","已评审检查点"],["review_stage","评审阶段"],["review_reason","评审结论"]]} format={{rank_ic:num,rank_ic_ir:num,cost_adjusted_top_k_excess_return:pct,research_gate_passed:(v)=>v?"通过":"未通过",review_stage:reviewStatusName}}/></Panel>
    <details className="panel"><summary>归档候选：验收未通过 / 观察期退出（{observation.archived_leaderboard?.length??0}）</summary><p>停止新交易，撤销未成交委托；账本及原有持仓保留，不虚构清仓。勾选顶部“查看归档”可查看持仓与成交详情。</p><Table rows={(observation.archived_leaderboard??[]).map(row=>({...row,review_reason:candidateReviewReason(row),observation_label:candidateObservationLabel(row)}))} columns={[["candidate_tag","版本"],["review_stage","归档类型"],["observation_label","观察情况"],["review_date","评审日"],["review_reason","具体原因"]]} format={{review_stage:reviewStatusName}}/></details>
  </>;
}

function TradeCycleComparison({ challenger, names, candidateAccountId }: { challenger: Challenger; names: Record<string, string>; candidateAccountId?:string }) {
  const analysis = challenger.trade_cycle_analysis??{};
  const accounts = [
    { id:"multi_sector_shadow", label:"因子基线" },
    { id:"multi_sector_intraday_shadow", label:"因子基线·日内执行" },
    { id:"qlib_candidate_20261029", label:"基线V2" },
    ...(candidateAccountId?[{id:candidateAccountId,label:"Qlib 候选者"}]:[]),
  ];
  const summaryRows = accounts.map(({id,label})=>({strategy:label,...(analysis[id]?.summary??{})}));
  return <><Panel title="完整持仓周期胜率与赔率" subtitle="从首次建仓到仓位归零算一个闭合周期；分批买卖合并，费用、滑点和分红计入；未清仓周期不进入胜率"><Table rows={summaryRows} columns={[["strategy","策略"],["closed_cycles","闭合周期"],["wins","盈利"],["losses","亏损"],["win_rate","胜率"],["payoff_ratio","赔率"],["profit_factor","盈亏因子"],["realized_pnl","累计已实现盈亏"],["average_return","平均周期收益"]]} format={{win_rate:pct,payoff_ratio:num,profit_factor:num,realized_pnl:money,average_return:pct}}/></Panel><div className="four-col">{accounts.map(({id,label})=><Panel key={id} title={`${label} · 个股持仓周期`} subtitle="每只股票所有已闭合周期聚合；当前开放周期单独标记"><Table strategyAccountId={id} rows={analysis[id]?.by_symbol??[]} columns={[["symbol","证券"],["closed_cycles","闭合"],["wins","赢"],["losses","亏"],["win_rate","胜率"],["payoff_ratio","赔率"],["profit_factor","盈亏因子"],["realized_pnl","已实现盈亏"],["open_cycle","持仓中"],["open_estimated_pnl","浮动盈亏"]]} format={{symbol:(v)=>security(v,names),win_rate:pct,payoff_ratio:num,profit_factor:num,realized_pnl:money,open_cycle:(v)=>v?"是":"否",open_estimated_pnl:money}}/></Panel>)}</div><div className="four-col">{accounts.map(({id,label})=><Panel key={id} title={`${label} · 已闭合周期明细`} subtitle="用于核验每次从建仓到清仓的真实结果"><Table strategyAccountId={id} rows={(analysis[id]?.closed_cycles??[]).slice(0,100)} columns={[["entry_date","建仓日"],["exit_date","清仓日"],["symbol","证券"],["pnl","净盈亏"],["return_rate","周期收益"],["dividend_income","分红"],["fees","费用"],["fill_count","成交笔数"],["result","结果"]]} format={{symbol:(v)=>security(v,names),pnl:money,return_rate:pct,dividend_income:money,fees:money}}/></Panel>)}</div></>;
}

function ChallengerEvidence({ challenger, showArchived=false }: { challenger: Challenger; showArchived?:boolean }) {
  const commonDays = Number(challenger.comparison.common_observation_days??0);
  const commonStart = text(challenger.comparison.common_start_date);
  const commonEnd = text(challenger.comparison.common_end_date);
  const replayRows = (challenger.historical_execution.strategies??[]).map((row)=>({strategy_id:row.strategy_id,...(row.summary??{})}));
  const globalView=challenger.candidate_observation?.global_strategy_view??challenger.comparison;
  const visibleAccounts=visibleStrategyAccounts(challenger.candidate_observation?.candidates??[],challenger.candidate_observation?.archived_candidates??[],Object.keys(chartPalette),showArchived);
  const globalMetrics=(globalView.metrics??[]).filter(row=>visibleAccounts.has(text(row.account_id)));
  const strategyCount=globalMetrics.length;
  return <><section className="kpis"><Kpi label="候选评审口径" value="版本独立" note="每个候选分别与两个固定对照计算"/><Kpi label="当前候选共同日" value={String(commonDays)} note="不受其他候选加入影响"/><Kpi label="当前共同区间" value={commonStart==="—"?"等待重叠数据":`${commonStart} → ${commonEnd}`} note="用于候选20/60日门禁"/><Kpi label="全局策略数量" value={String(strategyCount)} note={showArchived?"已包含归档策略历史":"仅含有效策略；与顶部查看归档联动"}/></section><Panel title="M5.3 完整历史执行重放" subtitle={`${text(challenger.historical_execution.start_date)} → ${text(challenger.historical_execution.end_date)} · ${text(challenger.historical_execution.evidence_status)} · 与模拟盘共用 PaperBroker`}><Table rows={replayRows} columns={[["strategy_id","策略"],["observation_days","交易日"],["total_return","累计收益"],["average_gross_exposure","平均仓位"],["annualized_volatility","年化波动"],["sharpe","夏普"],["max_drawdown","最大回撤"],["order_count","订单"],["fill_count","成交"],["transaction_cost","交易成本"],["reconciled","对账"]]} format={{total_return:pct,average_gross_exposure:pct,annualized_volatility:pct,sharpe:num,max_drawdown:pct,transaction_cost:money}}/></Panel><div className="two-col"><Panel title="前瞻观察证据" subtitle="预测生成5个交易日后自动成熟标签"><Table rows={challenger.forward.daily??[]} columns={[["observation_date","观察日"],["maturity_date","成熟日"],["rank_ic","Rank IC"],["selected_excess_return","Top-K超额"],["sample_count","样本"]]} format={{rank_ic:num,selected_excess_return:pct}}/></Panel><Panel title={`${strategyCount}策略公平 PK`} subtitle="全局表展示每个策略自身完整运行区间；候选晋级仍使用各版本独立共同日"><Table rows={globalMetrics} columns={[["account_id","账户"],["start_date","开始日"],["end_date","结束日"],["observation_days","运行日"],["total_return","累计收益"],["annualized_volatility","年化波动"],["sharpe","夏普"],["max_drawdown","最大回撤"]]} format={{total_return:pct,annualized_volatility:pct,sharpe:num,max_drawdown:pct}}/></Panel></div></>;
}

const chartPalette: Record<string, { label: string; color: string }> = {
  "multi_sector_shadow": { label: "因子基线", color: "#2489e8" },
  "multi_sector_intraday_shadow": { label: "因子基线·日内执行", color: "#f5a623" },
  "qlib_gru_shadow": { label: "Qlib GRU 挑战者", color: "#e84a5f" },
};
const candidateChartColors=["#8b6fd1","#19a36a","#ef7b45","#d950a0","#00a6a6","#6c7a89","#b98932"];

function PerformanceComparisonCharts({ comparison, observation, showArchived=false }: { comparison: Challenger["comparison"]; observation?:Challenger["candidate_observation"]; showArchived?:boolean }) {
  const candidateNames=Object.fromEntries([...(observation?.candidates??[]),...(observation?.archived_candidates??[])].filter(row=>row.display_name).map(row=>[row.account_id,text(row.display_name)]));
  const histories = comparison.independent_histories??comparison.histories??{};
  const visibleAccounts=visibleStrategyAccounts(observation?.candidates??[],observation?.archived_candidates??[],Object.keys(chartPalette),showArchived);
  const series = Object.entries(histories).filter(([accountId])=>visibleAccounts.has(accountId)&&(accountId!=="qlib_gru_shadow"||showArchived)).map(([accountId, rows],index) => ({
    accountId,
    label: chartPalette[accountId]?.label??candidateNames[accountId]??(accountId.startsWith("qlib_candidate_")?`Qlib 候选者 ${accountId.replace("qlib_candidate_","")}`:accountId),
    color: chartPalette[accountId]?.color??candidateChartColors[index%candidateChartColors.length],
    rows: rows as Row[],
  })).filter((series)=>series.rows.length > 0);
  const [selectedAccountIds,setSelectedAccountIds]=useState<Set<string>>(()=>new Set());
  const toggleAccount=(accountId:string)=>setSelectedAccountIds((current)=>{
    if (!current.size) return new Set([accountId]);
    const next=new Set(current);
    if (next.has(accountId)) next.delete(accountId); else next.add(accountId);
    return next;
  });
  const effectiveSelectedAccountIds=new Set([...selectedAccountIds].filter(id=>visibleAccounts.has(id)));
  const showAll=!effectiveSelectedAccountIds.size;
  return <><div className="strategy-focus" role="group" aria-label="选择需要高亮的策略"><span>高亮策略（可多选）</span><button className={showAll?"active":""} onClick={()=>setSelectedAccountIds(new Set())}>全部</button>{series.map((item)=><button key={item.accountId} aria-pressed={!showAll&&effectiveSelectedAccountIds.has(item.accountId)} className={!showAll&&effectiveSelectedAccountIds.has(item.accountId)?"active":""} onClick={()=>toggleAccount(item.accountId)}><i style={{background:item.color}}/>{item.label}</button>)}</div><div className="two-col"><LineComparisonChart title="累计权益走势" subtitle={showArchived?"含归档策略历史；各策略从自身首个运行日归一为100":"仅有效策略；与顶部查看归档联动，各策略从自身首个运行日归一为100"} series={series} valueKey="normalized_nav" formatValue={(value)=>`${(value*100).toFixed(1)}`} selectedAccountIds={effectiveSelectedAccountIds} onToggle={toggleAccount}/><LineComparisonChart title="每日收益率波动" subtitle={showArchived?"含归档策略历史，未运行区间留空":"已隐藏退出及验收未通过策略；勾选顶部查看归档可回看"} series={series} valueKey="daily_return" formatValue={(value)=>pct(value)} zeroLine selectedAccountIds={effectiveSelectedAccountIds} onToggle={toggleAccount}/></div></>;
}

function LineComparisonChart({ title, subtitle, series, valueKey, formatValue, zeroLine=false, yMin, yMax, selectedAccountIds, onToggle }: { title: string; subtitle: string; series: { accountId: string; label: string; color: string; rows: Row[] }[]; valueKey: string; formatValue: (value: number)=>string; zeroLine?: boolean; yMin?:number; yMax?:number; selectedAccountIds:Set<string>; onToggle:(accountId:string)=>void }) {
  const [hoverIndex,setHoverIndex]=useState<number|null>(null);
  const width = 720, height = 300, left = 12, right = 62, top = 18, bottom = 34;
  const dates = [...new Set(series.flatMap((item)=>item.rows.map((row)=>text(row.trade_date))))].sort();
  const dateIndex = new Map(dates.map((date,index)=>[date,index]));
  const points = series.flatMap((item)=>item.rows.map((row)=>({ date:text(row.trade_date), value:Number(row[valueKey]) })).filter((point)=>Number.isFinite(point.value)));
  if (!points.length) return <Panel title={title} subtitle={subtitle}><p className="empty">等待账户产生实际运行日权益快照</p></Panel>;
  const count = Math.max(dates.length, 1);
  const rawMin = Math.min(...points.map((point)=>point.value), zeroLine ? 0 : Infinity);
  const rawMax = Math.max(...points.map((point)=>point.value), zeroLine ? 0 : -Infinity);
  const padding = Math.max((rawMax - rawMin) * 0.12, zeroLine ? 0.001 : 0.005);
  const min = yMin??rawMin-padding, max = yMax??rawMax+padding, span = Math.max(max - min, 0.000001);
  const x = (date: string) => left + (count <= 1 ? 0 : Number(dateIndex.get(date)??0) / (count - 1)) * (width - left - right);
  const y = (value: number) => top + (max - value) / span * (height - top - bottom);
  const path = (rows: Row[]) => rows.map((row, index)=>`${index ? "L" : "M"}${x(text(row.trade_date)).toFixed(1)},${y(Number(row[valueKey])).toFixed(1)}`).join(" ");
  const last = series.map((item)=>({ ...item, value: Number(item.rows.at(-1)?.[valueKey]??0) }));
  const ticks = [0,.25,.5,.75,1];
  const showAll=!selectedAccountIds.size;
  const ordered=[...series].sort((a,b)=>Number(selectedAccountIds.has(a.accountId))-Number(selectedAccountIds.has(b.accountId)));
  const hoverDate=hoverIndex==null?null:dates[hoverIndex];
  const hoverRows=hoverDate?ordered.map((item)=>({item,row:item.rows.find((row)=>text(row.trade_date)===hoverDate)})).filter((entry)=>entry.row):[];
  const pointer=(event:React.MouseEvent<SVGRectElement>)=>{
    // The event target is the plot rectangle, whose bounds already exclude
    // axis margins. Map its CSS-pixel position directly to the date index.
    const rect=event.currentTarget.getBoundingClientRect();
    if(rect.width<=0||count===0)return;
    const ratio=Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width));
    setHoverIndex(Math.round(ratio*Math.max(count-1,0)));
  };
  return <Panel title={title} subtitle={subtitle}><div className="xq-chart"><div className="chart-legend">{last.map((item)=>{const focused=showAll||selectedAccountIds.has(item.accountId);return <button key={item.accountId} aria-pressed={!showAll&&selectedAccountIds.has(item.accountId)} className={focused?"active":"muted"} onClick={()=>onToggle(item.accountId)}><i style={{background:item.color}}/>{item.label}<b>{formatValue(item.value)}</b></button>})}</div><div className="xq-chart-stage">{hoverDate&&<div className="xq-float" style={{left:`${Math.min(78,Math.max(4,x(hoverDate)/width*100))}%`}}><b>{hoverDate}</b>{hoverRows.map(({item,row})=><span key={item.accountId}><i style={{background:item.color}}/>{item.label}<strong>{formatValue(Number(row?.[valueKey]))}</strong></span>)}</div>}<svg className="comparison-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>{ticks.map((tick)=><g key={tick}><line x1={left} x2={width-right} y1={top+tick*(height-top-bottom)} y2={top+tick*(height-top-bottom)} className="chart-grid"/><text x={width-right+8} y={top+tick*(height-top-bottom)+3}>{formatValue(max-tick*(max-min))}</text></g>)}{dates.filter((_,index)=>index===0||index===dates.length-1||index===Math.floor((dates.length-1)/2)).map((date)=><line key={date} x1={x(date)} x2={x(date)} y1={top} y2={height-bottom} className="chart-grid vertical"/>)}{zeroLine && min <= 0 && max >= 0 && <line x1={left} x2={width-right} y1={y(0)} y2={y(0)} className="chart-zero"/>}{ordered.map((item)=>{const focused=showAll||selectedAccountIds.has(item.accountId);const selected=selectedAccountIds.has(item.accountId);return <g key={item.accountId} opacity={focused?1:.1}><path d={path(item.rows)} fill="none" stroke={item.color} strokeWidth={selected?"3":"2"} strokeLinecap="round" strokeLinejoin="round"/></g>})}{hoverDate&&<g className="chart-crosshair"><line x1={x(hoverDate)} x2={x(hoverDate)} y1={top} y2={height-bottom}/>{hoverRows.map(({item,row})=><circle key={item.accountId} cx={x(hoverDate)} cy={y(Number(row?.[valueKey]))} r="3.5" fill="#fff" stroke={item.color} strokeWidth="2"/>)}</g>}<rect x={left} y={top} width={width-left-right} height={height-top-bottom} fill="transparent" onMouseMove={pointer} onMouseLeave={()=>setHoverIndex(null)}/><text x={left} y={height-8}>{text(dates[0])}</text><text x={(left+width-right)/2} y={height-8} textAnchor="middle">{text(dates[Math.floor((dates.length-1)/2)])}</text><text x={width-right} y={height-8} textAnchor="end">{text(dates.at(-1))}</text></svg></div></div></Panel>;
}

function PointInTimeEvidence({ challenger }: { challenger: Challenger }) {
  const gate = challenger.point_in_time.historical_replay_gate??{};
  return <><section className="kpis"><Kpi label="M5.4 时点存储" value={text(challenger.point_in_time.status)} note={`${text(challenger.point_in_time.feature_rows)} 条特征快照`}/><Kpi label="股票池门禁" value={text(gate.status)} note="成分必须在当时已被系统捕获" accent={text(gate.status)!=="PASS"}/><Kpi label="历史订单覆盖" value={pct(gate.coverage)} note={`${text(gate.eligible_orders)} / ${text(gate.tested_orders)}`}/><Kpi label="可信起点" value={text(gate.earliest_trustworthy_date)} note="此前收益只能作为诊断"/></section><Panel title="时点数据可用规则" subtitle={`数据指纹 ${text(challenger.point_in_time.source_fingerprint).slice(0,16)}…`}><Table rows={Object.entries((challenger.point_in_time.availability_rules??{}) as Row).map(([source,rule])=>({source,rule}))} columns={[["source","数据层"],["rule","可用时间规则"]]}/></Panel></>;
}

function GovernanceEvidence({ challenger }: { challenger: Challenger }) {
  return <><section className="kpis"><Kpi label="M5.5 部署状态" value={text(challenger.governance.lifecycle)} note={text(challenger.governance.release_id)}/><Kpi label="挑战者执行模式" value={text(challenger.governance.execution_mode)} note={text(challenger.governance.reason)} accent={text(challenger.governance.execution_mode)!=="PAPER_TRADING"}/><Kpi label="漂移状态" value={text(challenger.drift.status)} note={`${text(challenger.drift.matured_forward_days)} 个成熟前瞻日`} accent={text(challenger.drift.status)==="BREACH"}/><Kpi label="滚动 Rank IC" value={num(challenger.drift.rolling_rank_ic)} note={`换手 ${pct(challenger.drift.average_turnover)} · 成本率 ${pct(challenger.drift.cost_ratio)}`}/></section><div className="two-col"><Panel title="模型发布注册表" subtitle="候选模型不会自动替换当前挑战者"><Table rows={challenger.governance.releases??[]} columns={[["release_id","版本"],["model_id","模型"],["lifecycle","生命周期"],["data_cutoff","数据截止"],["artifact_hash","产物指纹"]]}/></Panel><Panel title="部署状态转换" subtitle="自动操作只能降级；晋级和恢复必须通过门禁"><Table rows={challenger.governance.transitions??[]} columns={[["created_at","时间"],["from_lifecycle","原状态"],["to_lifecycle","新状态"],["to_mode","执行模式"],["transition_type","类型"],["reason","原因"]]}/></Panel></div><Panel title="漂移监控" subtitle="滚动 IC、预测 PSI、特征 PSI、换手和成本共同决定是否降级"><Table rows={[challenger.drift]} columns={[["status","状态"],["observed_prediction_days","预测日"],["matured_forward_days","成熟日"],["rolling_rank_ic","滚动IC"],["score_psi","预测PSI"],["maximum_feature_psi","特征PSI"],["average_turnover","换手"],["cost_ratio","成本率"],["breaches","触发项"]]} format={{rolling_rank_ic:num,score_psi:num,maximum_feature_psi:num,average_turnover:pct,cost_ratio:pct}}/></Panel></>;
}

function LongTermReview({ challenger }: { challenger: Challenger }) {
  const review=challenger.long_term_review;
  const evaluation=review.evaluation??{};
  const checks=Object.entries(review.criteria??{}).map(([criterion,passed])=>({criterion,status:passed?"PASS":"WAIT"}));
  return <><Intro tag="M5.6 LONG-TERM REVIEW" title="20日只观察，60日后才允许提出晋级">评审只使用2026-07-30之后两个账户共有的可信交易日。Bootstrap置信区间、风险归一收益、回撤、成本、漂移和前瞻门禁必须同时通过；系统永远不会自动替换因子Champion。</Intro><section className="kpis"><Kpi label="评审状态" value={text(review.status)} note={text(review.decision_authority)} accent={text(review.status)!=="ELIGIBLE_FOR_MANUAL_PROMOTION_REVIEW"}/><Kpi label="可信共同日" value={text(evaluation.common_observation_days)} note={`初评 ${text(review.minimum_preliminary_days)} · 正式 ${text(review.minimum_formal_days)}`}/><Kpi label="年化配对超额" value={pct(evaluation.annualized_excess_return)} note={`CI ${pct(evaluation.annualized_excess_ci_low)} → ${pct(evaluation.annualized_excess_ci_high)}`}/><Kpi label="Deflated Sharpe置信度" value={pct(evaluation.deflated_sharpe_probability)} note={`${text(evaluation.trial_count)} 个模型试验修正`}/></section><div className="two-col"><Panel title="风险归一比较" subtitle="统一到10%年化波动后比较"><Table rows={[{account:"因子 Champion",total_return:evaluation.factor_total_return,risk_return:evaluation.factor_risk_normalized_return,volatility:evaluation.factor_annualized_volatility,max_drawdown:evaluation.factor_max_drawdown},{account:"Qlib Challenger",total_return:evaluation.qlib_total_return,risk_return:evaluation.qlib_risk_normalized_return,volatility:evaluation.qlib_annualized_volatility,max_drawdown:evaluation.qlib_max_drawdown}]} columns={[["account","账户"],["total_return","累计收益"],["risk_return","风险归一收益"],["volatility","年化波动"],["max_drawdown","最大回撤"]]} format={{total_return:pct,risk_return:pct,volatility:pct,max_drawdown:pct}}/></Panel><Panel title="正式晋级门禁" subtitle="全部通过也只获得人工评审资格"><Table rows={checks} columns={[["criterion","门禁"],["status","状态"]]}/></Panel></div></>;
}

function DataHealth({ quality }: { quality: DataQuality }) {
  const blocked = Number(quality.latest.blocking_count ?? 0);
  const warnings = Number(quality.latest.warning_count ?? 0);
  const affected = quality.checks.reduce((sum, row) => sum + Number(row.affected_count ?? 0), 0);
  return <><Intro tag="DATA QUALITY CENTER" title="坏数据不能变成订单">每日检查行情、复权、估值、财务、分红、ETF披露和研究产物。阻断级检查未通过时，综合账户不会撮合旧订单，也不会生成新订单。</Intro>
    <section className="kpis"><Kpi label="整体状态" value={text(quality.latest.status)} note={`数据日 ${text(quality.latest.trade_date)}`} accent={blocked>0}/><Kpi label="阻断项" value={String(blocked)} note="必须修复后才能生成订单" accent={blocked>0}/><Kpi label="警告项" value={String(warnings)} note="允许运行但需要跟踪"/><Kpi label="受影响记录" value={String(affected)} note={`${text(quality.latest.target_count)}只目标股票`}/></section>
    <Panel title="数据质量检查矩阵" subtitle="PASS / WARN / BLOCK 全部保留日度证据"><div className="quality-grid">{quality.checks.map((row)=><article key={text(row.code)} className={text(row.status).toLowerCase()}><header><b>{text(row.status)}</b><span>{text(row.category)}</span></header><h3>{text(row.code)}</h3><p>{text(row.message)}</p><small>{Number(row.affected_count)>0?`影响 ${text(row.affected_count)} 只：${text(row.affected_symbols)}`:"未发现异常"}</small></article>)}</div></Panel>
    <Panel title="最近数据健康记录" subtitle="用于观察连续失败和恢复"><Table rows={quality.history} columns={[["trade_date","交易日"],["status","状态"],["target_count","标的数"],["blocking_count","阻断"],["warning_count","警告"]]}/></Panel></>;
}

function Operations({ bank, execution, operations, acceptance, intraday, live, names, onRefresh }: { bank: Bank; execution: Execution; operations: OperationsCenter; acceptance: MonthlyAcceptance; intraday: IntradayExecution; live:LiveAccounts|null; names: Record<string, string>; onRefresh: () => Promise<void> }) {
  const recover = async () => { await json("/api/risk-state/recover", { method: "POST" }); await onRefresh(); };
  const acknowledge = async (id: number) => { await json(`/api/notifications/${id}/acknowledge`, { method: "POST" }); await onRefresh(); };
  const [previewDate, setPreviewDate] = useState(text(operations.preview.trade_date));
  const [preview, setPreview] = useState(operations.preview);
  const inspect = async () => setPreview(await json(`/api/tasks/daily-run/preview?trade_date=${previewDate}`));
  const firstTickEquity=Number(intraday.account_ticks[0]?.equity??0);
  const intradaySeries=intraday.account_ticks.map((row)=>({...row,trade_date:row.observed_at,normalized_nav:firstTickEquity?Number(row.equity)/firstTickEquity:1}));
  const liveAccountRows=Object.values(live?.accounts??{});
  const openExecution:Row & {fills?:Row[];pending_orders?:Row[];accounts?:Row[];account_ids?:string[]}={
    ...(operations.open_execution??{}),
    ...(live?{
      fills:liveAccountRows.flatMap((row)=>row.today_fills),
      pending_orders:liveAccountRows.flatMap((row)=>row.pending_orders),
    }:{}),
  };
  const accountLabel=(value:unknown)=>chartPalette[text(value)]?.label??(text(value).startsWith("qlib_candidate_")?`Qlib 候选者 ${text(value).replace("qlib_candidate_","")}`:text(value));
  return <><section className="ops-banner"><div><i/><small>SERVER SCHEDULER</small><h2>09:31 开盘撮合 · 18:30 收盘流水线</h2><p>开盘执行昨日计划；收盘后更新数据、计算信号并生成下一交易日计划</p></div><span>{bank.scheduler.enabled?"已启用":"已暂停"}</span></section>
    <Panel title="今日开盘执行" subtitle={`${text(openExecution.price_source)} · 纯基线、Qlib挑战者及全部候选者同步在开盘阶段生效`}><section className="kpis"><Kpi label="执行状态" value={text(openExecution.status??"WAITING_FOR_OPEN")} note={`${text(openExecution.trade_date)} · ${text(openExecution.scheduled_time??"09:31")}`}/><Kpi label="覆盖策略" value={String((openExecution.account_ids??[]).length)} note="独立账户、同一开盘价口径"/><Kpi label="今日成交" value={String((openExecution.fills??[]).length)} note="成交后立即进入持仓与权益" accent={Number((openExecution.fills??[]).length)>0}/><Kpi label="遗留待撮合" value={String((openExecution.pending_orders??[]).length)} note="缺行情或交易限制将继续重试" accent={Number((openExecution.pending_orders??[]).length)>0}/></section><Table rows={openExecution.accounts??[]} columns={[["account_id","策略账户"],["pending_before","开盘前委托"],["executions","执行事件"],["reconciliation","对账"]]} format={{account_id:accountLabel,reconciliation:(v)=>Boolean((v as Row)?.matched)?"一致":"异常"}}/><h4>今日开盘成交明细</h4><Table rows={openExecution.fills??[]} columns={[["account_id","策略"],["symbol","证券"],["side","方向"],["quantity","数量"],["price","成交价"],["fee","费用"]]} format={{account_id:accountLabel,symbol:(v)=>security(v,names),price:num,fee:money}}/></Panel>
    <Panel title="基线日内智能执行" subtitle={`${intraday.policy_id} · ${intraday.activation_date}起生效 · 仅改变成交时机，不改变日频目标`}><section className="kpis"><Kpi label="执行状态" value={text(intraday.latest.status??"等待交易时段")} note={text(intraday.latest.observed_at)}/><Kpi label="待处理委托" value={String(intraday.pending_orders.length)} note="基线综合账户" accent={intraday.pending_orders.length>0}/><Kpi label="本轮触发" value={String((intraday.latest.executions??[]).length)} note="VWAP / 价差 / 截止时间"/><Kpi label="本轮观察" value={String((intraday.latest.decisions??[]).length)} note="含持仓、可卖量和成本"/></section><Table strategyAccountId={text(intraday.account_id)} rows={intraday.latest.decisions??[]} columns={[["symbol","证券"],["side","方向"],["quantity","数量"],["price","现价"],["vwap","VWAP"],["spread_bps","价差bps"],["action","动作"],["reason","原因"]]} format={{symbol:(v)=>security(v,names),price:num,vwap:num,spread_bps:num}}/></Panel>
    <div className="two-col"><LineComparisonChart title="日内实时权益变化" subtitle="以当日首个QMT快照归一为100，每分钟刷新" series={[{accountId:intraday.account_id,label:"基线·日内执行",color:"#168a83",rows:intradaySeries}]} valueKey="normalized_nav" formatValue={(value)=>`${(value*100).toFixed(3)}`} selectedAccountIds={new Set([intraday.account_id])} onToggle={()=>{}}/><Panel title="日内实时成交" subtitle="成交后立即进入独立账户账本"><Table rows={intraday.fills.slice(-30).reverse()} columns={[["trade_date","成交日"],["symbol","证券"],["side","方向"],["quantity","数量"],["price","成交价"],["fee","费用"]]} format={{symbol:(v)=>security(v,names),price:num,fee:money}}/></Panel></div>
    <section className="kpis"><Kpi label="综合账户权益" value={money(execution.portfolio?.equity)} note={execution.account_id}/><Kpi label="实际 / 可执行目标" value={`${pct(execution.metrics.gross_exposure)} / ${pct(execution.metrics.target_exposure)}`} note={`理论目标 ${pct(execution.metrics.theoretical_target_exposure??execution.metrics.target_exposure)} · 已考虑一手约束`}/><Kpi label="累计收益 / 回撤" value={`${pct((Number(execution.portfolio?.equity??1_000_000)/1_000_000)-1)} / ${pct(execution.metrics.drawdown)}`} note={`${execution.history.length}个日度快照`}/><Kpi label="账户风险状态" value={text(execution.risk_state.effective_state)} note={text(execution.risk_state.reason_code)} accent={text(execution.risk_state.effective_state)!=="NORMAL"}/></section>
    <Panel title="理论目标到可执行目标" subtitle="一手金额超过可分配预算的股票被剔除；释放预算优先在同一行业内重新分配"><Table rows={execution.target_adjustments??[]} columns={[["symbol","证券"],["sector","行业"],["theoretical_weight","理论权重"],["executable_weight","可执行权重"],["minimum_lot_weight","一手占账户"],["reason","调整原因"]]} format={{symbol:(v)=>security(v,names),sector:(v)=>meta[text(v)]?.label??text(v),theoretical_weight:pct,executable_weight:pct,minimum_lot_weight:pct}}/></Panel>
    <section className="kpis"><Kpi label="下次自动运行" value={duration(operations.scheduler.seconds_to_next_run)} note={text(operations.scheduler.next_run)}/><Kpi label="待执行订单" value={String(operations.pending_orders.length)} note={`${text(preview.eligible_for_execution_count)}笔在所选交易日可撮合`} accent={operations.pending_orders.length>0}/><Kpi label="延期执行记录" value={String(operations.deferred_attempts.length)} note="涨跌停、T+1、现金不足"/><Kpi label="补跑预览" value={text(preview.market_session)} note={`${text(preview.trade_date)} · 只读检查`}/></section>
    <Panel title="运行保障状态" subtitle="正式交易日历、服务端调度和后台线程必须在流水线启动前可恢复"><div className="acceptance-head"><b>{text(operations.readiness.status)}</b><span><strong>{text(operations.readiness.market_session)}</strong><small>{text(operations.readiness.trade_date)} · {text(operations.readiness.repair_count)}项待自动修复</small></span><em>{Number(operations.readiness.blocking_count)>0?"存在阻断":"允许启动预检"}</em></div><div className="quality-grid">{(operations.readiness.checks??[]).map((row)=><article key={text(row.code)} className={text(row.status)==="PASS"?"pass":"warn"}><header><b>{text(row.status)}</b><span>{text(row.code)}</span></header><p>{text(row.message)}</p></article>)}</div></Panel>
    <Panel title="完整月度换仓验收" subtitle={`${acceptance.cycle_id} · 最早完成估算 ${acceptance.expected_earliest_completion}`}><div className="acceptance-head"><b>{pct(acceptance.progress)}</b><span><strong>{acceptance.status}</strong><small>观察期 {acceptance.start_date} → {acceptance.observation_date}</small></span><em>{acceptance.report_path?"验收报告已冻结":"证据积累中"}</em></div><div className="quality-grid">{acceptance.checks.map((row)=><article key={text(row.code)} className={Boolean(row.passed)?"pass":"warn"}><header><b>{Boolean(row.passed)?"PASS":"WAIT"}</b><span>{text(row.code)}</span></header><h3>{text(row.observed)} / {text(row.required)}</h3><p>{text(row.explanation)}</p></article>)}</div></Panel>
    <div className="two-col"><Panel title="待执行订单" subtitle="显示信号日、方向、数量和进入下一撮合窗口的资格"><Table rows={operations.pending_orders.slice(0,30)} columns={[["signal_date","信号日"],["symbol","证券"],["side","方向"],["quantity","数量"],["status","状态"],["reason_code","策略原因"]]} format={{symbol:(v)=>security(v,names)}}/></Panel><Panel title="延期与拒绝原因" subtitle="保留每次执行尝试，不覆盖此前失败原因"><Table rows={operations.deferred_attempts.slice(0,30)} columns={[["trade_date","交易日"],["symbol","证券"],["side","方向"],["outcome","结果"],["reason_code","原因"],["quantity","数量"]]} format={{symbol:(v)=>security(v,names)}}/></Panel></div>
    <Panel title="手动补跑影响预览" subtitle="预览不会启动任务、写入数据或改变模拟持仓"><div className="preview-controls"><input value={previewDate} onChange={(event)=>setPreviewDate(event.target.value)} maxLength={8}/><button className="recover" onClick={()=>void inspect()}>检查影响</button><b>{text(preview.calendar_source)}</b></div><section className="kpis"><Kpi label="历史运行" value={text(preview.prior_run_count)} note="该交易日已有任务数"/><Kpi label="全部待执行" value={text(preview.pending_order_count)} note="综合影子账户"/><Kpi label="预计可撮合" value={text(preview.eligible_for_execution_count)} note={`${text(preview.eligible_buy_count)}买 / ${text(preview.eligible_sell_count)}卖`}/><Kpi label="最近影子账本" value={text(preview.latest_shadow_date)} note="补跑前基线"/></section><Table rows={preview.steps??[]} columns={[["step_name","步骤"],["action","补跑动作"]]}/>{(preview.expected_effects??[]).map((item)=><p key={item}>· {item}</p>)}{(preview.warnings??[]).map((item)=><p className="error" key={item}>{item}</p>)}</Panel>
    <Panel title="账户风险状态机" subtitle="风险自动升级；状态降级必须人工确认"><div className="risk-machine">{["NORMAL","REDUCE_ONLY","SELL_ONLY","SUSPENDED"].map((state)=><div key={state} className={execution.risk_state.effective_state===state?"active":""}><b>{state}</b><small>{state==="NORMAL"?"允许正常调仓":state==="REDUCE_ONLY"?"只允许降低风险":state==="SELL_ONLY"?"目标强制降为零":"冻结全部订单"}</small></div>)}</div>{Boolean(execution.risk_state.recovery_required)&&<button className="recover" onClick={()=>void recover()}>确认恢复到 {text(execution.risk_state.proposed_state)}</button>}<Table rows={execution.risk_transitions.slice(0,10)} columns={[["trade_date","交易日"],["from_state","原状态"],["to_state","生效状态"],["proposed_state","建议状态"],["reason_code","原因"],["transition_type","类型"]]}/></Panel>
    <Panel title="风险监控" subtitle="阻断级告警会停止生成新订单"><div className="risk-alerts">{execution.risk_alerts.length?execution.risk_alerts.map((row)=><div key={text(row.code)} className={text(row.severity).toLowerCase()}><b>{text(row.severity)}</b><span>{text(row.message)}</span><small>{text(row.code)}</small></div>):<p className="empty">当前无风险告警</p>}</div></Panel>
    <Panel title="可恢复步骤账本" subtitle="每个步骤独立重试；同一交易日已完成步骤在重跑时自动跳过"><Table rows={operations.task_steps.slice(0,30)} columns={[["run_id","运行"],["trade_date","交易日"],["step_name","步骤"],["attempt","尝试"],["status","状态"],["error","错误"]]}/></Panel>
    <div className="two-col"><Panel title="每日综合流水线"><div className="architecture compact">{["交易日校验","全市场数据同步","五行业因子截面","行业择时与风险预算","影子撮合","现金持仓对账"].map((name,index)=><div key={name}><b>0{index+1}</b><span><strong>{name}</strong><small>{index<4?"组合决策层":"执行审计层"}</small></span></div>)}</div></Panel><Panel title="最近任务"><Table rows={operations.task_runs.slice(0,15)} columns={[["trade_date","交易日"],["source","来源"],["status","状态"],["started_at","开始"],["finished_at","结束"],["error","错误"]]}/></Panel></div>
    <div className="two-col"><Panel title="行业收益与持仓归因" subtitle="当日贡献、实际仓位和浮动盈亏"><Table rows={execution.attribution} columns={[["sector","行业"],["daily_pnl","当日盈亏"],["daily_contribution","收益贡献"],["actual_weight","实际仓位"],["unrealized_pnl","浮动盈亏"]]} format={{sector:(v)=>meta[text(v)]?.label??text(v),daily_pnl:money,daily_contribution:pct,actual_weight:pct,unrealized_pnl:money}}/></Panel><Panel title="目标与实际仓位偏差" subtitle="解释未成交、整手和T+1差异"><Table rows={execution.deviations} columns={[["symbol","证券"],["target_weight","目标"],["actual_weight","实际"],["weight_gap","偏差"],["reason","原因"]]} format={{symbol:(v)=>security(v,names),target_weight:pct,actual_weight:pct,weight_gap:pct}}/></Panel></div>
    <Panel title="完整收益归因" subtitle={`账户盈亏 ${money(execution.attribution_reconciliation.actual_pnl)} · 已解释 ${money(execution.attribution_reconciliation.explained_pnl)} · 未解释 ${money(execution.attribution_reconciliation.unexplained_pnl)}`}><Table rows={execution.return_attribution} columns={[["component","贡献来源"],["sector","行业"],["pnl","盈亏"],["contribution","收益贡献"],["detail","口径"]]} format={{sector:(v)=>meta[text(v)]?.label??text(v),pnl:money,contribution:pct}}/></Panel>
    <div className="two-col"><Panel title="组合风险归因" subtitle="波动、相关性、集中度与边际风险"><Table rows={execution.risk_attribution} columns={[["symbol","证券"],["sector","行业"],["weight","权重"],["volatility_contribution","波动贡献"],["correlation_contribution","相关性贡献"],["concentration_contribution","集中度"],["marginal_risk","边际风险"]]} format={{symbol:(v)=>security(v,names),sector:(v)=>meta[text(v)]?.label??text(v),weight:pct,volatility_contribution:pct,correlation_contribution:pct,concentration_contribution:pct,marginal_risk:num}}/></Panel><Panel title="执行缺口归因" subtitle="整手、T+1、涨跌停、现金不足和其他执行约束"><Table rows={execution.execution_attribution} columns={[["reason","原因"],["symbol_count","证券数"],["absolute_weight_gap","绝对仓位缺口"],["signed_weight_gap","净缺口"],["execution_events","执行事件"]]} format={{absolute_weight_gap:pct,signed_weight_gap:pct}}/></Panel></div>
    <Panel title="当前综合持仓" subtitle="股票名、代码、实际数量和成本"><Table rows={execution.positions} columns={[["symbol","证券"],["quantity","数量"],["available_quantity","可用"],["avg_cost","成本"]]} format={{symbol:(v)=>security(v,names),avg_cost:num}}/></Panel>
    <Panel title="公司行为入账" subtitle="现金分红、送转股和权益登记均采用幂等审计账本"><Table rows={execution.corporate_action_ledger} columns={[["trade_date","入账日"],["symbol","证券"],["action_type","类型"],["entitled_quantity","登记股数"],["cash_amount","现金"],["share_quantity","新增股数"]]} format={{symbol:(v)=>security(v,names),cash_amount:money}}/></Panel>
    <Panel title="最近委托与成交" subtitle="五行业目标统一进入影子账户，真实委托保持隔离"><Table rows={execution.orders.slice(0,30)} columns={[["signal_date","信号日"],["symbol","证券"],["side","方向"],["quantity","数量"],["status","状态"],["reason_code","原因"]]} format={{symbol:(v)=>security(v,names)}}/></Panel>
    <Panel title="任务告警中心" subtitle="流水线失败、数据阻断、风控暂停和延迟订单会在这里留痕"><div className="risk-alerts">{operations.notifications.length?operations.notifications.map((row)=><div key={row.id} className={text(row.severity).toLowerCase()}><b>{text(row.severity)}</b><span><strong>{text(row.title)}</strong><br/>{text(row.message)}</span><small>{text(row.trade_date)} · {text(row.code)} {!row.acknowledged&&<button className="recover" onClick={()=>void acknowledge(row.id)}>确认</button>}</small></div>):<p className="empty">当前无任务告警</p>}</div></Panel></>;
}

function Allocation({rows}:{rows:Row[]}){return <div className="allocation">{rows.map((row)=>{const m=sectorMeta(row.sector);return <div key={text(row.sector)}><div className="allocation-name"><i style={{background:m.color}}/><span><b>{m.label}</b><small>{m.thesis}</small></span></div><div className="bars"><span><i style={{width:`${Number(row.target_weight)*100}%`,background:`${m.color}55`}}/></span><span><i style={{width:`${Number(row.target_weight)*100}%`,background:m.color}}/></span></div><div className="values"><b>{pct(row.target_weight)}</b><small>全局选股后的行业暴露 · 非预算约束</small></div></div>})}</div>}
function SectorCards({rows,names}:{rows:Row[];names:Record<string,string>}){return <div className="sector-cards">{rows.map((row)=>{const m=sectorMeta(row.sector);return <article key={text(row.sector)} style={{"--sector":m.color} as React.CSSProperties}><header><span>{m.label}</span><b>{pct(row.target_weight)}</b></header><p>{m.thesis}</p><div>{text(row.selected).split(",").map((s)=><small key={s}><b>{names[s]??"未知证券"}</b><span>{s}</span></small>)}</div><footer><span>实际目标暴露 {pct(row.target_weight)}</span><span>仅展示汇总</span></footer></article>})}</div>}
function UniverseCard({universe,names}:{universe:Universe;names:Record<string,string>}){const m=sectorMeta(universe.sector);const selectedCount=universe.ranking.filter((row)=>Boolean(row.selected)).length;return <article className="universe-card" style={{"--sector":m.color} as React.CSSProperties}><header><div><span>{m.label}</span><h3>{universe.name}</h3><small>{universe.fund_code} · {universe.style}</small></div><b>{selectedCount}<small>目标持仓</small></b></header><div className="factor-chips">{Object.entries(universe.factor_weights).map(([f,w])=><span key={f}>{factorLabels[f]??f}<b>{pct(w)}</b></span>)}</div><Table rows={universe.ranking.slice(0,universe.sector==="bank"?12:10)} columns={[["rank","排名"],["symbol","证券"],["score","模型分"],["selected","目标"]]} format={{symbol:(v)=>security(v,names),score:num,selected:(v)=>v?"持有":"—"}}/></article>}
function Evidence({sectors}:{sectors:Sectors}){return <div className="evidence"><b>证据边界</b><p>{sectors.warning}</p><span>{sectors.evidence_status}</span></div>}
function Intro({tag,title,children}:{tag:string;title:string;children:ReactNode}){return <section className="intro"><span>{tag}</span><h2>{title}</h2><p>{children}</p></section>}
function SecurityHistoryModal({detail,initialAccountId,loading,error,onClose}:{detail:SecurityHistory|null;initialAccountId:string;loading:boolean;error:string;onClose:()=>void}){
  const [range,setRange]=useState("3M");
  const [accountId,setAccountId]=useState(initialAccountId);
  const [hovered,setHovered]=useState<(Row&{left:number;top:number})|null>(null);
  const [hoveredBar,setHoveredBar]=useState<number|null>(null);
  // A newly selected security is a new chart session, so reset transient UI.
  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(()=>{setRange("3M");setAccountId(initialAccountId);setHovered(null);setHoveredBar(null)},[detail?.symbol,initialAccountId]);
  if(loading)return <div className="security-modal-backdrop" onClick={onClose}><section className="security-modal loading-detail" onClick={(event)=>event.stopPropagation()}><b>正在载入K线与交易记录…</b></section></div>;
  if(error)return <div className="security-modal-backdrop" onClick={onClose}><section className="security-modal loading-detail" onClick={(event)=>event.stopPropagation()}><button className="modal-close" onClick={onClose}>×</button><b>{error}</b></section></div>;
  if(!detail)return null;
  const barCount=range==="1M"?20:range==="3M"?60:range==="6M"?120:range==="1Y"?250:range==="3Y"?750:detail.bars.length;
  const accountIds=[...new Set([initialAccountId,...detail.fills.map((row)=>text(row.account_id))].filter(Boolean))];
  const selectedAccountId=accountIds.includes(accountId)?accountId:accountIds[0]??"";
  const strategyFills=detail.fills.filter((row)=>text(row.account_id)===selectedAccountId);
  const bars=detail.bars.slice(-barCount);
  const visibleDates=new Set(bars.map((row)=>text(row.trade_date)));
  const fills=strategyFills.filter((row)=>visibleDates.has(text(row.trade_date)));
  const width=1080,height=520,left=12,right=66,top=22,priceBottom=365,volumeTop=395,volumeBottom=486;
  const values=[...bars.flatMap((row)=>[Number(row.high),Number(row.low)]),...fills.map((row)=>Number(row.price))].filter(Number.isFinite);
  const rawMin=Math.min(...values),rawMax=Math.max(...values),pricePad=Math.max((rawMax-rawMin)*.05,.01),min=rawMin-pricePad,max=rawMax+pricePad,span=Math.max(max-min,0.01);
  const step=(width-left-right)/Math.max(bars.length,1),body=Math.max(1,Math.min(6,step*.62));
  const x=(index:number)=>left+(index+.5)*step;
  const y=(value:number)=>top+(max-value)/span*(priceBottom-top);
  const maxVolume=Math.max(...bars.map((row)=>Number(row.vol??row.volume??0)),1);
  const volumeY=(value:number)=>volumeBottom-value/maxVolume*(volumeBottom-volumeTop);
  const movingAverage=(period:number)=>bars.map((_,index)=>index+1<period?null:bars.slice(index-period+1,index+1).reduce((sum,row)=>sum+Number(row.close),0)/period);
  const averages=[{period:5,color:"#f5a623",values:movingAverage(5)},{period:10,color:"#2489e8",values:movingAverage(10)},{period:20,color:"#d950a0",values:movingAverage(20)},{period:60,color:"#19a36a",values:movingAverage(60)}];
  const averagePath=(items:(number|null)[])=>items.map((value,index)=>value==null?null:`${items.slice(0,index).some((item)=>item!=null)?"L":"M"}${x(index).toFixed(1)},${y(value).toFixed(1)}`).filter(Boolean).join(" ");
  const dateIndex=new Map(bars.map((row,index)=>[text(row.trade_date),index]));
  const accountName=(value:unknown)=>chartPalette[text(value)]?.label??(text(value).startsWith("qlib_candidate_")?`Qlib候选 ${text(value).replace("qlib_candidate_","")}`:text(value));
  const activeBar=hoveredBar==null?bars.at(-1):bars[hoveredBar];
  const chartPointer=(event:React.MouseEvent<SVGRectElement>)=>{
    // The event target is the plot-area rect, not the complete SVG.  Mapping
    // its CSS pixel offset through the full viewBox width introduced a growing
    // rightward crosshair error.  Map directly to the number of visible bars.
    const rect=event.currentTarget.getBoundingClientRect();
    const ratio=Math.max(0,Math.min(1,(event.clientX-rect.left)/Math.max(rect.width,1)));
    setHoveredBar(Math.max(0,Math.min(bars.length-1,Math.floor(ratio*bars.length))));
  };
  return <div className="security-modal-backdrop" onClick={onClose}><section className="security-modal xq-security" onClick={(event)=>event.stopPropagation()}><header><div><small>行情 · 模拟成交复盘</small><h2>{detail.name}<em>{detail.symbol}</em></h2><p>{detail.coverage.start_date} → {detail.coverage.end_date} · {detail.coverage.trading_days}个交易日 · 当前策略{strategyFills.length}笔模拟成交</p></div><button className="modal-close" onClick={onClose}>×</button></header><div className="security-filters"><label>策略<select value={selectedAccountId} onChange={(event)=>{setAccountId(event.target.value);setHovered(null)}}>{accountIds.map((id)=><option key={id} value={id}>{accountName(id)}</option>)}</select></label><div className="range-tabs">{["1M","3M","6M","1Y","3Y","ALL"].map((item)=><button key={item} className={range===item?"active":""} onClick={()=>setRange(item)}>{item}</button>)}</div></div><div className="candlestick-wrap"><div className="xq-quote-strip"><b>{text(activeBar?.trade_date)}</b><span>开 <i>{num(activeBar?.open)}</i></span><span>高 <i className="up">{num(activeBar?.high)}</i></span><span>低 <i className="down">{num(activeBar?.low)}</i></span><span>收 <i>{num(activeBar?.close)}</i></span>{averages.map((average)=><span key={average.period} style={{color:average.color}}>MA{average.period} {num(average.values[hoveredBar??bars.length-1])}</span>)}</div>{hovered&&<div className="trade-tooltip" style={{left:`${hovered.left}%`,top:`${hovered.top}%`}}><b className={text(hovered.side).toLowerCase()}>{text(hovered.side)==="BUY"?"买入 BUY":"卖出 SELL"}</b><span>{text(hovered.trade_date)} · {accountName(hovered.account_id)}{hovered.inherited?" · 继承自因子基线":""}</span><strong>成交价 {Number(hovered.price).toFixed(3)}</strong><span>数量 {Number(hovered.quantity).toLocaleString("zh-CN")}股 · 费用 {money(hovered.fee)}</span></div>}<svg className="candlestick-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${detail.name} K线与买卖记录`}>{[0,.25,.5,.75,1].map((tick)=><g key={tick}><line x1={left} x2={width-right} y1={top+tick*(priceBottom-top)} y2={top+tick*(priceBottom-top)} className="chart-grid"/><text x={width-right+8} y={top+tick*(priceBottom-top)+3}>{(max-tick*span).toFixed(2)}</text></g>)}<line x1={left} x2={width-right} y1={volumeTop-12} y2={volumeTop-12} className="chart-divider"/><text x={left} y={volumeTop-16}>成交量</text>{bars.map((row,index)=>{const up=Number(row.close)>=Number(row.open),color=up?"#f0484d":"#19a36a",topY=y(Math.max(Number(row.open),Number(row.close))),bottomY=y(Math.min(Number(row.open),Number(row.close))),volume=Number(row.vol??row.volume??0);return <g key={text(row.trade_date)}><line x1={x(index)} x2={x(index)} y1={y(Number(row.high))} y2={y(Number(row.low))} stroke={color}/><rect x={x(index)-body/2} y={topY} width={body} height={Math.max(bottomY-topY,1)} fill={up?"#fff":color} stroke={color}/><rect x={x(index)-body/2} y={volumeY(volume)} width={body} height={volumeBottom-volumeY(volume)} fill={up?"#f0484d":"#19a36a"} opacity=".86"/></g>})}{averages.map((average)=><path key={average.period} d={averagePath(average.values)} fill="none" stroke={average.color} strokeWidth="1.35"/>)}<rect className="chart-hit-area" x={left} y={top} width={width-left-right} height={volumeBottom-top} fill="transparent" onMouseMove={chartPointer} onMouseLeave={()=>setHoveredBar(null)}/>{fills.map((fill,index)=>{const tradeDate=text(fill.trade_date),pointIndex=dateIndex.get(tradeDate);if(pointIndex==null)return null;const px=x(pointIndex),priceY=y(Number(fill.price)),buy=text(fill.side)==="BUY",lane=fills.slice(0,index).filter((row)=>text(row.trade_date)===tradeDate&&(text(row.side)==="BUY")===buy).length,badgeY=buy?Math.min(priceBottom-14,priceY+28+lane*22):Math.max(top+14,priceY-28-lane*22),connectorEnd=badgeY+(buy?-11:11);return <g key={`${text(fill.id)}-${index}`} className={buy?"trade-badge buy":"trade-badge sell"} onMouseEnter={()=>setHovered({...fill,left:px/width*100,top:badgeY/height*100})} onMouseLeave={()=>setHovered(null)}><line x1={px} x2={px} y1={priceY} y2={connectorEnd} className="trade-connector"/><circle cx={px} cy={priceY} r="3" className="trade-price-dot"/><rect x={px-10} y={badgeY-10} width="20" height="20" rx="3"/><text x={px} y={badgeY+3.5} textAnchor="middle">{buy?"B":"S"}</text></g>})}{hoveredBar!=null&&<g className="chart-crosshair"><line x1={x(hoveredBar)} x2={x(hoveredBar)} y1={top} y2={volumeBottom}/><line x1={left} x2={width-right} y1={y(Number(activeBar?.close))} y2={y(Number(activeBar?.close))}/></g>}<text x={left} y={height-8}>{text(bars[0]?.trade_date)}</text><text x={(left+width-right)/2} y={height-8} textAnchor="middle">{text(bars[Math.floor(bars.length/2)]?.trade_date)}</text><text x={width-right} y={height-8} textAnchor="end">{text(bars.at(-1)?.trade_date)}</text></svg><div className="trade-legend"><span><i className="buy"/>B 买入</span><span><i className="sell"/>S 卖出</span>{selectedAccountId==="multi_sector_intraday_shadow"&&<span>含分支日前继承成交</span>}</div></div><Panel title={`${accountName(selectedAccountId)} · 历史成交`} subtitle={selectedAccountId==="multi_sector_intraday_shadow"?"包含分支建立前从普通基线继承的成交，继承记录单独标明来源":"图表和明细始终使用同一个策略账户，不混合其他策略"}><Table rows={[...strategyFills].reverse()} columns={[["trade_date","日期"],["side","操作"],["quantity","数量"],["price","成交价"],["fee","费用"],["inherited","来源"]]} format={{price:(v)=>Number(v).toFixed(3),fee:money,inherited:(v)=>v?"继承自基线":"本策略"}}/></Panel></section></div>;
}

function SystemHealthBar({health}:{health:SystemHealth}){
  const [expanded,setExpanded]=useState(false);
  return <section className={`system-health-bar ${health.system_status.toLowerCase()}`}><button onClick={()=>setExpanded(!expanded)}><i/><b>{health.system_status==="HEALTHY"?"系统健康":health.system_status==="DEGRADED"?"系统降级":"交易已冻结"}</b><span>交易门禁 {health.trading_gate} · {health.blocking_count} 阻断 · {health.warning_count} 警告</span><em>{health.observed_at.slice(11,19)} {expanded?"收起":"详情"}</em></button>{expanded&&<div>{health.checks.map((row)=><article key={text(row.code)} className={text(row.status).toLowerCase()}><b>{text(row.status)}</b><span>{text(row.code)}</span><small>{text(row.message)}</small></article>)}</div>}</section>;
}

function Panel({title,subtitle,children}:{title:string;subtitle?:string;children:ReactNode}){return <section className="panel"><header><div><h3>{title}</h3>{subtitle&&<p>{subtitle}</p>}</div><span>•••</span></header>{children}</section>}
function Kpi({label,value,note,accent}:{label:string;value:string;note:string;accent?:boolean}){return <div className={`kpi ${accent?"accent":""}`}><small>{label}</small><b>{value}</b><span>{note}</span></div>}
function Table({rows,columns,format={},strategyAccountId}:{rows:Row[];columns:[string,string][];format?:Record<string,(v:unknown)=>ReactNode>;strategyAccountId?:string}){const openSecurity=useContext(SecurityNavigationContext);const inheritedAccountId=useContext(SecurityAccountContext);if(!rows.length)return <p className="empty">暂无数据</p>;return <div className="table-wrap"><table><thead><tr>{columns.map(([,l])=><th key={l}>{l}</th>)}</tr></thead><tbody>{rows.map((row,i)=><tr key={i}>{columns.map(([k])=>{const value=row[k],content=format[k]?format[k](value):text(value);const isSecurity=(k==="symbol"||k==="instrument")&&/^\d{6}\.(SH|SZ|BJ)$/.test(text(value));const accountId=text(row.account_id)!=="—"?text(row.account_id):(strategyAccountId??inheritedAccountId);return <td key={k}>{isSecurity?<button className="security-link" onClick={()=>openSecurity(text(value),accountId)}>{content}</button>:content}</td>})}</tr>)}</tbody></table></div>}
