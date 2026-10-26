"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {captureHelp,dateTime,isSnapshotStale,jobLabels,kindLabels,latestSnapshot} from "./desktop-view";
import "./broker.css";

type Order = { account_id:string; strategy_id:string; symbol:string; side:string;
  quantity:number; signal_date:string; idempotency_key:string };
type Intent = { id:string; source_account:string; strategy_id:string; symbol:string;
  side:string; quantity:number; signal_date:string; status:string; receipt_status:string };
type ReceiptReport = {status:string;issues:string[];observed_filled_quantity:number|null;
  observed_cumulative_quantity?:number|null;latest_source_mtime?:string|null};
type ReceiptLink = {id:string;intent_id:string;account_ref:string;trade_date:string;market:string;
  broker_order_id:string;code:string;side:string;quantity:number;note:string;reconciliation:ReceiptReport};
type CaptureJob = {id:string;status:string;created_at:string;expires_at:string;
  result:{reason_code?:string;imports?:{row_count:number}[]}|null};
type ExportKind = "positions"|"orders"|"trades";
type BrokerExport = { id:string; kind:ExportKind; account_ref:string; data_date:string;
  source_name:string; source_mtime:string; imported_at:string; row_count:number;
  freshness:string; rows_truncated:boolean; rows:Record<string,string|number|null>[] };
type Snapshot = { mode:string; limitations:string[]; source_orders:Order[]; intents:Intent[];
  broker_exports?:{candidates:{filename:string;source_mtime:string;bytes:number}[];
    snapshots:BrokerExport[];notice:string};
  receipt_links?:{links:ReceiptLink[];notice:string};
  foreground_capture?:{jobs:CaptureJob[];notice:string;scheduled:boolean};
  configured_client:{executable:string}|null;
  jobs:{id:string;status:string;created_at:string;result:Record<string,unknown>|null}[] };

export default function DesktopExecution({names}:{names:Record<string,string>}) {
  const [data,setData]=useState<Snapshot|null>(null);
  const [account,setAccount]=useState("");
  const [busy,setBusy]=useState(false);
  const [path,setPath]=useState("");
  const [error,setError]=useState("");
  const [exportFile,setExportFile]=useState("");
  const [exportKind,setExportKind]=useState<ExportKind>("positions");
  const [exportDate,setExportDate]=useState(()=>new Date().toLocaleDateString("sv-SE",{timeZone:"Asia/Shanghai"}).replaceAll("-",""));
  const [accountRef,setAccountRef]=useState("huatai_local");
  const [message,setMessage]=useState("");
  const [linkIntent,setLinkIntent]=useState("");
  const [linkSnapshot,setLinkSnapshot]=useState("");
  const [linkOrder,setLinkOrder]=useState("");
  const [linkNote,setLinkNote]=useState("");
  const [linkAcknowledged,setLinkAcknowledged]=useState(false);
  const [captureAcknowledged,setCaptureAcknowledged]=useState(false);
  const [view,setView]=useState<ExportKind|"history">("positions");
  const [showConsent,setShowConsent]=useState(false);
  const [uncertain,setUncertain]=useState(false);
  const [loading,setLoading]=useState(true);
  const [now,setNow]=useState(Date.now());
  const inFlight=useRef(false);
  const refresh=useCallback(async()=>{
    if(inFlight.current)return false;
    inFlight.current=true;
    try{
    const response=await fetch("/api/desktop-execution",{cache:"no-store",signal:AbortSignal.timeout(15000)});
    if(!response.ok)throw new Error("无法读取桌面执行状态");
    setData(await response.json());
    setNow(Date.now());
    return true;
    }finally{inFlight.current=false;setLoading(false);}
  },[]);
  useEffect(()=>{void refresh().catch(e=>setError(String(e)));},[refresh]);
  const activeCapture=data?.foreground_capture?.jobs.find(j=>["QUEUED","RUNNING","STOP_REQUESTED"].includes(j.status));
  const activeCaptureId=activeCapture?.id;
  useEffect(()=>{
    const timer=setInterval(()=>{setNow(Date.now());void refresh().catch(()=>setError("状态刷新失败，保留上次快照，请勿将其视为实时。"));},activeCaptureId?2000:30000);
    return()=>clearInterval(timer);
  },[activeCaptureId,refresh]);
  const captureAction=async(stopId?:string)=>{
    setBusy(true);setError("");setMessage("");
    try{
      const response=await fetch(stopId?`/api/desktop-execution/capture/${stopId}/stop`:"/api/desktop-execution/capture",{
        method:"POST",signal:AbortSignal.timeout(15000),headers:{"Content-Type":"application/json","X-MoneyMore-Desktop":"prepare-v1"},
        body:stopId?undefined:JSON.stringify({allow_foreground_once:captureAcknowledged,read_only_only:true})});
      const result=await response.json();
      if(!response.ok)throw new Error(typeof result.detail==="string"?result.detail:"采集授权或启动失败");
      setCaptureAcknowledged(false);setShowConsent(false);await refresh();
    }catch(e){if(!stopId)setUncertain(true);setError(e instanceof Error&&e.name==="TimeoutError"?"请求超时，启动结果未知；请先刷新状态，勿重复启动采集。":e instanceof Error?e.message:"采集失败");}
    finally{setBusy(false);}
  };
  const run=async(action:"probe"|"prepare"|"client")=>{
    setBusy(true);setError("");
    try{
      const response=await fetch(`/api/desktop-execution/${action}`,{
        method:"POST",headers:{"Content-Type":"application/json","X-MoneyMore-Desktop":"prepare-v1"},
        body:action==="prepare"?JSON.stringify({source_account:account}):action==="client"?JSON.stringify({executable:path}):undefined});
      const result=await response.json();
      if(!response.ok)throw new Error(result.detail??"操作失败");
      await refresh();
    }catch(e){setError(e instanceof Error?e.message:"操作失败");}
    finally{setBusy(false);}
  };
  const accounts=[...new Set(data?.source_orders.map(o=>o.account_id)??[])];
  const importExport=async()=>{
    setBusy(true);setError("");setMessage("");
    try{
      const response=await fetch("/api/desktop-execution/exports/import",{
        method:"POST",headers:{"Content-Type":"application/json","X-MoneyMore-Desktop":"prepare-v1"},
        body:JSON.stringify({filename:exportFile,kind:exportKind,trade_date:exportDate,account_ref:accountRef})});
      const result=await response.json();
      if(!response.ok)throw new Error(typeof result.detail==="string"?result.detail:"导出文件校验未通过");
      setMessage(`已读取 ${result.row_count} 条；仅保存券商观察快照，未修改模拟或真实交易账本。`);
      await refresh();
    }catch(e){setError(e instanceof Error?e.message:"导入失败");}
    finally{setBusy(false);}
  };
  const exportLabels:Record<ExportKind,string>={positions:"持仓",orders:"委托",trades:"成交"};
  const receiptLabels:Record<string,string>={WAITING_ORDER:"等待委托观察",WAITING_TRADES:"等待成交明细，数量未知",
    OBSERVED_FULL:"观察到全额成交，未核验",OBSERVED_PARTIAL:"观察到部分成交，未核验",
    OBSERVED_ORDER_ONLY:"仅观察到委托，未见成交",REVIEW_REQUIRED:"差异待复核"};
  const issueLabels:Record<string,string>={MARKET_UNKNOWN:"市场信息未知",SECURITY_OR_SIDE_CHANGED:"证券或方向不一致",
    DATE_UNVERIFIED:"行内日期未核验",AMBIGUOUS_ORDER:"委托不唯一",ORDER_TERMS_CHANGED:"数量或价格变化",
    SOURCE_TIME_COLLISION:"相同文件时点数据冲突",CUMULATIVE_FILL_REGRESSION:"累计成交数量倒退",
    TRADE_ID_CONFLICT:"成交编号内容冲突",OVERFILL:"成交数量超过委托",ORDER_MISSING_IN_LATEST:"最新快照缺少该委托",
    TRADE_MISSING_IN_LATEST:"最新快照缺少已观察成交",SUMMARY_DETAIL_MISMATCH:"委托汇总与成交明细不一致",
    SNAPSHOT_SCOPE_MISMATCH:"账户别名或日期范围不符",HISTORY_LIMIT_EXCEEDED:"历史观察超出核对上限"};
  const bindReceipt=async()=>{
    setBusy(true);setError("");setMessage("");
    try{
      const response=await fetch("/api/desktop-execution/receipts/link",{
        method:"POST",headers:{"Content-Type":"application/json","X-MoneyMore-Desktop":"prepare-v1"},
        body:JSON.stringify({intent_id:linkIntent,snapshot_id:linkSnapshot,broker_order_id:linkOrder,
          note:linkNote,acknowledge_observation_only:linkAcknowledged})});
      const result=await response.json();
      if(!response.ok)throw new Error(typeof result.detail==="string"?result.detail:"关联校验未通过");
      setMessage("已保存观察关联，不下单、不确认实盘成交、不修改模拟账本。");
      setLinkAcknowledged(false);await refresh();
    }catch(e){setError(e instanceof Error?e.message:"关联失败");}
    finally{setBusy(false);}
  };
  const snapshots=data?.broker_exports?.snapshots??[];
  const jobs=data?.foreground_capture?.jobs??[];
  const latestJob=jobs[0];
  const success=jobs.find(j=>j.status==="COMPLETED_UNVERIFIED");
  const current=view==="history"?undefined:latestSnapshot(snapshots,view);
  const stale=isSnapshotStale(current,now);
  const locked=busy||!!activeCapture;
  const refreshStatus=async()=>{try{if(await refresh()){setUncertain(false);setError("");}}catch(e){setError(String(e));}};
  return <div className="broker-workbench">
    <section className="broker-hero">
      <div className="broker-hero-top"><span>MONEYMORE / BROKER WORKSPACE</span><span className="broker-mode">只读模式 · 不下单</span></div>
      <div className="broker-hero-body"><div><h2>华泰券商工作台</h2><p>同步券商账户数据，核对持仓与交易记录。</p><div className="broker-connection"><i className={data?.configured_client?"configured":""}/><span>{loading?"正在读取配置":data?.configured_client?"客户端已配置":"客户端未配置"}</span><span>登录状态需在客户端确认</span></div></div>
        <div className="broker-hero-actions"><button className="broker-primary" disabled={loading||locked||uncertain||!data?.configured_client} onClick={()=>{setShowConsent(true);setCaptureAcknowledged(false);}}>↻ 同步券商数据</button><button className="broker-secondary" disabled={busy} onClick={()=>void refreshStatus()}>刷新状态</button><small>最近成功同步 {dateTime(success?.created_at)}</small></div></div>
    </section>
    {error&&<div className="broker-feedback error" role="alert">{error}</div>}
    {message&&<div className="broker-feedback" role="status">{message}</div>}
    {showConsent&&<section className="broker-consent" role="dialog" aria-modal="false" aria-label="确认本次只读同步"><h3>本次同步会临时使用前台</h3><p>请先登录华泰并解锁桌面。约30秒内不要操作键鼠，最长85秒。只导出持仓、当日委托和成交；不读取密码、不下单、不撤单。</p><label><input type="checkbox" checked={captureAcknowledged} onChange={e=>setCaptureAcknowledged(e.target.checked)}/>我授权本次临时使用前台，仅采集只读数据</label><div><button className="broker-primary" disabled={!captureAcknowledged||locked||uncertain} onClick={()=>void captureAction()}>开始本次同步</button><button disabled={busy} onClick={()=>setShowConsent(false)}>取消</button></div></section>}
    {activeCapture&&<section className="broker-progress" aria-live="polite"><span className="broker-spinner"/><div><strong>{activeCapture.status==="STOP_REQUESTED"?"正在安全停止":"正在同步券商数据"}</strong><p>持仓 → 当日委托 → 当日成交 → 校验入库。期间勿切换窗口，完成后自动最小化华泰。</p></div><button disabled={busy||activeCapture.status==="STOP_REQUESTED"} onClick={()=>void captureAction(activeCapture.id)}>停止同步</button></section>}
    {!activeCapture&&latestJob&&["BLOCKED","PARTIAL_IMPORT","INTERRUPTED"].includes(latestJob.status)&&<section className="broker-attention"><strong>{jobLabels[latestJob.status]}</strong><p>{captureHelp(latestJob.result?.reason_code)}</p><small>失败不覆盖旧快照，不会自动重复操作客户端。</small></section>}
    <div className="broker-metrics">{(["positions","orders","trades"] as ExportKind[]).map(kind=>{const item=latestSnapshot(snapshots,kind);const old=isSnapshotStale(item,now);return <button key={kind} className={view===kind?"selected":""} onClick={()=>setView(kind)}><span>最近快照 · {kindLabels[kind]}</span><strong>{loading?"…":item?item.row_count:"—"}<small>{item?"条":"未同步"}</small></strong><footer><span className={`broker-pill ${old?"muted":"recent"}`}>{item?(old?"历史快照":"近期文件"):"待采集"}</span><small>{dateTime(item?.source_mtime)}</small></footer></button>;})}</div>
    <section className="broker-data"><div className="broker-data-head"><div><h3>券商数据</h3><p>独立于模拟账户保存，不按同一只股票自动认领策略。</p></div><span className="broker-pill warn">账户身份待核验</span></div>
      <div className="broker-tabs" role="tablist" aria-label="券商数据类型">{(["positions","orders","trades","history"] as const).map(kind=><button key={kind} role="tab" aria-selected={view===kind} aria-controls="broker-data-panel" onClick={()=>setView(kind)}>{kind==="history"?"同步记录":kindLabels[kind]}</button>)}</div>
      <div id="broker-data-panel" role="tabpanel" aria-label={view==="history"?"同步记录":kindLabels[view]}>
        {view==="history"?<div className="broker-history">{!jobs.length?<div className="broker-empty"><strong>尚无同步记录</strong><p>从右上方开始第一次只读同步。</p></div>:jobs.map(job=><details key={job.id}><summary><span className={`broker-history-dot ${job.status==="COMPLETED_UNVERIFIED"?"done":""}`}/><strong>{jobLabels[job.status]??"状态待确认"}</strong><time>{dateTime(job.created_at)}</time></summary><div><p>{job.result?.imports?`持仓 / 委托 / 成交：${job.result.imports.map(i=>i.row_count).join(" / ")} 条；未产生交易。`:captureHelp(job.result?.reason_code)}</p><small>作业 {job.id}{job.result?.reason_code?` · ${job.result.reason_code}`:""}</small></div></details>)}</div>:loading?<div className="broker-empty">正在读取券商快照…</div>:!current?<div className="broker-empty"><span>⇄</span><strong>还没有{kindLabels[view]}快照</strong><p>点击“同步券商数据”读取。未采集不代表空仓或没有交易。</p></div>:<>
          <div className="broker-snapshot-meta"><span>数据日 {current.data_date} · 文件采集于 {dateTime(current.source_mtime)}</span><span className={`broker-pill ${stale?"warn":"muted"}`}>{stale?"快照已过期 · 非实时":"文件近期采集 · 业务状态待核验"}</span></div>
          {!current.row_count?<div className="broker-empty"><span>—</span><strong>该次导出没有{kindLabels[view]}记录</strong><p>仅表示这份快照为空，不据此认定当日从未交易或清算已完成。</p></div>:<div className="broker-table-scroll"><table><thead><tr><th>证券</th>{view==="positions"?<><th>持仓数量</th><th>可用数量</th></>:<><th>方向</th><th>数量</th>{view==="orders"&&<th>已成交</th>}<th>价格</th><th>委托编号</th>{view==="trades"&&<th>成交编号</th>}<th>时间</th></>}</tr></thead><tbody>{current.rows.map((row,i)=><tr key={`${current.id}-${i}`}><td><strong>{row.name??names[`${row.code}.${row.market}`]??"名称未知"}</strong><small>{row.code}{row.market?`.${row.market}`:""}</small></td>{view==="positions"?<><td>{row.quantity??"未知"}</td><td>{row.available_quantity??"未知"}</td></>:<><td><span className={`broker-side ${row.side==="BUY"?"buy":"sell"}`}>{row.side==="BUY"?"买入":row.side==="SELL"?"卖出":"未知"}</span></td><td>{row.quantity??"未知"}</td>{view==="orders"&&<td>{row.filled_quantity??"未知"}</td>}<td>{row.price??"—"}</td><td>{row.order_id??"—"}</td>{view==="trades"&&<td>{row.trade_id??"—"}</td>}<td>{row.time??"—"}</td></>}</tr>)}</tbody></table></div>}
          {current.rows_truncated&&<p className="broker-footnote">仅展示前100条，完整快照保存在本机。</p>}
        </>}
      </div>
    </section>
    <section className="broker-boundary"><div><span className="broker-pill muted">交易尚未启用</span><h3>同步成功 ≠ 券商已成交</h3><p>当前已验证数据导出与入库。账户身份、非空买卖回执及策略唯一关联仍待验收，不会自动提交真实委托。</p></div><div className="broker-checkpoints"><span className={success?"done":""}>① 只读同步{success?"已跑通":"待验收"}</span><span>② 账户与回执待核验</span><span>③ 真实交易未启用</span></div></section>
    <details className="broker-advanced"><summary><div><strong>高级工具与审计</strong><small>客户端设置、文件导入、实验计划和诊断，日常同步无需操作</small></div><span>展开工具</span></summary><fieldset disabled={locked}>
    <details className="broker-tool"><summary>客户端设置与只读诊断</summary>
    <section className="panel"><header><div><h3>本地桌面代理</h3><p>由本网页发起，MoneyMore本机子进程执行，不依赖Codex</p></div></header>
      <p>只在客户端变更或排障时操作。保存路径不会启动或登录客户端，后台诊断不会采集新快照。</p>
      <p>后台控件不可读时直接阻断；不会回退到抢焦点点击。界面探测不代表已经支持回执解析或下单。</p>
      <p>目标客户端：{data?.configured_client?.executable??"尚未配置"}</p>
      <label>已安装交易客户端的 exe 绝对路径 <input style={{minWidth:400}} value={path} onChange={e=>setPath(e.target.value)} placeholder="D:\券商客户端\交易程序.exe"/></label>
      <button disabled={busy||!path} onClick={()=>void run("client")}>保存只读客户端（不启动）</button>
      <div className="header-actions"><button disabled={busy} onClick={()=>void run("probe")}>{busy?"处理中…":"后台只读探测"}</button>
        <button disabled={busy} onClick={()=>void refresh().catch(e=>setError(String(e)))}>刷新状态</button></div>

    </section>
    </details>
    <details className="broker-tool"><summary>历史快照与手动文件导入</summary>
    <section className="panel"><header><div><h3>华泰券商只读快照</h3><p>读取已保存文件，不操作交易窗口；与各策略的模拟账本隔离</p></div></header>
      <p>{data?.broker_exports?.notice??"本地导出解析尚未加载。"}</p>
      <p>账户别名与数据日期由导入时指定，尚未独立核验；不按股票自动分配策略，也不凭空表断言没有交易。</p>
      <div className="header-actions">
        <label>导出文件 <select value={exportFile} onChange={e=>{
          const file=e.target.value;setExportFile(file);
          const kind=(["positions","orders","trades"] as ExportKind[]).find(k=>file.startsWith(k));
          if(kind)setExportKind(kind);
        }}><option value="">选择私密目录中已保存的文件</option>
          {data?.broker_exports?.candidates.map(f=><option key={f.filename} value={f.filename}>{f.filename}</option>)}
        </select></label>
        <label>类型 <select value={exportKind} onChange={e=>setExportKind(e.target.value as ExportKind)}>
          {Object.entries(exportLabels).map(([key,label])=><option key={key} value={key}>{label}</option>)}
        </select></label>
        <label>数据日期 <input value={exportDate} onChange={e=>setExportDate(e.target.value)} placeholder="YYYYMMDD" maxLength={8}/></label>
        <label>本地账户别名 <input value={accountRef} onChange={e=>setAccountRef(e.target.value)} maxLength={64}/></label>
        <button disabled={busy||!exportFile||!/^\d{8}$/.test(exportDate)||!accountRef} onClick={()=>void importExport()}>读取已保存导出（不刷新券商）</button>
      </div>
      {!data?.broker_exports?.snapshots.length&&<p className="empty">尚未导入券商快照，不代表空仓。</p>}
      {data?.broker_exports?.snapshots.map((item,index)=><details key={item.id} open={index===0}>
        <summary>{exportLabels[item.kind]} · {item.account_ref} · 数据日 {item.data_date} · {item.row_count} 条 · {item.freshness==="STALE"?"历史/过期快照，非实时":"近期文件，业务状态未核验"}</summary>
        <p>来源：{item.source_name}；文件时间：{new Date(item.source_mtime).toLocaleString("zh-CN")}；首次导入：{new Date(item.imported_at).toLocaleString("zh-CN")}</p>
        {!item.row_count?<p>此导出无数据行。不据此认定当日从未交易或清算已经完成。</p>:<div className="table-wrap"><table>
          <thead><tr>{["股票","方向","数量","可用 / 已成交","价格","委托编号","成交编号","时间"].map(s=><th key={s}>{s}</th>)}</tr></thead>
          <tbody>{item.rows.map((row,i)=><tr key={`${item.id}-${i}`}>
            <td>{row.name} <small>{row.code}</small></td>
            <td>{row.side==="BUY"?"买入":row.side==="SELL"?"卖出":"—"}</td><td>{row.quantity??"未知"}</td>
            <td>{item.kind==="positions"?row.available_quantity??"未知":item.kind==="orders"?row.filled_quantity??"未知":"—"}</td>
            <td>{row.price??"—"}</td><td>{row.order_id??"—"}</td><td>{row.trade_id??"—"}</td><td>{row.time??"—"}</td>
          </tr>)}</tbody></table></div>}
        {item.rows_truncated&&<p>仅展示前100条，完整快照保存在本机独立账本中。</p>}
      </details>)}
    </section>
    </details>
    <details className="broker-tool"><summary>实验性策略计划（仅准备，不下单）</summary>
    <section className="panel"><header><div><h3>策略 → 委托 → 桌面计划</h3><p>每次只选择一个模拟策略；不会将多个独立模拟账户资金合并为真实账户</p></div></header>
      <select value={account} onChange={e=>setAccount(e.target.value)}><option value="">选择来源策略账户</option>{accounts.map(a=><option key={a} value={a}>{a}</option>)}</select>
      <button disabled={busy||!account} onClick={()=>void run("prepare")}>准备关联计划（不下单）</button>
      <p>有效准备 {data?.intents.filter(i=>i.status!=="SOURCE_STALE").length??0} 条；已过期 {data?.intents.filter(i=>i.status==="SOURCE_STALE").length??0} 条。准备计划不是券商委托。</p>
      {!accounts.length&&<p>当前没有可准备的待处理模拟委托，无需操作。</p>}
      <div className="table-wrap"><table><thead><tr>{["策略","股票","方向","数量","信号日","计划状态","券商回执"].map(s=><th key={s}>{s}</th>)}</tr></thead>
        <tbody>{data?.intents.filter(i=>i.status!=="SOURCE_STALE").map(o=><tr key={o.id}><td>{o.source_account}<small> · {o.strategy_id}</small></td><td>{names[o.symbol]??o.symbol}</td><td>{o.side==="BUY"?"买入":"卖出"}</td><td>{o.quantity}</td><td>{o.signal_date}</td><td>仅准备，未提交</td><td>{o.receipt_status}</td></tr>)}</tbody></table></div>
      {!!data?.intents.some(i=>i.status==="SOURCE_STALE")&&<details><summary>历史过期计划（禁止沿用）</summary>{data.intents.filter(i=>i.status==="SOURCE_STALE").map(i=><p key={i.id}>{names[i.symbol]??i.symbol} · {i.symbol} · {i.signal_date} · 来源已变化</p>)}</details>}
    </section>
    </details>
    <details className="broker-tool"><summary>手工回执关联与差异复核</summary>
    <section className="panel"><header><div><h3>券商回执关联与差异对账</h3><p>手动明确归属后，只核对已保存证据；任何结果均不会自动补单</p></div></header>
      <p>{data?.receipt_links?.notice??"尚未加载观察关联。"}</p>
      <div className="header-actions">
        <label>桌面计划 <select value={linkIntent} onChange={e=>setLinkIntent(e.target.value)}><option value="">选择计划（不是下单）</option>
          {data?.intents.filter(i=>i.status!=="SOURCE_STALE").map(i=><option key={i.id} value={i.id}>{i.source_account} · {names[i.symbol]??i.symbol} · {i.side==="BUY"?"买入":"卖出"} {i.quantity} · {i.signal_date}</option>)}
        </select></label>
        <label>委托快照 <select value={linkSnapshot} onChange={e=>{setLinkSnapshot(e.target.value);setLinkOrder("");}}><option value="">选择含委托的快照</option>
          {data?.broker_exports?.snapshots.filter(s=>s.kind==="orders"&&s.row_count>0).map(s=><option key={s.id} value={s.id}>{s.account_ref} · {s.data_date} · {s.source_name}</option>)}
        </select></label>
        <label>券商委托编号 <input value={linkOrder} onChange={e=>setLinkOrder(e.target.value)} placeholder="保留编号前导零" maxLength={64}/></label>
        <label>关联说明 <input value={linkNote} onChange={e=>setLinkNote(e.target.value)} placeholder="说明为何归属于该计划" maxLength={200}/></label>
      </div>
      <label><input type="checkbox" checked={linkAcknowledged} onChange={e=>setLinkAcknowledged(e.target.checked)}/>我确认只建立观察关联，不代表已核验券商账户，不触发下单或补单</label>
      <button disabled={busy||!linkIntent||!linkSnapshot||!linkOrder||!linkNote.trim()||!linkAcknowledged} onClick={()=>void bindReceipt()}>保存观察关联（不下单）</button>

      {!data?.receipt_links?.links.length&&<p className="empty">尚无明确关联；不会因为股票相同而自动认领交易。空委托快照不能建立关联。</p>}
      <div className="table-wrap"><table><thead><tr>{["股票","观察归属","委托编号","委托数量","观察成交 / 汇总","核对状态","差异"].map(h=><th key={h}>{h}</th>)}</tr></thead>
        <tbody>{data?.receipt_links?.links.map(link=><tr key={link.id}>
          <td>{names[`${link.code}.${link.market}`]??link.code}<small> · {link.code}.{link.market}</small></td>
          <td>{data.intents.find(i=>i.id===link.intent_id)?.source_account??"历史计划"}<small> · {link.account_ref} · {link.trade_date}</small></td>
          <td>{link.broker_order_id}</td><td>{link.quantity}</td>
          <td>{link.reconciliation.observed_filled_quantity??"未知"} / {link.reconciliation.observed_cumulative_quantity??"未知"}</td>
          <td>{receiptLabels[link.reconciliation.status]??link.reconciliation.status}</td>
          <td>{link.reconciliation.issues.map(i=>issueLabels[i]??i).join("；")||"未发现数值矛盾，账户身份仍未核验"}</td>
        </tr>)}</tbody></table></div>
      <p>未知结果不视为失败或零成交；出现差异先复核回执，不自动重试报单。旧计划即使建立观察关联，也不会解除来源过期保护。</p>
    </section>
    </details>
    <details className="broker-tool"><summary>原始诊断档案</summary>
    <section className="panel"><header><div><h3>代理探测记录</h3><p>只读控件清单不是经过核验的委托／成交回执</p></div></header>
      {!data?.jobs.length&&<p className="empty">尚未发起探测</p>}
      {data?.jobs.map(job=><details key={job.id}><summary>{job.created_at} · {job.status==="BLOCKED"?"已阻断":job.status==="INSPECTED"?"已读取控件，业务能力未验收":job.status}</summary><pre style={{whiteSpace:"pre-wrap",maxHeight:360,overflow:"auto"}}>{JSON.stringify(job.result,null,2)}</pre></details>)}
    </section>
    </details>
    </fieldset></details>
  </div>;
}
