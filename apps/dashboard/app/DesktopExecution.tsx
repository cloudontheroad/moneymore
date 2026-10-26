"use client";

import { useCallback, useEffect, useState } from "react";

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
  const refresh=useCallback(async()=>{
    const response=await fetch("/api/desktop-execution",{cache:"no-store",signal:AbortSignal.timeout(15000)});
    if(!response.ok)throw new Error("无法读取桌面执行状态");
    setData(await response.json());
  },[]);
  useEffect(()=>{void refresh().catch(e=>setError(String(e)));},[refresh]);
  const activeCapture=data?.foreground_capture?.jobs.find(j=>["QUEUED","RUNNING","STOP_REQUESTED"].includes(j.status));
  useEffect(()=>{
    if(!activeCapture)return;
    const timer=setInterval(()=>void refresh().catch(e=>setError(String(e))),2000);
    return()=>clearInterval(timer);
  },[activeCapture,refresh]);
  const captureAction=async(stopId?:string)=>{
    setBusy(true);setError("");setMessage("");
    try{
      const response=await fetch(stopId?`/api/desktop-execution/capture/${stopId}/stop`:"/api/desktop-execution/capture",{
        method:"POST",signal:AbortSignal.timeout(15000),headers:{"Content-Type":"application/json","X-MoneyMore-Desktop":"prepare-v1"},
        body:stopId?undefined:JSON.stringify({allow_foreground_once:captureAcknowledged,read_only_only:true})});
      const result=await response.json();
      if(!response.ok)throw new Error(typeof result.detail==="string"?result.detail:"采集授权或启动失败");
      setCaptureAcknowledged(false);await refresh();
    }catch(e){setError(e instanceof Error&&e.name==="TimeoutError"?"请求超时，启动结果未知；请先刷新状态，勿重复启动采集。":e instanceof Error?e.message:"采集失败");}
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
  return <>
    <section className="panel"><header><div><h3>本地桌面代理</h3><p>由本网页发起，MoneyMore本机子进程执行，不依赖Codex</p></div></header>
      <p>当前阶段：只读探测、快照对账与计划准备。后台探测不激活窗口；下面显式授权的单次前台采集会临时切到华泰。不会输入密码或提交真实委托。</p>
      <p>后台控件不可读时直接阻断；不会回退到抢焦点点击。界面探测不代表已经支持回执解析或下单。</p>
      <p>目标客户端：{data?.configured_client?.executable??"尚未配置"}</p>
      <label>已安装交易客户端的 exe 绝对路径 <input style={{minWidth:400}} value={path} onChange={e=>setPath(e.target.value)} placeholder="D:\券商客户端\交易程序.exe"/></label>
      <button disabled={busy||!path} onClick={()=>void run("client")}>保存只读客户端（不启动）</button>
      <div className="header-actions"><button disabled={busy} onClick={()=>void run("probe")}>{busy?"处理中…":"后台只读探测"}</button>
        <button disabled={busy} onClick={()=>void refresh().catch(e=>setError(String(e)))}>刷新状态</button></div>
      {error&&<p className="error">{error}</p>}
      {(data?.limitations??[]).map(item=><p key={item}>· {item}</p>)}
    </section>
    <section className="panel"><header><div><h3>单次前台只读采集</h3><p>MoneyMore独立代理执行持仓、当日委托、当日成交导出；不依赖Codex，不下单</p></div></header>
      <p>{data?.foreground_capture?.notice??"采集状态尚未加载"}</p>
      <p>请先登录华泰并解锁桌面。授权只对本次任务有效，最长85秒；期间请勿切换窗口或操作华泰。登录、验证码和未知弹窗均会阻断。未启用周期采集。</p>
      <label><input type="checkbox" checked={captureAcknowledged} onChange={e=>setCaptureAcknowledged(e.target.checked)}/>我授权本次临时占用前台，仅采集三类只读数据</label>
      <button disabled={busy||!!activeCapture||!captureAcknowledged||!data?.configured_client} onClick={()=>void captureAction()}>开始本次只读采集</button>
      {activeCapture&&<button disabled={busy} onClick={()=>void captureAction(activeCapture.id)}>停止采集（下个检查点生效）</button>}
      {!data?.configured_client&&<p>请先在上方保存已登录华泰的客户端路径。</p>}
      {!data?.foreground_capture?.jobs.length&&<p>尚无采集任务；文件导入和前台自动采集是两条不同记录。</p>}
      {data?.foreground_capture?.jobs.map(j=><details key={j.id}><summary>{new Date(j.created_at).toLocaleString("zh-CN")} · {({QUEUED:"排队中",RUNNING:"正在只读采集",STOP_REQUESTED:"正在停止",STOPPED:"已停止",INTERRUPTED:"授权失效或服务中断",BLOCKED:"已阻断，未接管",PARTIAL_IMPORT:"部分导入，需复核",COMPLETED_UNVERIFIED:"采集完成，业务身份未核验"} as Record<string,string>)[j.status]??j.status}</summary>
        {j.result?.reason_code&&<p>原因代码：{j.result.reason_code}。不会自动重复抢回前台。</p>}
        {j.result?.imports&&<p>持仓 / 委托 / 成交导入条数：{j.result.imports.map(r=>r.row_count).join(" / ")}；未产生交易。</p>}
      </details>)}
    </section>
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
    <section className="panel"><header><div><h3>策略 → 委托 → 桌面计划</h3><p>每次只选择一个模拟策略；不会将多个独立模拟账户资金合并为真实账户</p></div></header>
      <select value={account} onChange={e=>setAccount(e.target.value)}><option value="">选择来源策略账户</option>{accounts.map(a=><option key={a} value={a}>{a}</option>)}</select>
      <button disabled={busy||!account} onClick={()=>void run("prepare")}>准备关联计划（不下单）</button>
      <p>已准备 {data?.intents.length??0} 条；实际委托编号与成交数量尚未取得，不能按按钮成功推断成交。</p>
      <div className="table-wrap"><table><thead><tr>{["策略","股票","方向","数量","信号日","计划状态","券商回执"].map(s=><th key={s}>{s}</th>)}</tr></thead>
        <tbody>{data?.intents.map(o=><tr key={o.id}><td>{o.source_account}<small> · {o.strategy_id}</small></td><td>{names[o.symbol]??o.symbol}</td><td>{o.side==="BUY"?"买入":"卖出"}</td><td>{o.quantity}</td><td>{o.signal_date}</td><td>{o.status==="PREPARED_ONLY"?"仅准备，未提交":o.status==="SOURCE_STALE"?"来源已变化，禁止沿用":o.status}</td><td>{o.receipt_status}</td></tr>)}</tbody></table></div>
    </section>
    <section className="panel"><header><div><h3>券商回执关联与差异对账</h3><p>手动明确归属后，只核对已保存证据；任何结果均不会自动补单</p></div></header>
      <p>{data?.receipt_links?.notice??"尚未加载观察关联。"}</p>
      <div className="header-actions">
        <label>桌面计划 <select value={linkIntent} onChange={e=>setLinkIntent(e.target.value)}><option value="">选择计划（不是下单）</option>
          {data?.intents.map(i=><option key={i.id} value={i.id}>{i.source_account} · {names[i.symbol]??i.symbol} · {i.side==="BUY"?"买入":"卖出"} {i.quantity} · {i.signal_date}</option>)}
        </select></label>
        <label>委托快照 <select value={linkSnapshot} onChange={e=>{setLinkSnapshot(e.target.value);setLinkOrder("");}}><option value="">选择含委托的快照</option>
          {data?.broker_exports?.snapshots.filter(s=>s.kind==="orders"&&s.row_count>0).map(s=><option key={s.id} value={s.id}>{s.account_ref} · {s.data_date} · {s.source_name}</option>)}
        </select></label>
        <label>券商委托编号 <input value={linkOrder} onChange={e=>setLinkOrder(e.target.value)} placeholder="保留编号前导零" maxLength={64}/></label>
        <label>关联说明 <input value={linkNote} onChange={e=>setLinkNote(e.target.value)} placeholder="说明为何归属于该计划" maxLength={200}/></label>
      </div>
      <label><input type="checkbox" checked={linkAcknowledged} onChange={e=>setLinkAcknowledged(e.target.checked)}/>我确认只建立观察关联，不代表已核验券商账户，不触发下单或补单</label>
      <button disabled={busy||!linkIntent||!linkSnapshot||!linkOrder||!linkNote.trim()||!linkAcknowledged} onClick={()=>void bindReceipt()}>保存观察关联（不下单）</button>
      {message&&<p role="status">{message}</p>}
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
    <section className="panel"><header><div><h3>代理探测记录</h3><p>只读控件清单不是经过核验的委托／成交回执</p></div></header>
      {!data?.jobs.length&&<p className="empty">尚未发起探测</p>}
      {data?.jobs.map(job=><details key={job.id}><summary>{job.created_at} · {job.status==="BLOCKED"?"已阻断":job.status==="INSPECTED"?"已读取控件，业务能力未验收":job.status}</summary><pre style={{whiteSpace:"pre-wrap",maxHeight:360,overflow:"auto"}}>{JSON.stringify(job.result,null,2)}</pre></details>)}
    </section>
  </>;
}
