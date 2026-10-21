"use client";

import { useCallback, useEffect, useState } from "react";

type Order = { account_id:string; strategy_id:string; symbol:string; side:string;
  quantity:number; signal_date:string; idempotency_key:string };
type Intent = { id:string; source_account:string; strategy_id:string; symbol:string;
  side:string; quantity:number; signal_date:string; status:string; receipt_status:string };
type Snapshot = { mode:string; limitations:string[]; source_orders:Order[]; intents:Intent[];
  configured_client:{executable:string}|null;
  jobs:{id:string;status:string;created_at:string;result:Record<string,unknown>|null}[] };

export default function DesktopExecution({names}:{names:Record<string,string>}) {
  const [data,setData]=useState<Snapshot|null>(null);
  const [account,setAccount]=useState("");
  const [busy,setBusy]=useState(false);
  const [path,setPath]=useState("");
  const [error,setError]=useState("");
  const refresh=useCallback(async()=>{
    const response=await fetch("/api/desktop-execution",{cache:"no-store"});
    if(!response.ok)throw new Error("无法读取桌面执行状态");
    setData(await response.json());
  },[]);
  useEffect(()=>{void refresh().catch(e=>setError(String(e)));},[refresh]);
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
  return <>
    <section className="panel"><header><div><h3>本地桌面代理</h3><p>由本网页发起，MoneyMore本机子进程执行，不依赖Codex</p></div></header>
      <p>当前阶段：只读探测与计划准备。不会激活交易窗口、移动鼠标、输入密码或提交真实委托。</p>
      <p>后台控件不可读时直接阻断；不会回退到抢焦点点击。界面探测不代表已经支持回执解析或下单。</p>
      <p>目标客户端：{data?.configured_client?.executable??"尚未配置"}</p>
      <label>已安装交易客户端的 exe 绝对路径 <input style={{minWidth:400}} value={path} onChange={e=>setPath(e.target.value)} placeholder="D:\券商客户端\交易程序.exe"/></label>
      <button disabled={busy||!path} onClick={()=>void run("client")}>保存只读客户端（不启动）</button>
      <div className="header-actions"><button disabled={busy} onClick={()=>void run("probe")}>{busy?"处理中…":"后台只读探测"}</button>
        <button disabled={busy} onClick={()=>void refresh().catch(e=>setError(String(e)))}>刷新状态</button></div>
      {error&&<p className="error">{error}</p>}
      {(data?.limitations??[]).map(item=><p key={item}>· {item}</p>)}
    </section>
    <section className="panel"><header><div><h3>策略 → 委托 → 桌面计划</h3><p>每次只选择一个模拟策略；不会将多个独立模拟账户资金合并为真实账户</p></div></header>
      <select value={account} onChange={e=>setAccount(e.target.value)}><option value="">选择来源策略账户</option>{accounts.map(a=><option key={a} value={a}>{a}</option>)}</select>
      <button disabled={busy||!account} onClick={()=>void run("prepare")}>准备关联计划（不下单）</button>
      <p>已准备 {data?.intents.length??0} 条；实际委托编号与成交数量尚未取得，不能按按钮成功推断成交。</p>
      <div className="table-wrap"><table><thead><tr>{["策略","股票","方向","数量","信号日","计划状态","券商回执"].map(s=><th key={s}>{s}</th>)}</tr></thead>
        <tbody>{data?.intents.map(o=><tr key={o.id}><td>{o.source_account}<small> · {o.strategy_id}</small></td><td>{names[o.symbol]??o.symbol}</td><td>{o.side==="BUY"?"买入":"卖出"}</td><td>{o.quantity}</td><td>{o.signal_date}</td><td>{o.status==="PREPARED_ONLY"?"仅准备，未提交":o.status==="SOURCE_STALE"?"来源已变化，禁止沿用":o.status}</td><td>{o.receipt_status}</td></tr>)}</tbody></table></div>
    </section>
    <section className="panel"><header><div><h3>代理探测记录</h3><p>只读控件清单不是经过核验的委托／成交回执</p></div></header>
      {!data?.jobs.length&&<p className="empty">尚未发起探测</p>}
      {data?.jobs.map(job=><details key={job.id}><summary>{job.created_at} · {job.status==="BLOCKED"?"已阻断":job.status==="INSPECTED"?"已读取控件，业务能力未验收":job.status}</summary><pre style={{whiteSpace:"pre-wrap",maxHeight:360,overflow:"auto"}}>{JSON.stringify(job.result,null,2)}</pre></details>)}
    </section>
  </>;
}
