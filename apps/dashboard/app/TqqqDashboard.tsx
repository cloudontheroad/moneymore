"use client";

import { useCallback, useEffect, useState } from "react";
import ProjectShell from "./ProjectShell";
import { LeveragedEtfPage, type LeveragedEtfPaper } from "./Dashboard";

export default function TqqqDashboard() {
  const [accounts,setAccounts]=useState<Record<string,LeveragedEtfPaper|null>>({});
  const load=useCallback(async()=>{
    await Promise.all(["leveraged-etf-paper","leveraged-etf-v3-paper"].map(async key=>{
      try {
        const response=await fetch("/api/"+key,{cache:"no-store",signal:AbortSignal.timeout(15_000)});
        if(!response.ok)throw new Error("TQQQ 数据服务暂不可用");
        const account=await response.json() as LeveragedEtfPaper;
        setAccounts(current=>({...current,[key]:account}));
      } catch {
        setAccounts(current=>({...current,[key]:null}));
      }
    }));
  },[]);
  useEffect(()=>{const initial=setTimeout(()=>void load(),0);const timer=setInterval(()=>void load(),30_000);return()=>{clearTimeout(initial);clearInterval(timer)}},[load]);
  const v2=accounts["leveraged-etf-paper"],v3=accounts["leveraged-etf-v3-paper"];
  return <ProjectShell active="/tqqq" eyebrow="US LEVERAGED ETF · PAPER" title="TQQQ 策略" description="Cycle v2、Momentum Dip v3 和满仓基准。">
    {v2&&v3?<LeveragedEtfPage account={v2} experiment={v3}/>:<>
      {([["Cycle v2",v2],["Momentum Dip v3",v3]] as const).map(([title,account])=>
        <section className="panel" key={title}><h3>{title}</h3>
          {account?<p>信号日 {String(account.latest_signal?.signal_date??"—")} · 模拟净值 ${account.equity.toLocaleString()}</p>:
          <p>{account===null?"数据服务故障或请求超时":"正在加载…"}</p>}
        </section>)}
      <button onClick={()=>void load()}>重新连接</button>
    </>}
  </ProjectShell>;
}
