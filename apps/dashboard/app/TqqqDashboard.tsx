"use client";

import { useCallback, useEffect, useState } from "react";
import ProjectShell from "./ProjectShell";
import { LeveragedEtfPage, type LeveragedEtfExperiment, type LeveragedEtfPaper } from "./Dashboard";

export default function TqqqDashboard() {
  const [v2,setV2]=useState<LeveragedEtfPaper|null>();
  const [v3,setV3]=useState<LeveragedEtfPaper|null>();
  const [v4,setV4]=useState<LeveragedEtfExperiment|null>();
  const load=useCallback(async()=>{
    async function refresh<T>(key:string, update:(value:T|null)=>void) {
      try {
        const response=await fetch("/api/"+key,{cache:"no-store",signal:AbortSignal.timeout(15_000)});
        if(!response.ok)throw new Error("TQQQ 数据服务暂不可用");
        update(await response.json() as T);
      } catch {
        update(null);
      }
    }
    await Promise.all([
      refresh<LeveragedEtfPaper>("leveraged-etf-paper", value=>setV2(current=>value??current??null)),
      refresh<LeveragedEtfPaper>("leveraged-etf-v3-paper", value=>setV3(current=>value??current??null)),
      refresh<LeveragedEtfExperiment>("leveraged-etf-v4-experiment", value=>setV4(current=>value??current??null)),
    ]);
  },[]);
  useEffect(()=>{const initial=setTimeout(()=>void load(),0);const timer=setInterval(()=>void load(),30_000);return()=>{clearTimeout(initial);clearInterval(timer)}},[load]);
  return <ProjectShell active="/tqqq" eyebrow="US LEVERAGED ETF · PAPER" title="TQQQ 策略" description="Cycle V2、Momentum Dip V3、风控 V4 与满仓基准。">
    {v2&&v3?<LeveragedEtfPage account={v2} experiment={v3} v4={v4}/>:<>
      {([["Cycle v2",v2],["Momentum Dip v3",v3]] as const).map(([title,account])=>
        <section className="panel" key={title}><h3>{title}</h3>
          {account?<p>信号日 {String(account.latest_signal?.signal_date??"—")} · 模拟净值 ${account.equity.toLocaleString()}</p>:
          <p>{account===null?"数据服务故障或请求超时":"正在加载…"}</p>}
        </section>)}
      <section className="panel"><h3>风控 V4</h3>{v4
        ? <p>信号日 {String(v4.latest_signal?.signal_date??"—")} · 回放净值 ${v4.equity.toLocaleString()}</p>
        : <p>{v4===null?"V4 回放暂未就绪":"正在加载 V4 回放…"}</p>}
      </section>
      <button onClick={()=>void load()}>重新连接</button>
    </>}
  </ProjectShell>;
}
