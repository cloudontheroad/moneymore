"use client";

import { useCallback, useEffect, useState } from "react";
import ProjectShell from "./ProjectShell";
import UsOvernight, { type UsOvernightData } from "./UsOvernight";

export default function UsOvernightDashboard(){
  const [data,setData]=useState<UsOvernightData|null>(null);const [error,setError]=useState("");
  const [session,setSession]=useState("");
  const load=useCallback(async()=>{try{const query=session?"?session="+encodeURIComponent(session):"";const response=await fetch("/api/us-overnight-dashboard"+query,{cache:"no-store",signal:AbortSignal.timeout(8_000)});if(!response.ok)throw new Error("隔夜策略数据服务暂不可用");setData(await response.json() as UsOvernightData);setError("");}catch{setError("美股隔夜服务故障或请求超时");}},[session]);
  useEffect(()=>{const initial=setTimeout(()=>void load(),0);const timer=setInterval(()=>void load(),30_000);return()=>{clearTimeout(initial);clearInterval(timer)}},[load]);
  return <ProjectShell active="/us-overnight" eyebrow="US OVERNIGHT · PAPER SIGNALS" title="美股隔夜" description="美东盘前生成纸面候选，当日开盘买入，下一交易日开盘卖出。">{data&&!error?<UsOvernight data={data} onSession={setSession}/>:<section className="project-loading"><b>{error||"正在加载美股隔夜项目…"}</b><button onClick={()=>void load()}>重新连接</button></section>}</ProjectShell>;
}
