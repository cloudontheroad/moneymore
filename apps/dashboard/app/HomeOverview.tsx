"use client";

import { useEffect, useState, type CSSProperties } from "react";
import Link from "next/link";
import ProjectShell from "./ProjectShell";

type ProjectState={status:string;metric:string;note:string;ready:boolean};
const initial:ProjectState={status:"连接中",metric:"—",note:"正在读取最新快照",ready:false};
const percent=(value:unknown)=>`${(Number(value??0)*100).toFixed(1)}%`;

export default function HomeOverview(){
  const [aShare,setAShare]=useState<ProjectState>(initial); const [tqqq,setTqqq]=useState<ProjectState>(initial); const [overnight,setOvernight]=useState<ProjectState>(initial);
  useEffect(()=>{
    const controller=new AbortController();
    const read=async(url:string)=>{const response=await fetch(url,{cache:"no-store",signal:controller.signal});if(!response.ok)throw new Error(String(response.status));return response.json() as Promise<Record<string,unknown>>};
    void read("/api/multi-sector-execution").then(data=>{const portfolio=data.portfolio as Record<string,unknown>|null;const positions=Array.isArray(data.positions)?data.positions:[];setAShare({status:String(data.status||"运行中"),metric:portfolio?`¥${Number(portfolio.equity??0).toLocaleString("zh-CN",{maximumFractionDigits:0})}`:"等待净值",note:`${positions.length} 只持仓 · ${String(data.trade_date??"等待交易日")}`,ready:true})}).catch(()=>!controller.signal.aborted&&setAShare({status:"暂不可用",metric:"—",note:"进入项目页可单独重试",ready:false}));
    void read("/api/leveraged-etf-paper").then(data=>setTqqq({status:String(data.status||"模拟运行"),metric:percent(data.total_return),note:`权益 $${Number(data.equity??0).toLocaleString("en-US",{maximumFractionDigits:0})} · Cycle v2`,ready:true})).catch(()=>!controller.signal.aborted&&setTqqq({status:"暂不可用",metric:"—",note:"不影响其他项目",ready:false}));
    void read("/api/us-overnight-dashboard").then(data=>{const candidates=Array.isArray(data.candidates)?data.candidates:[];const policy=(data.policy??{}) as Record<string,unknown>;setOvernight({status:data.status==="CANDIDATES_READY"?"候选已生成":"研究中",metric:`${candidates.length} / ${policy.top_n??10}`,note:data.latest_session?`最新 ${String(data.latest_session)}`:"等待首次离线运行",ready:true})}).catch(()=>!controller.signal.aborted&&setOvernight({status:"暂不可用",metric:"—",note:"不影响其他项目",ready:false}));
    return()=>controller.abort();
  },[]);
  const cards=[{href:"/a-share",code:"CN",title:"A股组合",description:"全局选股、仓位风控、模拟执行与运行对账",state:aShare,color:"#a8e063"},{href:"/tqqq",code:"3X",title:"TQQQ",description:"Cycle v2、Momentum Dip v3 与满仓基准",state:tqqq,color:"#7e91e8"},{href:"/us-overnight",code:"ON",title:"美股隔夜",description:"收盘候选、隔夜持有与次日退出研究",state:overnight,color:"#53bfb0"}];
  return <ProjectShell active="/" eyebrow="ONE HOME · THREE INDEPENDENT PROJECTS" title="投资系统总览" description="首页只读取三个项目的最小快照。每个项目进入后才加载自己的完整数据，一个服务变慢不会再卡住整个网站。"><section className="project-cards">{cards.map(card=><Link href={card.href} className="project-card" key={card.href} style={{"--project-color":card.color} as CSSProperties}><header><span>{card.code}</span><i className={card.state.ready?"ready":""}/></header><div><small>{card.state.status}</small><h2>{card.title}</h2><p>{card.description}</p></div><footer><strong>{card.state.metric}</strong><span>{card.state.note}</span><b>进入项目 →</b></footer></Link>)}</section><section className="home-principles"><span>分页面加载</span><span>项目故障隔离</span><span>首页最小数据</span><span>30 秒项目内刷新</span></section></ProjectShell>;
}
