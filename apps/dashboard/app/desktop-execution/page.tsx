import DesktopExecution from "../DesktopExecution";

export default function DesktopExecutionPage() {
  return <main style={{maxWidth:1400,margin:"0 auto",padding:24}}>
    <header><h1>MoneyMore · 桌面执行</h1><p>独立入口，不等待策略研究与收益报表加载。仅只读采集与计划准备，不下单。</p><a href="/">返回组合总览</a></header>
    <DesktopExecution names={{}} />
  </main>;
}
