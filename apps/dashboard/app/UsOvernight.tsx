type Candidate = {rank:number;symbol:string;name?:string;score:number;review_score?:number;entry_open?:number;exit_open?:number;shares?:number;allocation_usd?:number;net_return?:number;pnl_usd?:number};
type Ledger = {trade_session:string;exit_session:string;status:string;positions:Candidate[];daily_return?:number|null;pnl_usd?:number|null;cost_usd?:number|null;fee_model?:string};

export type UsOvernightData = {
  mode:string; status:string; latest_session?:string|null; generated_at?:string|null;
  expected_session?:string; reason?:string|null; market_forecast_bps?:number|null;
  schedule:{timezone:string;signal_cutoff:string;entry_window:string[];exit_window:string[]};
  policy:{top_n:number;round_trip_cost_bps:number};
  candidates:Candidate[];
  universe:{size:number};
  opend:{status:string;error?:string|null};
  paper_book?:{
    start_session?:string;capital_usd?:number;allocation_usd?:number;
    available_sessions:string[];selected_session:string;
    record?:Ledger|null;
    comparison?:{status:string;added?:string[];removed?:string[];quote_count?:number;record?:Ledger|null;pnl_delta_usd?:number|null;return_delta?:number|null}|null;
  };
};

const statusText:Record<string,string> = {
  READY:"候选已生成", CASH:"策略空仓", DATA_PENDING:"等待日线或股票池",
  EXPIRED:"盘前截止已过", STALE:"候选已过期", WAITING_FOR_FIRST_RUN:"等待盘前信号",
};

const dollars = (n?:number|null) => n == null ? "—" : n.toLocaleString("en-US",{style:"currency",currency:"USD"});
const percent = (n?:number|null) => n == null ? "—" : `${(100*n).toFixed(2)}%`;
const ledgerStatus:Record<string,string> = {READY:"待开盘价",OPEN:"已记录买入，待下一交易日开盘",SETTLED:"已结算"};
const reviewStatus:Record<string,string> = {
  PENDING:"等待盘前复核（美东 09:20）", UNCHANGED:"盘前名单无换股，当天未建立盘前更新仓",
  INSUFFICIENT_QUOTES:"盘前报价覆盖不足", LATE:"盘前复核超过截止时间",
  MISSED:"未取得有效盘前复核",
};

function PositionTable({title,record,rows,allocation,topN,empty,review=false,reason}:{
  title:string;record?:Ledger|null;rows:Candidate[];allocation?:number;topN:number;empty:string;review?:boolean;reason?:string|null;
}) {
  const columns = ["排名","代码","名称",review ? "盘前复核分数" : "基准分数","分配资金","模拟股数","买入开盘价","卖出开盘价","净损益","净收益率"];
  return <section aria-label={title}>
    <h4>{title}</h4>
    {record && <>
      <p>{ledgerStatus[record.status] ?? statusText[record.status] ?? record.status} · {record.trade_session} 买入 · {record.exit_session} 卖出 · {rows.length} / {topN} 只</p>
      <p>当日净损益 <b>{dollars(record.pnl_usd)}</b> · 当日净收益率 <b>{percent(record.daily_return)}</b> · 手续费 <b>{dollars(record.cost_usd)}</b>
        {record.fee_model === "legacy_bps" && "（历史固定费率）"}</p>
    </>}
    <div className="table-wrap"><table aria-label={title}>
      <thead><tr>{columns.map(column=><th key={column}>{column}</th>)}</tr></thead>
      <tbody>{rows.length ? rows.map(row=><tr key={row.symbol}><td>{row.rank}</td><td>{row.symbol}</td>
        <td>{row.name??"—"}</td><td>{(review ? row.review_score : row.score)?.toFixed(4)??"—"}</td>
        <td>{dollars(row.allocation_usd??allocation)}</td><td>{row.shares?.toFixed(4)??"—"}</td>
        <td>{dollars(row.entry_open)}</td><td>{dollars(row.exit_open)}</td>
        <td>{dollars(row.pnl_usd)}</td><td>{percent(row.net_return)}</td></tr>) :
        <tr><td colSpan={columns.length}>{empty}{reason && <p>{reason}</p>}</td></tr>}</tbody>
    </table></div>
  </section>;
}

export default function UsOvernight({data,onSession}:{data:UsOvernightData;onSession:(date:string)=>void}) {
  const status = statusText[data.status] ?? data.status;
  const book = data.paper_book;
  const comparison = book?.comparison;
  const baseline = book?.record;
  const review = comparison?.record;
  const historical = Boolean(book?.selected_session && book.selected_session !== data.expected_session);
  return <div className="us-overnight-page">
    <section className="us-overnight-hero">
      <div><small>US OVERNIGHT · PAPER SIGNALS</small><h2>美股隔夜 Top-{data.policy.top_n}</h2>
        <p>前一交易日收盘数据选股，当日开盘入场，下一交易日开盘退出。</p></div>
      <div><span>运行模式</span><b>{data.mode}</b><small>{status}</small></div>
    </section>
    <div className="us-summary-grid">
      <article><small>交易日（纽约）</small><b>{data.expected_session??"—"}</b><span>盘前截止 {data.schedule.signal_cutoff}</span></article>
      <article><small>股票池</small><b>TOP {data.universe.size||"—"}</b></article>
      <article><small>模拟手续费</small><b>按笔计费</b><span>富途香港美股固定式</span></article>
      <article><small>市场门槛预测</small><b>{data.market_forecast_bps?.toFixed(2)??"—"} bp</b></article>
    </div>
    <section className="panel us-candidates">
      <header>
        <div><h3>每日候选与模拟交易</h3>
          {book?.start_session && <p>基准资金 {dollars(book.capital_usd)} · 每只 {dollars(book.allocation_usd)} · 美东交易日</p>}
        </div>
        {book?.start_session && <label>买入日期 <input type="date" aria-label="买入日期" min={book.start_session}
          max={data.expected_session??book.available_sessions[0]} value={book.selected_session}
          onChange={event=>onSession(event.target.value)}/></label>}
      </header>
      <PositionTable title="收盘基准仓" record={baseline} rows={baseline?.positions ?? (book?.start_session ? [] : data.candidates)}
        allocation={book?.allocation_usd} topN={data.policy.top_n}
        empty={baseline ? statusText[baseline.status] ?? baseline.status : historical ? "所选日期没有模拟记录" : status}
        reason={baseline || historical ? undefined : data.reason}/>
      <PositionTable title="盘前更新仓" record={review} rows={review?.positions ?? []}
        allocation={book?.allocation_usd} topN={data.policy.top_n} review
        empty={comparison ? reviewStatus[comparison.status] ?? comparison.status : "所选日期没有盘前更新记录"}/>
      {review && <p>盘前换入 {comparison?.added?.join("、")} · 换出 {comparison?.removed?.join("、")} · 净损益增量 {dollars(comparison?.pnl_delta_usd)}
        {comparison?.return_delta != null && `（${(comparison.return_delta * 100).toFixed(2)} 个百分点）`}</p>}
    </section>
  </div>;
}
