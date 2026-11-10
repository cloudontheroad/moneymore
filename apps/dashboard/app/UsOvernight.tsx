type Candidate = {rank:number;symbol:string;name?:string;score:number;entry_open?:number;exit_open?:number;shares?:number;allocation_usd?:number;net_return?:number;pnl_usd?:number};

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
    record?:{trade_session:string;exit_session:string;status:string;positions:Candidate[];daily_return?:number|null;pnl_usd?:number|null}|null;
  };
};

const statusText:Record<string,string> = {
  READY:"候选已生成", CASH:"策略空仓", DATA_PENDING:"等待日线或股票池",
  EXPIRED:"盘前截止已过", STALE:"候选已过期", WAITING_FOR_FIRST_RUN:"等待盘前信号",
};

const dollars = (n?:number|null) => n == null ? "—" : n.toLocaleString("en-US",{style:"currency",currency:"USD"});
const percent = (n?:number|null) => n == null ? "—" : `${(100*n).toFixed(2)}%`;
const ledgerStatus:Record<string,string> = {READY:"待开盘价",OPEN:"已记录买入，待下一交易日开盘",SETTLED:"已结算"};

export default function UsOvernight({data,onSession}:{data:UsOvernightData;onSession:(date:string)=>void}) {
  const status = statusText[data.status] ?? data.status;
  const book = data.paper_book;
  const record = book?.record;
  const rows = record?.positions ?? (book?.start_session ? [] : data.candidates);
  const selectedStatus = record ? ledgerStatus[record.status] ?? statusText[record.status] ?? record.status : status;
  return <div className="us-overnight-page">
    <section className="us-overnight-hero">
      <div><small>US OVERNIGHT · PAPER SIGNALS</small><h2>美股隔夜 Top-{data.policy.top_n}</h2>
        <p>前一交易日收盘数据选股，当日开盘入场，下一交易日开盘退出。</p></div>
      <div><span>运行模式</span><b>{data.mode}</b><small>{status}</small></div>
    </section>
    <div className="us-summary-grid">
      <article><small>交易日（纽约）</small><b>{data.expected_session??"—"}</b><span>盘前截止 {data.schedule.signal_cutoff}</span></article>
      <article><small>股票池</small><b>TOP {data.universe.size||"—"}</b></article>
      <article><small>往返成本</small><b>{data.policy.round_trip_cost_bps} bp</b></article>
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
      {record ? <>
        <p>{selectedStatus} · {record.trade_session} 买入 · {record.exit_session} 卖出 · {rows.length} / {data.policy.top_n} 只</p>
        <p>当日净损益 <b>{dollars(record.pnl_usd)}</b> · 当日净收益率 <b>{percent(record.daily_return)}</b></p>
      </> : rows.length === 0 && <div className="us-empty">
        <b>{book?.selected_session !== data.expected_session ? "所选日期没有模拟记录" : selectedStatus}</b>
        {book?.selected_session === data.expected_session && data.reason && <span>{data.reason}</span>}
      </div>}
      {rows.length>0 && <div className="table-wrap"><table>
          <thead><tr>{["排名","代码","名称","盘前分数","分配资金","模拟股数","买入开盘价","卖出开盘价","净损益","净收益率"].map(title=><th key={title}>{title}</th>)}</tr></thead>
          <tbody>{rows.map(row=><tr key={row.symbol}><td>{row.rank}</td><td>{row.symbol}</td>
            <td>{row.name??"—"}</td><td>{row.score?.toFixed(4)??"—"}</td>
            <td>{dollars(row.allocation_usd??book?.allocation_usd)}</td><td>{row.shares?.toFixed(4)??"—"}</td>
            <td>{dollars(row.entry_open)}</td><td>{dollars(row.exit_open)}</td>
            <td>{dollars(row.pnl_usd)}</td><td>{percent(row.net_return)}</td></tr>)}</tbody>
      </table></div>}
    </section>
  </div>;
}
