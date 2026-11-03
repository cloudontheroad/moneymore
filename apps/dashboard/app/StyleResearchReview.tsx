type Row = Record<string, unknown>;

export type StyleResearchReviewData = {
  title?: string; reason?: string; time_range?: string;
  gates?: Row[]; comparison?: Row[];
  completed_studies?: number; account_cases?: number; fills?: number;
};

const text = (value: unknown) => value == null ? "—" : String(value);
const measuredPercent = (value: unknown) => value == null || !Number.isFinite(Number(value))
  ? "未完成" : `${(Number(value) * 100).toFixed(2)}%`;

export default function StyleResearchReview({review}: {review?: StyleResearchReviewData | null}) {
  if (!review) return null;
  return <section className="panel" aria-label="风格均衡研究验收">
    <header><div><h3>风格均衡研究验收</h3><p>{text(review.time_range)} · 历史研究，不计入前瞻日，不自动替换现有账户</p></div><span>•••</span></header>
    <h4>{text(review.title)}</h4><p>{text(review.reason)}</p>
    <div className="table-wrap"><table><thead><tr><th>验收项</th><th>结论</th><th>依据</th></tr></thead>
      <tbody>{(review.gates ?? []).map((row, index) => <tr key={index}><td>{text(row.stage)}</td><td>{text(row.status)}</td><td>{text(row.reason)}</td></tr>)}</tbody>
    </table></div>
    {!!review.comparison?.length && <details><summary>查看六组同口径成本对照（独立21日窗口均值，非连续账户收益）</summary>
      <div className="table-wrap"><table><thead><tr><th>研究方法</th><th>10bps窗口均值</th><th>2bps窗口均值</th><th>同单纯成本影响</th><th>双创股票袖套均值</th><th>完整归因窗口</th></tr></thead>
        <tbody>{review.comparison.map((row, index) => <tr key={index}><td>{text(row.name)}</td><td>{measuredPercent(row.return_10bps)}</td><td>{measuredPercent(row.return_2bps)}</td><td>{measuredPercent(row.pure_cost_delta)}</td><td>{measuredPercent(row.board_weight)}</td><td>{text(row.matched_windows)}</td></tr>)}</tbody>
      </table></div>
      <p className="muted">双创指科创板和创业板，不等于科技行业；均值仅统计持股且非终端退出日。已计交易成本，分红为税前口径。成本重跑可能改变交易路径；同单成本影响固定原订单，单独计算摩擦差额。未匹配持仓单列，不补零，也不冒充选股增益。</p>
    </details>}
  </section>;
}
