# TQQQ 高位风险的论文复核与交易检验

## 论文中的信号分工

- Greenwood、Shleifer、You 从美国行业和国际板块中筛出暴涨事件，记录后续两年内的回撤，同时保留继续上涨的对照组。上涨加速、波动上升、发行新股、年轻公司领涨用于区分事件。美国两年翻倍样本 40 次，21 次随后出现至少 40% 的回撤。随后崩盘的组从首次预警到峰值平均还涨约 30%，耗时约五个月。两年风险适合做背景判断。 [Bubbles for Fama，2019](https://www.hbs.edu/ris/Publication%20Files/Bubbles%20for%20Fama_e680848b-4f23-478a-b6cd-e5564e6a431b.pdf)
- Chen、Hong、Stein 以过去六个月个股换手率相对更早十八个月的上升，结合过去三年涨幅，预测未来收益分布的负偏度。指数层面的证据较弱。后续研究在美国市场整体的时间序列上未发现成交量有预测力。这里以 ETF 成交股数和历史负偏度作为代理，目标改成未来四周的 TQQQ 下行。 [Forecasting Crashes，2001](https://scholar.harvard.edu/files/stein/files/forecasting-crashes-jfe-sep-01.pdf) · [Aggregate Stock Market Volume and Negative Return Skewness，2006](https://www.sciencedirect.com/science/article/pii/S0927539805000526)
- Andersen、Fusari、Todorov 从短期期权曲面提取尾部风险，发现波动率水平无法涵盖其变化。这里用 SPX SKEW、VVIX 及其短期变化做代理，覆盖的是标普期权市场。原论文的尾部测度需要完整的期权报价，和这些指数有区别。 [Pricing of Short-Term Market Risk，2017](https://www.nber.org/papers/w21491) · [Cboe 指数方法](https://cdn.cboe.com/api/global/us_indices/governance/Volatility_Index_Methodology_Selected_Broad_Based_Index_Equity_and_ETF_Volatility_Indices.pdf)
- Arif、Sul 在 49 国的 240 次行业暴涨中检验基本面，发现行业净经营资产应计增长与后续崩盘及盈利低于预期有关。净经营资产变化同时覆盖营运资本和长期经营资产。可将其与新股发行比例组合成独立于行情的脆弱性指标。该信号需要按当时披露时间构建，留待独立验证。 [Does Accounting Information Identify Bubbles for Fama?，2024](https://www.sciencedirect.com/science/article/pii/S0165410124000417) · [SEC EDGAR XBRL 数据](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)

## 美国行业复核

使用 Kenneth French 1926-07 至 2026-09 的价值加权行业收益和市值加权市场收益。剔除 Other，至少 10 家公司，前五年行业累计收益至少 50%（1930 年底前豁免），两年行业总收益和行业累计收益减市场累计收益均达门槛，同一行业 24 个月内只记录第一次。完整观察未来 24 个月，按含信号月收盘净值的月度路径计算峰谷回撤。French 公开组合及其历史修订与原论文逐股构造有差异。 [Kenneth French 数据库](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)

| 两年涨幅门槛 | 截至 2012-03 的事件 | 后续回撤至少 40% | 2012-03 后完整事件 | 后续回撤至少 40% |
|---|---:|---:|---:|---:|
| 50% | 137 | 35（26%） | 12 | 1 |
| 100% | 26 | 16（62%） | 4 | 0 |
| 150% | 11 | 10（91%） | 2 | 1 |

100% 组中，最终大跌的 16 例在触发后平均还涨 27.3%，平均 5.3 个月才到本次回撤的起始峰值。26 例触发后 12 个月的平均总收益为 +7.25%。2012 年后的事件较少，且同一时期不同行业的结果可能相关。三个门槛各自应用去重，事件集合可重叠。

## TQQQ 四周检验

从每周最后一个交易日收盘发出信号，下一开盘起算，四周后同口径结算。尾部事件指其间任一开盘或收盘较该入场开盘低至少 20%。共同样本要求 QQQ 的 63 日收盘高点距信号不超过 21 个交易日，四周窗口已结算，各组数据都完整。滚动重训的正则化逻辑回归仅使用当日已结算标签。基础输入为高点距离、5/21 日收益、63 日波动、HYG/IEI 与 VIX/VIX3M 五日变化。逐组增加半导体相对涨幅和加速、成交与负偏度、期权代理。代码检查了 2022-10-14 截断点的特征及预测一致性。

警报使用各模型此前 104 周预测的第 80 百分位，至少需 26 个有效预测。按警报日期顺序选取互不重叠的四周离场窗口，下一开盘卖出，四周后下一开盘回补，现金按零利息，扣除单边 25 bps。下表是离场收益减满仓收益的均值，单位为百分点。各组在独立触发下的交易次数不同，时间上不重叠的窗口仍可能处于同一市场周期。

| 输入 | 355 周 AUC | 互不重叠窗口命中 / 警报 | 平均净离场收益差 | 高点 3% 内的 AUC | 高点 3% 内的平均净离场收益差 |
|---|---:|---:|---:|---:|---:|
| 基础 | 0.515 | 6 / 31 | −2.79 | 0.456 | −2.21 |
| 加半导体上涨与加速 | 0.555 | 5 / 28 | −4.92 | 0.537 | −1.25 |
| 加成交与负偏度 | 0.530 | 7 / 31 | −1.12 | 0.548 | +0.74 |
| 加 SPX 期权代理 | 0.446 | 4 / 32 | −3.88 | 0.374 | −4.85 |
| 全部 | 0.528 | 4 / 27 | −1.37 | 0.527 | +0.20 |

共同预测样本为 2016-04-08 至 2026-10-02，355 周有 46 个尾部标签。高点 3% 内有 269 周、31 个尾部标签。仅保留 2026 年前，高点 3% 内成交与负偏度组有 22 个不重叠警报、4 次命中，平均净离场收益差为 +0.42 个百分点。换成 15% 或 25% 下行标签时，完整样本所有模型的平均离场收益差仍为负。高点 3% 内的成交与负偏度组随标签变化依次为 −0.11、+0.74、−3.97 个百分点。成交使用 Yahoo 历史成交股数，缺少流通股数分母。期权使用官方日线。 [Cboe SKEW](https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv) · [Cboe VVIX](https://cdn.cboe.com/api/global/us_indices/daily_prices/VVIX_History.csv)

## 2026 年 10 月复盘

将模型系数冻结在 2025 年底。9 月 25 日半导体上涨组已经预警，10 月 2 日没有预警。9 月 28 日开盘至 10 月 6 日收盘，TQQQ 仍涨 7.6%。从 9 月 28 日开盘算，四周窗口内最低开收盘下跌约 16.1%。从 10 月 5 日开盘算，四周窗口内最低开收盘下跌约 23.4%。10 月 9 日上涨组再次预警，期权和成交组到 10 月 16 日才越过各自门槛，当日 QQQ 已较高点跌约 6.6%。冻结模型在 2026 年只有 19 个已结算观察、3 个尾部标签。10 月 9 日及之后的四周结果截至 11 月 6 日尚未完整结算。

因此保留半导体上涨与加速作观察背景，单独触发清仓未通过历史检验。下一项可单独验证的假设是，行业融资和净经营资产扩张在高位积累脆弱性，信用与市场内部转弱提供执行确认。验证时需保留当时行业成员和财报首次公开时间，跨行业留出，对比旧确认规则在同样交易次数下的提前量、误卖成本和回撤。这轮产出为研究脚本和结果，模拟盘未切换。

复跑 `python scripts/review_tqqq_crash_literature.py --history <market.csv> --context <peak-context.csv> --alternate <peak-styles-alt.csv> --volume <QQQ-and-SOXX-volume.csv> --skew <SKEW_History.csv> --vvix <VVIX_History.csv> --industries <49_Industry_Portfolios_CSV.zip> --factors <F-F_Research_Data_Factors_CSV.zip>`。项目行情缓存沿用上一轮，成交量文件以交易日为索引，QQQ、SOXX 各一列。
