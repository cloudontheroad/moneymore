# TQQQ 研究重启：杠杆状态与独立收益来源

## 原始研究及公开实现

| 来源 | 值得吸收的方法 | 迁移边界 |
|---|---|---|
| [Gayed 与 Bilello：Leverage for the Long Run](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2741701) | 把是否加杠杆与是否持有股票拆开，均线用于识别较适合承担杠杆的状态 | 原研究主要基于美股宽基。QQQ 的 200 日规则以及风险期持有 1 倍 QQQ 是本地迁移对照 |
| [Hurst、Ooi、Pedersen：A Century of Evidence on Trend-Following Investing](https://www.aqr.com/-/media/AQR/Documents/Insights/Journal-Article/AQR-JPM-Fall-2017.pdf) | 将 1、3、12 个月信号等权组合，按月执行，并检查信号延迟 | 原证据来自多市场多空组合。单一纳指长仓版只是机制迁移 |
| [Bongaerts、Kang、van Dijk：Conditional Volatility Targeting](https://repub.eur.nl/pub/130215/Bongaerts-Kang-van-Dijk-Conditional-volatility-targeting-2020-FAJ.pdf) 与 [Cederburg 等：On the Performance of Volatility-Managed Portfolios](https://www.lehigh.edu/~xuy219/research/COWY.pdf) | 对极端波动单独处理，用实时可实现版本与原始持有比较 | 本地对照只处理高波动尾部，并把杠杆限制在 TQQQ 权重 0–100% |
| [Avellaneda、Zhang：Path-Dependence of Leveraged ETF Returns](https://epubs.siam.org/doi/abs/10.1137/090760805) | 拆开标的趋势、实现方差及持有成本，校验合成路径与真实 ETF | 合成数据只作上市前危机压力测试。模型必须单独核对融资、费用与真实 ETF 跟踪误差 |
| [Zarattini、Aziz：Can Day Trading Really Be Profitable?](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4416622) | QQQ/TQQQ 开盘区间突破，独立于隔夜和多日仓位择时 | 论文结果待本地复现，需要分钟数据，按突破发生后的可成交价，计开盘价差及滑点 |
| [Bailey、López de Prado：The Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) | 记录全部试验，评估参数搜索与非正态收益对夏普的影响 | 今后冻结测试方案及参数，保留试验总账，历史滚动窗口不算独立样本 |

公开项目方面，[ravelab/l-etf](https://github.com/ravelab/l-etf) 的滚动区间和实盘 ETF/合成序列分层适合作为验收参照；[dbigham/TQQQ](https://github.com/dbigham/TQQQ) 的融资成本建模和真实 ETF 校验适合借鉴。其温度曲线以及择优后的多层阈值应独立验证，不能直接作为在线信号。[pysystemtrade](https://github.com/pst-group/pysystemtrade/blob/develop/docs/backtesting.md) 的预测组合、仓位计算和执行分层适合作为架构参照。公开项目的回测收益不作为本地验收证据。

## 第一批迁移对照

运行 `python scripts/review_tqqq_literature.py --source yahoo --as-of 2026-11-06`。2013 年首个交易日至 2026-11-06 连续执行，资金与信号跨年承接。真实 ETF 日线，整股，收盘信号，下一开盘成交。满仓基准仅买入一次，现金收益为零，未计现金分红。月度规则在新月份首个收盘读取上月末信号，随后开盘执行。以下均为方法迁移测试，没有搜索参数。

| 策略 | 6 bps 年化 | 25 bps 年化 | 25 bps 夏普 | 25 bps 最大回撤 | 25 bps 三年窗口胜满仓 |
|---|---:|---:|---:|---:|---:|
| 满仓 TQQQ | 38.50% | 38.49% | 0.84 | -81.75% | — |
| v3 | 32.71% | 28.75% | 0.86 | -51.95% | 27.60% |
| 200 日趋势：TQQQ / 现金 | 36.13% | 35.13% | 0.87 | -57.11% | 31.85% |
| 200 日趋势：TQQQ / QQQ | 39.53% | 37.48% | 0.89 | -60.03% | 32.88% |
| 1/3/12 月趋势投票 | 25.86% | 25.10% | 0.73 | -58.69% | 25.62% |
| 仅高波动减半 | 36.77% | 36.30% | 0.86 | -70.28% | 37.21% |

2022 年，按 25 bps，200 日趋势切换到 QQQ 的收益约 -54%，v3 约 -38%；2023–2025 年前者年化 68.49%，v3 61.61%，满仓 83.25%。200 日规则揭示去杠杆和退出股票的成本差异。首轮结果仍没有找到持续大幅领先的候选。三年窗口互相重叠，胜率描述历史覆盖面，不代表统计独立样本。

下一批研究优先独立复现 QQQ/TQQQ 日内突破，固定参数、逐分钟验证填单、隔离与现有日线策略的收益归因。日线方向保留 3x/1x/现金的状态切换及尾部波动作为固定基准。首先验证跨时段表现与成本敏感性，再决定是否组合。
