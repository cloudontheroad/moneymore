# MoneyMore

MoneyMore 是一个面向个人投资者的A股量化研究与模拟交易系统。当前主策略是银行多因子Top-K组合，并采用Qlib式动态 `risk_degree` 完成组合择时。

当前模式：`PAPER_ONLY`。系统不连接任何真实券商账户，也不会自动发送真实委托。

### 2026-10-21 实时行情迁移

本机实时行情已切换为AlphaFeed（报价、五档、VWAP、实时估值及日内模拟），日线/基本面继续Tushare；不再依赖MiniQMT登录。研究和模型仍在本机，不迁入QMT。

配置：`.env` 中 `MONEYMORE_REALTIME_DATA_SOURCE=alphafeed`、`MONEYMORE_MARKET_DATA_SOURCE=tushare`。SDK安装使用项目可选依赖 `[alphafeed]`；Key放在Git忽略目录 `state/private/alphafeed.env`，变量名 `ALPHAFEED_API_KEY`。健康接口 `/api/health` 返回实际选择，`/api/live-accounts` 返回实际行情来源；最新质量摘要在 `state/runtime/alphafeed-health.json`。

缺失/过期盘口、报价与盘口时间不同步或非连续交易时段，模拟执行采取阻断保护。收盘后接口与47标的已验证；盘中连续稳定性仍待验收。见 [验收记录](docs/ALPHAFEED_ACCEPTANCE_20261021.md)。切换不会重写历史收益，也不恢复真实交易能力。

## 当前能力

新增“桌面执行”页：由网页触发本机只读UIA代理、配置普通交易客户端并准备关联计划；不抢焦点、不输入密码、不发送真实委托。客户端后台读取和券商回执解析尚待验收，详见[桌面执行说明](docs/DESKTOP_EXECUTION.md)。

- Tushare历史行情、估值、财务和分红数据；
- 原始数据不可变快照与Parquet研究表；
- point-in-time财务特征；
- 21因子注册表和横截面预处理；
- 因子Rank IC、ICIR和分层收益分析；
- 银行股历史股票池；
- 银行多因子预测分数；
- Qlib式Top-K缓冲换仓；
- 固定风险度、KAMA和波动率目标择时；
- Qlib `risk_degree` 语义的动态组合仓位；
- Qlib深度学习挑战者与候选者使用100%风险预算，不再固定40%仓位；
- 因子基线采用全候选池统一横截面Top-K和100%总风险预算；行业只做展示归因，不参与名额与权重；
- 因子和既有挑战账户继续按各自不可变协议运行；后续Qlib v2候选使用LightGBM＋多种子GRU异构融合、12维价量/趋势/风险特征，以及带等权收缩的置信度Top-10权重，单股目标仍限制在5%～15%；
- Qlib v2研究门禁失败时只记录分数，不允许新增模拟持仓；任何候选仍需独立前瞻观察和人工晋级；
- T日信号、T+1开盘模拟撮合；
- 订单、成交、持仓、现金和对账账本；
- 交易连续性补偿规划器：故障恢复后基于最新目标与券商快照计算差额，含未完成订单门禁和持久化幂等预约；当前仅规划/验证，不向券商发送真实委托；
- 每日18:30服务端自动流水线；
- 组合研究与运行观测前台。

详细路线见[规划文档](docs/PLAN.md)。

## 策略结构

```text
Tushare点时数据
      ↓
MoneyMore因子注册表
      ↓
横截面预处理与多因子分数
      ↓
Qlib式Top-K策略
Top 8 / Rank 12缓冲 / 最多替换2只
      ↓
Qlib risk_degree择时
固定80% / KAMA / 12%波动率目标
      ↓
目标权重与风险约束
      ↓
PaperBroker次日开盘模拟成交
      ↓
订单、成交、持仓和现金对账
```

Qlib在本项目中是研究和策略接口的主体规范。MoneyMore当前使用轻量原生实现对齐Qlib的信号、Top-K和 `risk_degree` 语义，以保留A股交易规则和本地数据控制。后续接入原生Qlib模型训练器时，不需要修改模拟撮合和前台接口。

## 当前银行策略

模型：`bank_multifactor_no_quality_candidate`

因子组：

- 价值：47.0588%
- 防御：29.4118%
- 动量：23.5294%
- 质量：0%

组合规则：

- 当前银行股票池39只；
- 选择Top 8；
- 持仓跌出Rank 12才触发正常退出；
- 每期最多替换2只；
- 每只股票基础权重为银行子账户10%；
- 固定基准总风险度80%。

当前择时：`vol_target_12`

- 60日实现波动率；
- 年化目标波动率12%；
- `risk_degree`范围20%～80%；
- 10个百分点为调整台阶；
- 每20个交易日重新评估；
- 状态为 `PROSPECTIVE_CANDIDATE`。

最终单股目标权重计算：

```text
单股目标权重 = 10% × risk_degree ÷ 80%
```

例如 `risk_degree=70%`：

- 银行子账户总持股目标约70%；
- 8只股票每只约8.75%；
- 银行子账户约30%保留现金；
- 如果银行预算占总账户10%，则总账户银行持股约7%。

## 环境要求

- Windows；
- Python 3.11或更高版本；
- Node.js仅用于前台；
- Tushare Token；
- 建议16GB内存和SSD。

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

复制环境变量文件：

```powershell
Copy-Item .env.example .env
```

在本地 `.env` 中填写：

```text
TUSHARE_TOKEN=你的Token
```

不得提交 `.env`、Token、行情数据、运行状态或本地账本。

## 数据同步

同步股票数据：

```powershell
.\.venv\Scripts\python.exe -m moneymore.cli sync-tushare `
  --asset stocks `
  --start 20150101 `
  --end 20260727 `
  --batch-days 200
```

数据目录：

```text
data/raw        Tushare不可变原始快照
data/processed  去重后的研究表和研究产物
```

ETF需要独立权限。当前ETF权限尚未开通，不应使用降级或网页抓取方式伪造ETF数据。

## 运行银行择时研究

```powershell
.\.venv\Scripts\python.exe scripts\research_bank_timing.py
```

该命令使用完全相同的Top-K、交易费用和T+1成交规则，对比：

- `fixed_80`
- `kama_10_2_30`
- `vol_target_12`

输出：

```text
data/processed/bank_timing_report.parquet
data/processed/bank_timing_degrees.parquet
data/processed/bank_timing_equity.parquet
```

历史结果已参与模型开发，只能作为验证证据。2026-07-27之后的模拟记录才属于新前瞻证据。

## 启动组合控制台

```powershell
.\scripts\start_dashboard.ps1
```

访问：

- 前台：[http://127.0.0.1:3000](http://127.0.0.1:3000)
- API健康检查：[http://127.0.0.1:8788/api/health](http://127.0.0.1:8788/api/health)
- 银行组合数据：[http://127.0.0.1:8788/api/bank-dashboard](http://127.0.0.1:8788/api/bank-dashboard)
- 择时研究数据：[http://127.0.0.1:8788/api/bank-timing](http://127.0.0.1:8788/api/bank-timing)
- 模拟成交数据：[http://127.0.0.1:8788/api/bank-execution](http://127.0.0.1:8788/api/bank-execution)

停止服务：

```powershell
.\scripts\stop_dashboard.ps1
```

服务只监听本机回环地址，不对公网开放。

## 每日流水线

服务端每天18:30（Asia/Shanghai）自动运行，不依赖Codex自动任务。

流水线执行：

1. 交易日检查；
2. 行情和基本数据增量入库；
3. 前一日委托开盘模拟撮合；
4. 因子和银行多因子分数计算；
5. Top-K缓冲换仓；
6. 动态 `risk_degree` 计算；
7. 下一交易日目标委托生成；
8. 订单、成交、持仓和现金对账；
9. 运行证据归档。

流水线报告：

```text
state/bank-shadow/YYYYMMDD.json
```

信号和报告均按日期固化，避免同一交易日被静默覆盖。

## 测试

后端：

```powershell
.\.venv\Scripts\ruff.exe check src tests scripts\research_bank_timing.py
.\.venv\Scripts\pytest.exe -q
```

前端：

```powershell
cd apps\dashboard
pnpm run lint
pnpm test
```

## 重要口径

- Top-K是交易策略：决定何时调仓、买谁、卖谁和换多少；
- `risk_degree`是组合择时：决定银行子账户投入多少风险资金；
- 因子分数不等于买入建议；
- 历史最优不等于前瞻有效；
- KAMA在本银行组合中表现较差，不能因算法成熟而直接采用；
- 波动率目标只是前瞻候选，不是已证明的最优策略；
- 页面中的持仓、订单和收益均为模拟结果；
- 项目不提供收益保证或个股投资建议。

## 项目边界

当前阶段不做：

- 真实券商账户连接；
- 程序化实盘下单；
- 高频或日内交易；
- 未授权网页行情抓取；
- 用未来披露数据回填历史；
- 用同一历史区间反复搜索并宣称得到纯样本外结论。

后续工作以持续模拟、模型治理、组合归因和前瞻证据积累为主。

## 多行业统一影子账户

当前前台主账户为 `multi_sector_shadow`。银行、红利、工业有色、芯片和创业板成长的目标仓位会统一拆分为最终股票权重，并进入同一个模拟资金、订单、成交、持仓和现金账本。

```text
五行业研究快照
→ 行业风险预算与择时仓位
→ 最终单股目标权重
→ T+1 开盘影子撮合
→ 综合对账与行业归因
```

原 `bank_shadow` 继续保留，用作银行策略的独立基线，不再代表前台综合账户。

综合执行接口：

- [http://127.0.0.1:8788/api/multi-sector-execution](http://127.0.0.1:8788/api/multi-sector-execution)

该接口同时返回日度净值、目标与实际仓位偏差、行业收益归因和风险告警。数据陈旧或对账失败属于阻断级事件，不生成新的模拟订单。

公司行为已经进入综合模拟账户：系统在权益登记日冻结持股数量，在支付日计入税后现金分红，在送转股上市日增加股份并摊薄持仓成本。所有入账记录均幂等并参与现金、股份和成交的统一对账。

数据质量中心每日检查目标股票行情、复权、估值、财务、分红、ETF披露和行业评分。检查结果分为 `PASS`、`WARN` 和 `BLOCK`；出现阻断项时不会撮合旧订单，也不会生成新订单。

- [http://127.0.0.1:8788/api/data-quality](http://127.0.0.1:8788/api/data-quality)

综合账户使用四级风险状态机：`NORMAL`、`REDUCE_ONLY`、`SELL_ONLY`、`SUSPENDED`。风险升级自动执行，风险解除后必须在前台人工确认恢复；所有状态转换均保留审计记录。

每日流水线支持步骤级恢复。银行数据与模型、行业研究、综合组合执行分别记录尝试状态；单步失败会自动重试一次，同一交易日手动补跑会跳过已经完成的步骤。任务失败、数据阻断、风险暂停、对账失败和订单延期会写入本地告警中心，并可在前台运维页确认。

- 任务、步骤与告警：[http://127.0.0.1:8788/api/task-operations](http://127.0.0.1:8788/api/task-operations)

模型治理采用内容寻址版本。综合模型每次运行都会绑定代码、配置、股票池和数据截止日的哈希指纹；内容未变化时复用同一版本，任一内容变化时自动生成新版本。历史诊断、前瞻候选和正式影子信号使用独立证据阶段，订单可反查产生它的具体模型版本。

- 模型版本、证据与绑定：[http://127.0.0.1:8788/api/model-registry](http://127.0.0.1:8788/api/model-registry)

综合影子账户提供三类归因：

- 收益归因：行业配置、行业内选股、开盘交易时机、现金、分红、滑点和交易成本；
- 风险归因：近60日样本下的波动、相关性、集中度和边际风险贡献；
- 执行归因：整手、T+1、涨跌停、现金不足及其他撮合约束造成的仓位缺口。

每天同时记录归因对账。只有各收益分项之和与账户实际权益变化的差额不超过一分钱，归因状态才视为匹配。

## 产品化运维中心

前台“运行与对账”页面聚合下一次自动运行时间、任务步骤、待执行订单、延期原因、告警历史和归因账本。手动补跑前可以输入 `YYYYMMDD` 进行只读影响预览，查看哪些步骤会跳过、哪些步骤会运行，以及多少旧订单具备撮合资格。

- 运维中心：[http://127.0.0.1:8788/api/operations-center](http://127.0.0.1:8788/api/operations-center)
- 补跑预览示例：[http://127.0.0.1:8788/api/tasks/daily-run/preview?trade_date=20260729](http://127.0.0.1:8788/api/tasks/daily-run/preview?trade_date=20260729)

预览接口不会启动流水线或修改模拟账户。本地交易日历未覆盖指定日期时，页面会明确显示 `WEEKDAY_ESTIMATE`，避免把工作日推断误认为正式交易所日历。

## 月度换仓验收

正式影子证据从 `20260727` 开始累计。验收至少需要20个交易观察日，并真实覆盖首次建仓、持有、卖出、同日买卖换仓、新标的买入、每日对账和模型冻结。前台运维中心显示每项检查的 `PASS/WAIT` 状态。

- 当前验收状态：[http://127.0.0.1:8788/api/monthly-acceptance](http://127.0.0.1:8788/api/monthly-acceptance)

只有全部检查通过，系统才会在 `state/monthly-acceptance/` 固化不可变验收报告。代码已经完成并不代表策略已经通过月度验收；实际交易证据不足时状态始终为 `COLLECTING_EVIDENCE`。

## 运行保障

每日任务首先同步上交所正式交易日历，滚动覆盖过去31天和未来90天。正式日历没有覆盖目标日期时，流水线失败并告警，不允许用普通工作日推断继续撮合。数据质量检查中的 `OFFICIAL_CALENDAR_COVERAGE` 是阻断级检查。

- 运行就绪状态：[http://127.0.0.1:8788/api/operational-readiness](http://127.0.0.1:8788/api/operational-readiness)

就绪状态含义：

- `READY`：正式日历、服务端调度和调度线程均正常；
- `REQUIRES_SYNC`：启动流水线后必须先补齐正式日历；
- `BLOCKED`：调度被暂停或后台线程未运行，需要人工处理。

页面中的 `WEEKDAY_ESTIMATE` 永远只用于预览警告，不能授权生成或撮合模拟订单。

统一健康状态总线：

- [http://127.0.0.1:8788/api/system-health](http://127.0.0.1:8788/api/system-health)

健康总线统一检查守护心跳、API/Web进程、QMT行情时效、调度线程、SQLite账本、账户对账和磁盘容量。出现阻断项时返回 FROZEN，开盘撮合与日内执行共用该门禁并停止新增交易。Dashboard每5秒刷新该状态；本地守护器每10秒写入 state/runtime/supervisor-heartbeat.json，子进程重启记录保存在 logs/supervisor-events.jsonl。

量化主机的Windows可用性配置由管理员脚本管理：

- 安装：powershell -ExecutionPolicy Bypass -File scripts/configure_quant_host.ps1 -Action Install
- 审计：powershell -ExecutionPolicy Bypass -File scripts/configure_quant_host.ps1 -Action Audit
- 回滚：powershell -ExecutionPolicy Bypass -File scripts/configure_quant_host.ps1 -Action Rollback

安装动作禁止有用户登录时被Windows Update自动重启，注册 SYSTEM 级 MoneyMore开机任务和当前交易用户的MiniQMT登录任务。原Windows Update策略首次安装时备份至 state/runtime/windows-availability-before.json；脚本不会配置不安全的Windows自动登录。
