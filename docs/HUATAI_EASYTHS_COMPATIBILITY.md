# 华泰客户端与EasyTHS兼容性核查

2026-10-22；固定上游2.0.7 / c7ea854。状态：静态内核线索存在，运行时只读业务未通过验收。

## 实体机重新登录后的运行时证据

本次通过computer-use只读访问 `D:\htwt\xiadan.exe` 的“网上股票交易系统5.0”。没有启动EasyTHS服务、发送键鼠输入、提交/撤销订单或自动登录；报告不保存账户标识、资金值或姓名。

- 已取得两层59648窗格、HexinScrollWnd/HexinScrollWnd2导航结构。
- 查询菜单下有资金股票、当日委托、当日成交、历史委托、历史成交。这些菜单存在不等于已成功切换或读出数据。
- 当前资金股票页可读1013冻结金额、1014市值、1015总资产、1016可用、1017可取、1027持仓盈亏、1026当日盈亏、1029当日盈亏比。
- **明确差异**：上游资金查询只枚举Static/Text并要求1012资金余额；当前完整控件树未见1012。另有1356窗格显示金额，但其业务含义未验收，不擅自映射balance。上游FundsResult的balance为必填，因此按现有可见结构原版字段装配会缺balance；完整上游execute未运行，不宣称实际收到其异常。
- 持仓表格容器1047→200→Custom1存在，但此次UIA文本未返回行/列；不能宣称持仓读取已通过，也不能把没有行文本当成空仓。
- 状态栏显示已连接，并提示正在清算、查询可能不是当日最终结果。资金可读不等于最终对账或实盘执行就绪。

结论升级为：**运行时窗口与部分资金字段可读；原版资金查询存在明确字段不匹配；持仓/委托/成交回读未验收。** 没有证据支持“改路径即可直接使用EasyTHS”。下一步应做华泰专用资金映射与只读表格回读，不让用户继续安装通用同花顺试错。

## 已有证据

- 宿主安装的 `D:\htwt\xiadan.exe` 文件版本2026.7.7.0，CompanyName为杭州核新软件技术有限公司。
- 可执行文件静态字符串含上游定位使用的 AfxMDIFrame140s、AfxWnd140s、HexinScrollWnd、HexinScrollWnd2、CVirtualGridCtrl、SysTreeView32。只说明同类界面组件，不能证明运行时层级、编号、可访问性或业务兼容。
- 华泰官网列有华泰证券智能交易软件（同花顺版）。普通同花顺的H分类未出现华泰入口，不代表华泰官方客户端没有同花顺内核。
- 固定上游文档、更新记录及代码中未查到明确的华泰专用适配。
- 本次进程与computer-use窗口清单均无华泰交易窗口。没有运行时数据；未自动启动、登录、提权或执行交易操作。

## 不能直接复用的假设

- 资金查询硬编码两层AfxMDIFrame140s、59648及静态字段1012—1017。
- 持仓查询还依赖HexinScrollWnd和CVirtualGridCtrl层级、菜单名称、剪贴板列名。
- 委托查询实际进入F3撤单页，依赖3348过滤框、3349查询按钮、1047表格。普通查询页有表格不代表此路径成立。
- 买入代码使用F1、1032/1033/1034并发送Enter；成功判断包含“无弹窗”假设。原样运行有真实报单风险，且本地成功不等于券商受理。禁止作为兼容性探针。
- 原版入口包含自动账户读取、弹窗处理和验证码识别，不启动整个服务来试错，不自动处理认证或风险确认。

## 本次落地

`easyths_compatibility.py`为无GUI依赖的纯函数；只对已取得的控件清单核对class/automation_id，输出缺失/存在/歧义。即使全部存在也只标SELECTOR_INVENTORY_ONLY，compatible仍为未知，账户/层级/回执/提交均不通过。

原只读worker增加class和automation_id及上述报告；编辑框只保留元数据，不读取内容，Password/Document跳过。不增加点击、键盘、提权或下单方法。本轮未运行该worker去绕过当前computer-use限制；其运行时功能仍待合适受支持的诊断通道验收。

下一真实门禁是已登录客户端的运行时控件和业务回读；如果再次受权限或工具限制，明确记录，不让用户换软件碰运气。尚未证明EasyTHS原版可直接用华泰，也尚未证明无法适配。

来源：

- https://www.htsc.com/site-services/downloads
- https://github.com/noimank/easyths/blob/main/easyths/operations/funds_query.py
- https://github.com/noimank/easyths/blob/main/easyths/operations/holding_query.py
- https://github.com/noimank/easyths/blob/main/easyths/operations/order_query.py
- https://github.com/noimank/easyths/blob/main/easyths/operations/buy.py
# 华泰只读适配进展（2026-10-22）

新增 `huatai_readonly.normalize_funds`，仅解析本地控件清单，不导入或调用交易操作。资金字段逐项保留十进制字符串，缺失、重复、掩码、非法数值明确返回未知；不会利用资产等式推算缺失的余额，也不会把未确认的 1356 控件映射为余额。字段映射仍是待业务验收的临时映射，不能据此下单。

后台清单 worker 已接入该诊断：对外只返回字段状态，不返回金额。清算提示单独标记。持仓、委托、成交表仍返回 UNREADABLE / rows=null，而不是空列表，避免将读不到误认为没有持仓或没有成交。本次没有操作客户端、发送键鼠输入或真实委托；完成的是离线适配与测试，不是实盘验收。

