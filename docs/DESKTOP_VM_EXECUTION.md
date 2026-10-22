# Windows 隔离执行机：EasyTHS 方案

决策日期：2026-10-21。状态：完成选型、主机只读盘点、安全配置样例与健康检查模块；虚拟机未安装，客户端与回执未验收，真实提交未实现。

## 选型

选择 EasyTHS 作为专用 Windows 虚拟机内的 GUI 执行适配底座，不作为完整订单管理系统。

- easytrader 历史用户积累更大，且有华泰适配，但当前默认分支 `rm` 仅保留 “Removed according to regulations.”，不作为新增部署依赖，不从镜像或旧分支绕过移除继续部署。
- EasyTHS 当前检查版本 2.0.7，源码固定参考提交 `c7ea8545f0ee3b7e2cb3f2ef44191f5e300110ba`（2026-10-20）；要求 Python 3.12。不是因为维护活跃就宣称已达实盘可靠性。
- EasyTHS 已有 HTTP 服务、串行队列、单操作超时、账户定向能力，适合外置执行层。它会设置焦点、输入按键、点击表格和使用剪贴板，因此仅允许在专用 guest 桌面运行。
- 自动化技术可行性与券商授权是两回事。项目开源、能点击普通客户端、使用虚拟机均不代表获得程序化交易接入许可；券商支持与适用规则未确认时不启用真实提交。

来源：

- https://github.com/noimank/easyths
- https://noimank.github.io/easyths/getting-started/ths-client/
- https://github.com/shidenggui/easytrader
- https://raw.githubusercontent.com/shidenggui/easytrader/rm/README.md

## 主机与资源

只读盘点：Windows 11 家庭版中文版、Ultra 9 275HX、24 逻辑处理器、约 63 GiB 内存；C 盘约 180 GiB、D 盘约 347 GiB 可用；常见路径未检测到 VMware/VirtualBox。

Windows 报告 HypervisorPresent=true，但 CPU 的虚拟化与 SLAT 字段返回 false；已存在 hypervisor 时不能据此直接断定 BIOS 未开启，必须以虚拟机实际启动验证。不要为安装虚拟机自动关闭 VBS、内存完整性等安全功能。

主方案 VMware Workstation Pro：无需为 Hyper-V 升级宿主 Windows；微软明确家庭版不支持 Hyper-V 角色。VMware 软件免费不等于 guest Windows 许可免费，guest 必须使用合法授权系统。

- 虚拟机名称 MoneyMore-Execution；位置建议 `D:\MoneyMoreVM\`（尚未创建）。
- 4 vCPU、8 GiB 固定内存、100 GiB 动态虚拟磁盘。只跑交易客户端和 Python 服务，不搬迁行情库、Qlib、GPU 训练。
- guest 建议 Windows 11 x64；交易客户端使用项目推荐的官方同花顺远航版新版交易页面。先确认其实际支持现有华泰账户；不预先承诺必须更换券商。
- 先测试可用的模拟账户；真实账户登录、密码、验证码、协议和风险提示由用户处理，不保存在 MoneyMore 中。

参考：

- https://knowledge.broadcom.com/external/article/368667/download-and-license-free-vmware-desktop.html
- https://learn.microsoft.com/en-us/windows-server/virtualization/hyper-v/get-started/Install-Hyper-V

## 链路与隔离

宿主 MoneyMore（数据、策略、风控、审计） → 受认证的私有隧道 → guest EasyTHS → guest 交易客户端 → 委托及成交回读 → 宿主独立执行账本。

HTTP 是本地 GUI 适配器的接口，不是券商官方 API。网页最后才接入，不影响现有模拟账户和数据流水线。

1. guest EasyTHS 仅监听 `127.0.0.1:7648`，必须替换样例 token；不要使用上游示例的 0.0.0.0、空 token、任意 CORS/IP 默认值。
2. 未来宿主只访问经认证隧道转发的 `127.0.0.1:17648`；隧道未配置时保持阻断。不开放公网、路由器映射或局域网无认证 API。
3. VM 使用 NAT，管理通道若需新增服务或防火墙规则，单独审批并只允许宿主，不能由此开放整个 API。
4. 不将交易密码、完整账户信息、API token、guest 镜像、日志纳入 Git。关闭宿主/guest 共享剪贴板、拖放、共享业务目录等集成功能需要用户确认后配置；guest 自己的剪贴板仍供 GUI 使用。
5. 不在宿主启动 EasyTHS；不接 MCP 给 AI 自由调用买卖。只开放 MoneyMore 后续明确允许的动作。
6. 不启用自动登录或关闭锁屏保护来凑无人值守。guest 锁屏/退出登录/验证码出现时阻断并通知。guest 客户端与代理使用匹配权限，默认普通权限；不自动提权。

## 交互会话与守护

- GUI 代理须在 guest 用户交互会话中运行，不作为 Session 0 Windows 服务，不假定“无论用户是否登录均运行”可操作 GUI。
- 宿主 VM 窗口被遮挡、最小化、控制台关闭、宿主锁屏、RDP 断开分别验收，不能互相替代。guest 内可占前台，宿主不得被抢焦点。
- 不依赖 RDP 保持交易桌面，不承诺断开后仍正常；控制台关闭也须确认 VM 继续运行且 guest UI 可读。
- 重启后先验证 VM、会话、账户、回执与未知订单，再恢复执行。禁止因接口超时自动重发买单。
- VM 不能解决宿主断电、重启和睡眠；宿主掉线仍停止交易。不要宣称用了 VM 就实现高可用。
- 基础镜像快照仅用于安装恢复。开始积累委托后不能直接回滚旧账本继续下单，必须以券商回执重建状态；宿主审计账本为不可回滚的执行记录。

## 验收与交付顺序

| 阶段 | 工作 | 通过证据 |
| --- | --- | --- |
| VM1 基础环境 | 官方安装包、合法 Windows、网络、Python 3.12、固定版本 | guest 启动成功；不影响宿主 GUI；版本与校验记录 |
| VM2 只读业务 | 账户、资金、可卖持仓、当日委托、各笔成交、历史记录 | 与客户端人工核对；缺失返回未知而非 0；账户匹配 |
| VM3 隔离稳定性 | 遮挡/最小化/控制台关闭、断网/进程退出/弹窗/重启 | 有界超时、不会重复动作、宿主无键鼠或剪贴板干扰 |
| VM4 模拟闭环 | 一笔计划、模拟委托号、部分成交、撤单、重启恢复 | 委托/成交唯一关联；数量守恒；恢复不重发 |
| VM5 MoneyMore 集成 | 来源策略、审批与风控、独立执行账本、网页 | 不合并多个全资金模拟账户；不污染既有历史 |
| VM6 实盘评审 | 授权与适用规则、风控、人工确认及紧急停止 | 另行验收与明确授权，当前未启用 |

EasyTHS 持仓/委托接口的成功只说明 GUI 操作结果，HTTP 200 或任务 COMPLETED 不等于券商成交。任务超时可发生于券商受理之后：必须持久化 SUBMISSION_UNKNOWN，先读委托与成交、按账户/日期/券商编号去重；无法唯一匹配时冻结该订单并人工复核。

上游“历史委托”描述与实际成交语义必须检查字段，不按接口名字猜测。不得只按股票和数量绑定策略；同一真实账户的多策略须先明确资金分配、净额合单和分摊规则。当前仅允许一组选定模拟策略的后续映射设计，不自动执行全部挑战账户。

## 本次已落实

- 已另存官方完整Git历史、固定提交源码ZIP并校验，详见 [上游源码快照](EASYTHS_UPSTREAM_SNAPSHOT.md)；未运行第三方代码。

- `deploy/desktop-vm/easyths.config.example.toml`：guest 安全配置样例，未启动服务。
- `src/moneymore/desktop_vm.py`：固定 loopback URL、GET-only、无代理、拒绝重定向、3 秒超时、无重试，只读进程健康检查。
- 模块不导入 EasyTHS、不操作 GUI，不提供报单、撤单、重连或任意命令能力；返回健康仍明确标注账户和回执未验收。
- 运行：在私有隧道已建好且 token 在进程私有环境变量 `MONEYMORE_VM_API_TOKEN` 后，使用项目 Python 执行 `-m moneymore.desktop_vm`。未配置返回 BLOCKED；不要把 token 填入命令参数。

当前阻断：未安装虚拟化软件、未提供 guest Windows 安装介质/许可、未建立 guest 与隧道、未验证同花顺对该华泰账户的支持。本次没有下载安装器、修改系统或防火墙、重启电脑、登录账户或启动 GUI 交易服务。
