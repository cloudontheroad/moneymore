# TQQQ 公司扩张研究数据

这轮先建立可按申报时间查询的财务底座，范围覆盖科技行业及其他行业对照。研究沿用 Arif 和 Sul 的年度净经营资产口径。净经营资产等于总资产减现金和短期投资、减总负债、再加短期和长期债务。本财年与上财年的净经营资产差额除以两期平均总资产。论文还将公司指标做横截面排名，再按当时市值汇总到行业。提取脚本保留原始财务字段和可完整计算的公司年报值。行业市值加权需要另行补齐历史证券身份和市值。 [论文作者稿](https://mitsloan.mit.edu/sites/default/files/inline-files/Arif_Sul.pdf)

## 申报主表

SEC 原站从本机请求返回 403，采用公开 SEC Financial Statement Data Sets 镜像的固定版本 `75008a0c4943e108944d0bc3982a12ab699ad70a`。逐表查询确定该镜像的申报历史覆盖 **2009-04-15 至 2026-03-31**。`filings.parquet` 已在本地通过行数、申报编号唯一性与日期核对。表中有 **405,585** 份年报、季报及修订，涵盖 **16,673** 个不同的 CIK，其中 **18,131** 份是修订版。样本由历史申报构建，包含后来退出市场的主体，也可能包含非普通股或非美国上市主体。不能直接把 CIK 数当成可交易股票数。 [SEC 原始数据说明](https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets) · [固定镜像](https://huggingface.co/datasets/erlenbusch/sec-edgar/tree/75008a0c4943e108944d0bc3982a12ab699ad70a)

| 按申报时 SIC 分类 | 历史 CIK 数 | 申报数 |
|---|---:|---:|
| 电子设备（FF49 Chips） | 595 | 15,591 |
| 其中半导体及相关器件（SIC 3674） | 261 | 6,529 |
| 电脑硬件（Hardw） | 152 | 3,875 |
| 软件（Softw） | 1,228 | 22,479 |
| 测量及控制设备（LabEq） | 184 | 5,766 |

分类采用每份申报的 SIC，再映射至 Kenneth French 49 行业。一个 CIK 历史上可以更换行业，表内各行业的公司数不可相加。 [SEC 字段字典](https://www.sec.gov/files/financial-statement-data-sets.pdf) · [Kenneth French 行业数据](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)

## 补充数据

| 来源 | 已下载并验证 | 用途 |
|---|---|---|
| Zalize 从 SEC Frames 整理的财务快照 | 272,123 条、12,072 个 CIK，财报期 2016-02-26 至 2026-07-05 | 核对覆盖率。缺原始申报日期和数值版本，隔离在原始目录 |
| Varve 财务免费样本 | 27 个 CIK、1,780 条季报、524 条年报，披露日期从 2009-07-22 至 2026-10-30 | 较近申报、资本开支及股数的抽检样本。季度有 1,561 条资本开支及 1,473 条期末股数 |
| Varve 拆股免费样本 | 10 个 CIK、14 条拆股记录 | 验证拆股会怎样改变期末股数。NVDA 的 2024 年 10 拆 1 已对照公司公告 |
| Yahoo 拆股观察 | 200 个科技证券代码，199 个成功，78 个 CIK 共 130 条拆股事件，事件日期 2009-05-11 至 2026-09-03 | 用于补充校正。请求按四个行业各选历史申报数最多的 50 个可匹配代码，代码来自近期名录，仍需核对历史代码和证券身份 |

Zalize 财务快照使用的是当前公布的历史数值，且行业关联也只来自一个当前截面。Varve 是少量公司样本，不能用其覆盖度代替历史行业覆盖。 [Zalize 数据字典](https://huggingface.co/datasets/zalizedata/us-public-company-financials-dataset) · [Varve 样本说明](https://varvedata.com/data) · [NVIDIA 拆股公告](https://investor.nvidia.com/files/doc_downloads/2024/06/nvidia-2024-stock-split_faq_investors.pdf)

## 历史财务数值补档

通过 SECDatabase 2020-09-30 的公开 XBRL 存档取四个科技行业的财务数值，并用 CIK 加申报编号关联 SEC 主表的受理时间和当时行业。已落盘 36 个分片，去重后有 **453,821 条财务数值、50 个标签、4,664 份申报、1,257 个 CIK**，受理日期为 **2009-04-30 至 2020-09-30**。其中 3,866 份有资本开支，1,504 份有普通股发行现金流标签。1,177 份原始年报中，11 份具备严格计算年度净经营资产同比所需的全部标准标签。该存档的每个分片包含多个主体和时期，各公司申报历史的完整度不一。覆盖清单见 `data/processed/tqqq-expansion/s3-history/manifest.json`。 [历史归档与表说明](https://github.com/secdatabase/SEC-XBRL-Financial-Statement-Dataset)

## 研究用口径与复跑

- 以 CIK 和 accession 关联申报与数值。保留历史比较期、原报和修订版本。尚未校验日内时区时，按受理日和正式申报日中较晚者的下一交易日使用。
- 全量数值提取完成时，`facts.parquet` 保存总资产、负债、现金、投资、债务、资本开支、研发、并购、普通股发行现金流、回购和股数。保留单位、报表期间、口径维度与来源标签。
- `annual_noa.parquet` 只使用同份未标为修订的年度报表内可比的两期合并数值。所需字段缺失时留空。尚未整合公司自定义标签，严格字段覆盖是下限。
- 股数、增发现金流、回购和拆股是四种不同数据。后续按历史拆股事件统一股数口径，再区分股权融资、并购换股和员工股权激励。Varve 样本仅用来核验校正。

原始文件位于 `data/raw/tqqq-expansion/`，处理后的 Parquet 位于 `data/processed/tqqq-expansion/`。项目已忽略这两个数据目录。复跑使用 `duckdb` 和项目既有的 `pandas`。

```powershell
$archive = 'https://huggingface.co/datasets/erlenbusch/sec-edgar/resolve/75008a0c4943e108944d0bc3982a12ab699ad70a/sec_edgar.duckdb'
python scripts/build_tqqq_expansion_data.py --archive $archive --sic data/raw/tqqq-expansion/Siccodes49.zip --through 2026-03-31 --output data/processed/tqqq-expansion/fsds-remote --filings-only
python scripts/sync_tqqq_expansion_history.py --filings data/processed/tqqq-expansion/fsds-remote/filings.parquet --output data/processed/tqqq-expansion/s3-history --shards 36
```

首次复跑可从 Kenneth French 数据库下载 `Siccodes49.zip` 放入原始目录。远程索引读取已成功，扫描数值明细时镜像返回 429，广覆盖明细尚未落盘。全量提取使用本地 DuckDB 文件，去掉 `--filings-only`。可用 `--industries Chips Softw Hardw LabEq` 限定为四个科技行业，或用 `--fact-since` 限定近期财务明细并保留年度比较所需的前期数值。
