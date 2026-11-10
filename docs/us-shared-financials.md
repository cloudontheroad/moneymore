# MoneyMore 共享美股财务数据

财务数据落在项目公共 `data/processed`，由 `moneymore.data.us_financials` 读取。TQQQ 研究使用历史行业分类汇总，隔夜 Top-N 研究按已核验的 SEC 公司编号（CIK）取数。

| 表 | 粒度 |
|---|---|
| `us_sec_filings.parquet` | SEC 申报编号，含 CIK、申报日期、受理时间和可用的历史行业信息 |
| `us_reported_financial_facts.parquet` | CIK、申报编号、标签、期间、单位和来源，保留已知修订及同口径冲突 |
| `us_observed_stock_splits.parquet` | 拆股观察，含生效日、公布日及历史身份核验状态 |

截至 2026-11-10，本地共享库共 **7,085,314 条财务观测、9,947 个 CIK、145,141 份有数值的申报**。申报索引共 405,703 份。宽覆盖 FSDS 明细为 2020-10-01 至 2026-03-31，共 6,525,553 条合并口径观测。2026-03-31 后补档现有 47 个 CIK 的 10,988 条观测，最晚披露日为 2026-11-04。 [近端补档与行业检验](research/TQQQ_EXPANSION_CYCLES_20261110.md)

读接口把决策时间转为纽约日期，只使用此前已公开的观测。公开日期采用可取得的 SEC 受理日、正式申报日以及该数值自己的申报日中较晚者。使用同一口径下最近公开的版本，遇到当日冲突留空。月度精度的 SEC FSDS 报表期和精确到日的 Company Facts 分开。未经核验的 Yahoo ticker 拆股观察留在数据表，按 CIK 读取的接口只返回有可靠身份、已公开且已经生效的记录。

年度扩张入口只配对相近财年跨度且口径一致的美元数据，默认排除财年末距决策日超过 550 天的记录。输出资本开支、研发及普通股发行现金流的同比变化，以及两期各自的披露日期。两种资本开支标签分别出数，避免将不同口径的公司样本合并。普通股发行现金流与拆股调整后的股数变化及论文的发股公司比例是不同口径。行业结果同时输出已配对的公司数。

```powershell
python scripts/publish_shared_us_financials.py
python scripts/review_us_expansion.py --as-of 2026-10-06 --industry Chips Softw Hardw LabEq
python scripts/review_us_expansion.py --as-of 2026-10-06 --cik 1045810 2488 789019
```

项目内也可直接调用 `reported_facts_as_of`、`annual_expansion_as_of` 和 `observed_splits_as_of`，传入同一个 `ParquetStore`。Top-N 的历史证券身份需要由调用方按当时有效区间提供，当前 ticker 名录的拆股观察不用于历史映射。本轮入口用于数据验证和研究。

2026-10-06 的样例中，电子设备行业 135 家公司可比较同一资本开支标签，其中 83 家同比增加。把同一批公司放回 2026-07-01，也是 83 家增加，且 135 家的增加状态逐一相同。2026 年二季度后的明细目前来自少量补档发行人，这一比较同时受更新覆盖影响。相同标签在 2021-11-19 为 87 / 182，2022-01-03 为 89 / 181，各年份覆盖公司不同。按 CIK 的样例复核了 NVDA、AMD、MSFT 和 AMZN 各自适用的资本开支标签及披露日。

## 来源与回补

- 早期逐申报明细来自 SECDatabase 的 2020-09 存档，保留四个科技行业。SEC FSDS 镜像的申报索引包含全部行业，数值按筛选过的 XBRL 标签提取。 [历史明细说明](research/TQQQ_EXPANSION_DATA_20261109.md) · [SEC FSDS](https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets) · [归档镜像](https://huggingface.co/datasets/erlenbusch/sec-edgar/tree/75008a0c4943e108944d0bc3982a12ab699ad70a)
- 本轮补档的原始选定标签明细覆盖 2020 年四季度至 2026 年一季度的申报。2020 年四季度至 2023 年有 9,412,893 条，2024 年至 2026 年一季度有 6,030,036 条。共享表只纳入符合合并报表口径的标准标签观测，发布后的行数较少。
- 2026 年补档保留 SEC Company Facts 数值和原始申报编号。JevonsIntel 档案涵盖 25 个公司编号，额外缓存来自 Filings Terminal 的公开逐公司查询。对于新申报没有原始 SIC 的记录，使用该 CIK 前一次已公开的 SEC 行业分类，并在 `industry_basis` 标记来源。 [JevonsIntel 来源清单](https://jevonsintel.com/) · [Filings Terminal 数据实现](https://github.com/masonjbennett/filings-terminal/blob/main/api/facts.js)
- 未包含原始披露版本的宽覆盖快照仍留在原始数据目录，供覆盖审计。拆股记录保留提供方和身份验证状态。 [Varve 数据说明](https://varvedata.com/data)

有本地 SEC DuckDB 归档时，可用下列命令重建近年明细。历史申报索引和 2020 年以前的科技行业补档命令在原研究报告。

```powershell
$archive = 'data/raw/tqqq-expansion/sec-edgar-75008a0.duckdb'
$sic = 'data/raw/tqqq-expansion/Siccodes49.zip'
python scripts/build_tqqq_expansion_data.py --archive $archive --sic $sic --through 2023-12-31 --fact-since 2020-10-01 --output data/processed/tqqq-expansion/fsds-bridge
python scripts/build_tqqq_expansion_data.py --archive $archive --sic $sic --through 2026-03-31 --fact-since 2024-01-01 --output data/processed/tqqq-expansion/fsds-recent
python scripts/publish_shared_us_financials.py
```
