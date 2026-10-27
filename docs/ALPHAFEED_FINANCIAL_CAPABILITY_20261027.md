# AlphaFeed 财务补缺能力核验（2026-10-27）

结论：本次公开接口、已安装SDK和实际账户返回均未发现可补齐单季度营收同比及其公告日/报告期的接口，不能用来解除财务修复版门禁。不是仅因当前Key报无权限而得出的结论，也不能据此声称未来或未公开产品永远不会支持。

## 核验依据

1. 官方 https://api.alphafeed.org/openapi.json 实际HTTP 200，paths为depth/depth-batch、instruments、klines/klines-batch/ex-factors/intraday/intraday-batch、quotes，共9个路由，没有financials/income财务路由。
2. OpenAPI及本地SDK虽有FinancialsResponse类型定义，但没有任何公开路由引用该类型。存在类型占位不能当作财务服务可用。
3. 官方文档索引 https://docs.alphafeed.org/llms.txt 与本地SDK资源对象均未提供财报/利润表资源；SDK只有klines、quotes、depth、instruments。
4. 使用本机私密Key只进行了2次有界请求、关闭重试：批量instruments及批量quotes，各查询001246.SZ（缺指标）、600900.SH（指标过旧）、600519.SH（对照）。三个标的均返回正常。
5. instruments仅基础标识、名称、上市日、股数、涨跌停、价格步长等；quotes仅价量及换手等行情扩展字段。未返回ROE、单季度营收增速、财报公告日或报告期。

本次没有写入任何财务因子、生产推荐、订单或历史收益；没有试探不存在的收费接口、升级套餐或回退QMT。

## 下一条可验证路线

Tushare官方利润表接口income提供revenue/total_revenue、ann_date/f_ann_date、end_date及report_type，可先小样本验证缺失股票的原始利润表是否足够，再按同一合并范围、报告期及公告可用时间构造单季度收入同比。

不能直接用累计or_yoy替代单季度q_sales_yoy；也不能混用营业收入与营业总收入、母公司表与合并表，或忽略修订版本可用日期。若同比基期尚未存在/不适用，应明确NA及策略适用性，不能承诺所有新股均可补出同比。本次尚未调用或验收该替代路线。

参考：https://github.com/alphafeed-org/alphafeed-python-sdk 、https://docs.alphafeed.org/ 、https://tushare.pro/document/2?doc_id=33 。
