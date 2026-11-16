"""Ex-post return-based style diagnostics, not causal stock-selection attribution."""
import json
import numpy as np
import pandas as pd
from audit_ashare_pareto_inputs import ROOT, OUT
from report_ashare_pareto_stability import collect


def factors():
    dest=OUT/'return-attribution';dest.mkdir(exist_ok=True)
    path=dest/'daily-factor-proxies.parquet'
    if path.exists():return pd.read_parquet(path).set_index('date')
    panel=pd.read_parquet(OUT/'label-repair-v2/panel.parquet')
    source=ROOT/'state/mature-multifactor/p1-native-20261103-a'
    bars=pd.read_parquet(source/'bars-input.parquet');bars.trade_date=bars.trade_date.astype(str)
    prices=bars.assign(adjusted=bars.close*bars.adj_factor).pivot(index='trade_date',columns='ts_code',values='adjusted').sort_index()
    daily=prices.pct_change(fill_method=None);days=list(prices.index)
    sections={d:g.set_index('symbol') for d,g in panel.groupby('date')}
    rows=[]
    for day in days:
        if not '20260101'<=day<='20260930':continue
        signal=days[days.index(day)-1];f=sections[signal].copy();r=daily.loc[day].reindex(f.index)
        valid=np.isfinite(r)&f.total_mv.gt(0)
        if valid.sum()<800:raise ValueError('INSUFFICIENT_ATTRIBUTION_COVERAGE')
        cap=f.loc[valid,'total_mv'];row={'date':day,'signal_date':signal,'market':float((r.loc[valid]*cap/cap.sum()).sum()),'market_assets':int(valid.sum())}
        for family in ('value_quality','improvement','momentum'):
            score=f[family+'_comparable'];observed=valid&score.notna();rank=score.loc[observed].rank(pct=True)
            high=rank.index[rank.ge(.8)];low=rank.index[rank.le(.2)]
            if len(high)<100 or len(low)<100:raise ValueError('INSUFFICIENT_STYLE_PROXY_COVERAGE')
            row[family]=float(r.loc[high].mean()-r.loc[low].mean())
        rows.append(row)
    result=pd.DataFrame(rows).set_index('date');result.reset_index().to_parquet(path,index=False);return result


def main():
    proxies=factors();cols=['market','value_quality','improvement','momentum'];rows=[]
    for bps in (10,20):
        folders=[(OUT/f'repaired-replays/bps{bps}',''),(OUT/f'baseline-replays/bps{bps}','')]
        folders += [(OUT/f'qlib-replays/interval{i}/full-proof-v4/bps{bps}',f'_{i}') for i in (15,20,25)]
        for folder,suffix in folders:
            if not (folder/'results.json').exists():continue
            metrics,returns=collect(folder,bps,suffix)
            for record in metrics:
                name=record['strategy'];y=returns[name].reindex(proxies.index)
                if y.isna().any():raise ValueError('ATTRIBUTION_DATE_MISMATCH')
                x=np.column_stack([np.ones(len(proxies)),proxies[cols]])
                coef=np.linalg.lstsq(x,y.to_numpy(),rcond=None)[0];fitted=x@coef;residual=y.to_numpy()-fitted
                contributions={col:float(coef[i+1]*proxies[col].mean()*252) for i,col in enumerate(cols)}
                intercept=float(coef[0]*252)
                assert abs(sum(contributions.values())+intercept-float(y.mean()*252))<1e-8
                rows.append({'strategy':name,'bps':bps,'style_coefficients':dict(zip(cols,map(float,coef[1:]))),'annualized_arithmetic_style_contributions':contributions,'unexplained_intercept_not_pure_alpha':intercept,'annualized_arithmetic_mean_return':float(y.mean()*252),'r_squared':float(1-np.var(residual)/np.var(y)) if np.var(y)>0 else None,'fee_and_slippage_over_initial_cash':float((record['fees']+record['slippage'])/10_000_000),'mean_exposure':record['mean_exposure'],'same_exposure_market_proxy_mean_annualized':float(proxies.market.mean()*252*record['mean_exposure']),'proxy_not_executable_counterfactual':True})
    result={'method':'ex-post OLS daily net returns on prior-known Top1000 cap-weight market and comparable-factor top-minus-bottom daily return proxies','not_causal_attribution':True,'not_tradeable_factor_portfolios':True,'includes_intercept_trade_timing_cash_rights_costs_and_idiosyncratic_effects':True,'compounded_return_not_equal_arithmetic_sum':True,'cases':rows}
    path=OUT/'return-attribution/results.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(path,flush=True)


if __name__=='__main__':main()
