"""Original baseline/V2 constructors under the common previous-close share engine."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits
from filelock import FileLock
from audit_ashare_pareto_inputs import ROOT, OUT
from replay_four_strategy_2026 import StudyStore
from moneymore.research.baseline_v2_model import build_v2_snapshot
from moneymore.research.baseline_v2_daily_readiness import construct_daily_targets
from moneymore.research.sector_model import build_global_factor_snapshot
from moneymore.portfolio_constructor import global_topk_portfolio, trailing_return_correlation
import research_ashare_paper_routes as engine


class BaselineConstructors:
    def __init__(self,bps=10):
        self.store=StudyStore()
        self.instruments=self.store.read('instruments').drop_duplicates('ts_code').set_index('ts_code')
        basics=[pd.read_parquet(p) for p in sorted((ROOT/'state/four-strategy-2026-jan-sep-v1/official').glob('daily_basic-*.parquet'))]
        self.store.cache['daily_basic']=pd.concat([self.store.base.read('daily_basic'),*basics],ignore_index=True).drop_duplicates(['ts_code','trade_date'],keep='last')
        bars=pd.read_parquet(ROOT/'state/mature-multifactor/p1-native-20261103-a/bars-input.parquet')
        self.prices=bars.assign(price=bars.close*bars.adj_factor).pivot(index='trade_date',columns='ts_code',values='price')
        self.prices.index=pd.to_datetime(self.prices.index)
        candidate=json.loads((ROOT/'state/factor-baseline-candidates/20261029.json').read_bytes())
        self.policy=candidate['factor_policy']
        weights=Path(candidate.get('publication',{}).get('source_weights',ROOT/'state/baseline-v2-release/corrected-refits/20261029T092148390912Z/state/baseline-v2-validation/weight-validation.json'))
        report=json.loads(weights.read_bytes())
        assert report['protocol']['as_of']=='20251231'
        self.coefficients=report['diagnostic_coefficients_not_released']
        assert self.coefficients==self.policy['experimental_linear_model']['coefficients']
        self.build_policy={k:v for k,v in self.policy.items() if k not in ['experimental_linear_model','input_overlay','validated_linear_model']}
        self.selection=yaml.safe_load((ROOT/'configs/sector_models.yaml').read_text(encoding='utf-8'))['global_selection']
        self.shared_cache=OUT/'baseline-score-cache-v1'
        self.cache=self.shared_cache if bps==10 else OUT/'baseline-score-cache-v1-bps20'
        self.cache.mkdir(exist_ok=True)

    def __call__(self,kind,signal,day,members,held,weights,halted):
        self.store.day=signal
        frozen=set(held)&halted
        def hold(reason):
            keep=(set(held)&members)|frozen
            return weights.reindex(sorted(keep)).dropna().to_dict(),keep,reason
        path=self.cache/f'{kind}-{signal}.parquet'
        shared=self.shared_cache/path.name
        if self.cache!=self.shared_cache and not path.exists() and shared.exists():
            try:cached=pd.read_parquet(shared)
            except (OSError,ValueError):pass
            else:cached.to_parquet(path,index=False)
        if path.exists():scores=pd.read_parquet(path)
        elif kind=='baseline':
            scores=build_global_factor_snapshot(self.store,sorted(members),self.selection['factors'])
            scores.to_parquet(path,index=False)
        elif kind=='v2':
            universe=self.instruments.reindex(sorted(members)).copy().reset_index(drop=True)
            universe['symbol']=sorted(members);universe['effective_date']=signal
            universe['board']=['STAR' if s.startswith('68') else 'CHINEXT' if s.startswith('30') else 'MAIN' for s in universe.symbol]
            try:
                with patch('moneymore.research.baseline_v2_model.active_strategy_universe',return_value=universe):
                    _,variants,_,_=build_v2_snapshot(ROOT,self.store,signal,policy=self.build_policy)
                scores=variants['industry_size_comparable'].copy()
                scores['score']=scores[list(self.coefficients)].to_numpy()@np.array(list(self.coefficients.values()))
                scores.to_parquet(path,index=False)
            except ValueError as error:
                if str(error) not in {'EMPTY_OR_DUPLICATE_V2_CROSS_SECTION','INSUFFICIENT_V2_CROSS_SECTION'}:raise
                return hold('V2_INPUT_HOLD:'+str(error))
        else:raise ValueError('UNKNOWN_BASELINE_KIND')
        if (set(held)&members)-set(scores.symbol):return hold('MEMBER_SCORE_MISSING')
        if kind=='baseline':
            s=self.selection
            _,targets,_=global_topk_portfolio(scores,set(held),symbol_column='symbol',top_k=s['top_k'],exit_rank=s['exit_rank'],max_replacements=s['max_replacements'],minimum_weight=s['minimum_weight'],maximum_weight=s['maximum_weight'],correlation=trailing_return_correlation(self.prices,pd.Timestamp(signal),s['correlation_lookback']),correlation_penalty=s['correlation_penalty'],cluster_correlation_threshold=s['cluster_correlation_threshold'],maximum_cluster_members=s['maximum_cluster_members'])
        else:
            _,targets,_=construct_daily_targets(scores,held,self.prices,signal,day,self.policy,universe_members=members)
        if targets is None:return hold('CONSTRUCTOR_HOLD')
        # Actual untradeability cannot consume cash; retain locked holdings and
        # scale only the freely allocated sleeve before prior-close share sizing.
        locked=weights.reindex(sorted(frozen)).dropna().to_dict()
        free={s:float(w) for s,w in targets.items() if s not in frozen}
        budget=max(0.,1-sum(locked.values()));total=sum(free.values())
        if total>budget:free={s:w*budget/total for s,w in free.items()}
        return {**free,**locked},frozen,None


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bps',type=int,choices=[10,20],default=10);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    engine.OUT=OUT/('baseline-smoke-verified' if args.smoke else 'baseline-replays');engine.KINDS=['baseline','v2']
    engine.OUT.mkdir(parents=True,exist_ok=True)
    with FileLock(str(engine.OUT/f'bps{args.bps}.lock'),timeout=7200),threadpool_limits(limits=2):
        engine.run(args.bps,smoke=args.smoke,target_builder=BaselineConstructors(args.bps))
