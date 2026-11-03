import numpy as np
import pandas as pd
from moneymore.research.prediction_quality import prediction_quality,daily_rank_ic_evaluator


def test_topk_tie_is_not_disguised_as_fine_ranking():
    result=prediction_quality([1.]*122+[0.]*878)
    assert not result['passed'] and result['top_score_count']==122
    assert prediction_quality(np.arange(1000))['passed']
    assert prediction_quality([2.]*25+[1.]*25+[0.]*950)['passed']
    assert not prediction_quality([float('inf'),*range(1000)])['passed']


def test_daily_rankic_agrees_with_spearman_and_constant_is_zero():
    class Dataset:
        def prepare(self,segment):
            size=3 if segment=='train' else 4
            return pd.DataFrame(index=pd.MultiIndex.from_product([pd.date_range('2020-01-01',periods=2),range(size)],names=['datetime','instrument']))
    class Labels:
        def get_label(self):return np.array([1,2,3,4,1,2,3,4])
    evaluate=daily_rank_ic_evaluator(Dataset())
    labels=Labels()
    name,value,higher=evaluate(np.array([1,2,3,4,4,3,2,1]),labels)
    assert name=='daily_rank_ic' and higher and abs(value)<1e-12
    assert evaluate(np.ones(8),labels)[1]==0.


def test_daily_review_cannot_silently_release_a_model_quality_hold():
    from moneymore.candidate_lifecycle import review_candidate
    previous={'status':'REVIEW_HOLD','model_quality_hold':True,'reason':'boundary tie'}
    assert review_candidate({}, {},as_of='20261104',trustworthy_start=None,policy={},previous=previous)==previous
