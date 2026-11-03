"""Thin Qlib data adapter for LightGBM's documented LambdaRank objective."""
import numpy as np
import pandas as pd
from qlib.contrib.model.gbdt import LGBModel


class RankingLGBModel(LGBModel):
    def _prepare_data(self,dataset,reweighter=None):
        prepared=super()._prepare_data(dataset,reweighter)
        for data,segment in prepared:
            frame=dataset.prepare(segment,col_set=['feature','label'])
            dates=frame.index.get_level_values('datetime')
            if not dates.is_monotonic_increasing:raise ValueError('RANKING_REQUIRES_CONTIGUOUS_DATE_GROUPS')
            groups=frame.groupby(level='datetime',sort=False).size().to_numpy()
            relevance=np.minimum(19,np.floor(frame['label'].iloc[:,0].groupby(dates).rank(pct=True).to_numpy()*20)).astype(int)
            data.set_group(groups);data.set_label(relevance)
        return prepared
