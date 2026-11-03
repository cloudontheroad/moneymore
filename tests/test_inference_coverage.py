import numpy as np
import pandas as pd
from moneymore.research.inference_coverage import inference_eligibility


def test_future_labels_do_not_affect_inference_eligibility():
    data=pd.DataFrame({'known':[1.,2.,np.nan],'future_label':[.1,np.nan,.3]})
    before=inference_eligibility(data,['known'])
    data['future_label']=[np.nan,1e20,np.nan]
    pd.testing.assert_series_equal(before,inference_eligibility(data,['known']))
    assert before.tolist()==[True,True,False]
