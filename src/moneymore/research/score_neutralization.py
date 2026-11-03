"""Contemporaneous cross-sectional OLS audit, no future labels accepted."""
import numpy as np
import pandas as pd


def neutralize_scores(scores, controls):
    if not scores.index.is_unique or not controls.index.is_unique or not scores.index.equals(controls.index):
        raise ValueError('NEUTRALIZATION_IDENTITY_MISMATCH')
    if not np.isfinite(scores).all() or not np.isfinite(controls).all().all():raise ValueError('NONFINITE_NEUTRALIZATION_INPUT')
    scale=controls.std(ddof=0).replace(0,1.)
    z=(controls-controls.mean())/scale
    x=np.column_stack([np.ones(len(scores)),z.to_numpy()])
    coefficient,_,rank,_=np.linalg.lstsq(x,scores.to_numpy(),rcond=None)
    if len(scores)-rank<2:raise ValueError('NO_RESIDUAL_DEGREES_OF_FREEDOM')
    fitted=x @ coefficient;residual=scores.to_numpy()-fitted
    if np.linalg.norm(residual)<=1e-12*max(1.,np.linalg.norm(scores)):residual[:]=0.
    error=float(np.max(np.abs(x.T @ residual))/len(scores))
    if error>1e-9*max(1.,float(np.std(scores))):raise ValueError('SCORE_RESIDUAL_NOT_ORTHOGONAL')
    return pd.Series(residual,index=scores.index,name='neutral_score'),{'matrix_rank':int(rank),'observations':len(scores),
        'maximum_normalized_orthogonality_error':error,'future_labels_used':False}
