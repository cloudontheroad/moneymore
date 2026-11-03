"""Exact binary-availability attribution, not an economic return attribution."""
import numpy as np
import pandas as pd


def missing_group_drift(start, end, weights, end_weights, benchmark, end_benchmark):
    for group in (start, end):
        if not group.index.is_unique or not group.isin([0., 1.]).all():
            raise ValueError("MISSING_GROUP_MUST_BE_OBSERVED_BINARY")
    support = weights.index.union(end_weights.index)
    if not support.isin(start.index.intersection(end.index)).all():
        raise ValueError("HELD_GROUP_UNOBSERVABLE")
    for vector, universe in ((weights, support), (end_weights, support),
                             (benchmark, start.index), (end_benchmark, end.index)):
        if not vector.index.is_unique or not vector.index.isin(universe).all():
            raise ValueError("INVALID_WEIGHT_UNIVERSE")
        if not np.isfinite(vector).all() or (vector < 0).any() or abs(vector.sum()-1)>1e-8:
            raise ValueError("INVALID_NORMALIZED_WEIGHTS")
    w0=weights.reindex(support,fill_value=0.);w1=end_weights.reindex(support,fill_value=0.)
    common=start.index.intersection(end.index)
    old_only=start.index.difference(end.index);new_only=end.index.difference(start.index)
    b0=benchmark.reindex(start.index,fill_value=0.);b1=end_benchmark.reindex(end.index,fill_value=0.)
    components={
        "portfolio_price_drift":float((w1-w0) @ start.reindex(support)),
        "portfolio_label_migration":float(w1 @ (end.reindex(support)-start.reindex(support))),
        "benchmark_weight_drift":float(-(b1.reindex(common)-b0.reindex(common)) @ start.reindex(common)),
        "benchmark_label_migration":float(-b1.reindex(common) @ (end.reindex(common)-start.reindex(common))),
        "benchmark_entry":float(-b1.reindex(new_only) @ end.reindex(new_only)),
        "benchmark_exit":float(b0.reindex(old_only) @ start.reindex(old_only)),
    }
    error=float(w1 @ end.reindex(support)-b1 @ end-w0 @ start.reindex(support)+b0 @ start)
    residual=error-sum(components.values())
    if abs(residual)>1e-10:raise ValueError("MISSING_GROUP_ATTRIBUTION_DOES_NOT_RECONCILE")
    return {**components,"error":error,"identity_residual":residual,
            "held_became_unknown":int(((start.reindex(support)==0)&(end.reindex(support)==1)).sum()),
            "held_became_observed":int(((start.reindex(support)==1)&(end.reindex(support)==0)).sum())}
