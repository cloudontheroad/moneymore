"""Inference eligibility uses only signal-time feature availability."""
import numpy as np


def inference_eligibility(frame, required):
    if not frame.index.is_unique or len(required)!=len(set(required)) or set(required)-set(frame.columns):
        raise ValueError('INVALID_INFERENCE_FEATURE_SCHEMA_OR_KEYS')
    return np.isfinite(frame[list(required)]).all(axis=1)
