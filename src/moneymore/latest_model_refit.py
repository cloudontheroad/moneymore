"""Fixed-round deployment fitting, separate from OOS evaluation artifacts."""
from __future__ import annotations


def fit_gru_fixed_rounds(model, frame, *, epochs):
    """Qlib GRU.fit without validation restores initial weights; don't use it here."""
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("positive research-selected epoch count required")
    clean = frame.dropna()
    if len(clean) < model.batch_size:
        raise ValueError("insufficient complete rows for a GRU batch")
    for _ in range(epochs):
        model.train_epoch(clean["feature"], clean["label"])
    model.fitted = True
    return {"epochs": epochs, "samples": len(clean), "has_oos_score": False}
