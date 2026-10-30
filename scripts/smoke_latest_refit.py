"""Tiny synthetic compatibility check, never investment evidence."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from qlib.contrib.model.pytorch_gru import GRU

from moneymore.latest_model_refit import fit_gru_fixed_rounds
from moneymore.qlib_challenger import QlibPanelDataset


def main():
    rng = np.random.default_rng(17)
    index = pd.MultiIndex.from_product([pd.to_datetime(["2026-10-22"]), [f"S{i}" for i in range(32)]], names=["datetime", "instrument"])
    frame = pd.DataFrame(rng.normal(size=(32, 721)).astype("float32"), index=index,
        columns=pd.MultiIndex.from_tuples([("feature", f"x{i}") for i in range(720)] + [("label", "LABEL0")]))
    model = GRU(d_feat=12, hidden_size=64, num_layers=2, dropout=.1,
        batch_size=8, seed=17, GPU=0 if torch.cuda.is_available() else -1)
    before = [p.detach().cpu().clone() for p in model.gru_model.parameters()]
    result = fit_gru_fixed_rounds(model, frame, epochs=2)
    changed = any(not torch.equal(a, b.detach().cpu()) for a, b in zip(before, model.gru_model.parameters()))
    predictions = model.predict(QlibPanelDataset(frame, {"forward": ("2026-10-22", "2026-10-22")}), "forward")
    assert changed, "GRU weights did not change"
    assert len(predictions) == 32 and np.isfinite(predictions).all()
    result.update(status="SYNTHETIC_COMPATIBILITY_PASS", weights_changed=changed,
        predictions=len(predictions), cuda=torch.cuda.is_available(), investment_evidence=False)
    out = Path(__file__).resolve().parents[1] / "state/weekly-learning/20261030-refit-smoke.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    # Exit while the local CUDA model still exists; returning first can
    # trigger its native destructor before the outer teardown guard.
    if os.name == "nt":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
    # Same Windows native teardown issue as the research worker. Assertions and
    # report persistence have completed; do not run unstable CUDA DLL teardown.
    if os.name == "nt":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
