"""Run Qlib's native DDG-DA workflow on the prepared US Alpha300 handler."""

from __future__ import annotations

from pathlib import Path

import qlib
from qlib.contrib.rolling.ddgda import DDGDA
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.meta.data_selection.dataset import InternalData
import qlib.contrib.meta.data_selection.dataset as meta_dataset_module
from joblib import Parallel as JoblibParallel
from functools import partial
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "state/us-overnight-open2open-v1/ddgda"


def main() -> int:
    # Qlib expresses disabled early stopping as None; LightGBM >=4.6 requires
    # an integer and uses 0 for the same meaning.
    original_fit = LGBModel.fit

    def compatible_fit(self, *args, **kwargs):
        if self.early_stopping_rounds is None:
            self.early_stopping_rounds = 0
        return original_fit(self, *args, **kwargs)

    LGBModel.fit = compatible_fit

    def compatible_calc_perf(self, pred, label):
        frame = pd.DataFrame({"pred": pred, "label": label})
        correlations = frame.groupby("datetime", group_keys=True).corr(method="spearman")
        return correlations.xs("pred", level=-1)["label"]

    InternalData._calc_perf = compatible_calc_perf
    meta_dataset_module.Parallel = partial(JoblibParallel, prefer="threads")
    qlib.init(
        provider_uri=str(WORK / "provider"),
        region="us",
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {
                "uri": "sqlite:///" + str((WORK / "mlflow_ddgda_v3.db").resolve()).replace("\\", "/"),
                "default_exp_name": "Experiment",
            },
        },
    )
    runner = DDGDA(
        conf_path=WORK / "workflow_alpha300.yaml",
        h_path=str((WORK / "alpha300_handler.pkl").resolve()),
        working_dir=WORK,
        exp_name="us_open2open_ddgda",
        rolling_exp="us_open2open_ddgda_rolling",
        horizon=1,
        step=20,
        train_start="2023-09-08",
        # Last complete 20-session DDG-DA block.  Nov 3-4 would otherwise form
        # an official RollingGen tail shorter than MetaTaskDS's five-day floor.
        test_end="2026-11-02",
        meta_1st_train_end="2024-12-31",
        segments="2026-05-07",
        hist_step_n=5,
        fea_imp_n=30,
    )
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
