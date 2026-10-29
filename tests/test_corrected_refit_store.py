import importlib.util
import sys
from pathlib import Path

import pandas as pd

from moneymore.research.security_identity import SecurityIdentity

scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("corrected_refit", scripts / "refit_verified_overlay.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.path.remove(str(scripts))


def test_factor_view_uses_only_overlay_and_never_fills():
    class Store:
        curated = Path("unused")

        def read(self, *args, **kwargs):
            raise AssertionError("must not fall back to uncorrected factors")

    overlay = pd.DataFrame({
        "ts_code": ["A", "A"], "trade_date": ["20240101", "20240102"],
        "derived_adj_factor": [1.0, float("nan")],
    })
    adapter = module.CorrectedStore(Store(), overlay, SecurityIdentity([]))
    frame = adapter.read("adj_factor", filters=[("trade_date", ">=", "20240102")])
    assert len(frame) == 1
    assert frame.adj_factor.isna().all()
    assert "derived_adj_factor" in overlay


def test_daily_alias_query_expands_then_filters_stable_identity():
    class Store:
        curated = Path("unused")

        def read(self, table, columns=None, filters=None):
            assert set(filters[0][2]) == {"OLD", "NEW"}
            return pd.DataFrame({"ts_code": ["NEW"], "trade_date": ["20250218"], "close": [5.0]})

    identity = SecurityIdentity([{"old": "OLD", "new": "NEW", "effective": "20250217", "source": "test"}])
    adapter = module.CorrectedStore(Store(), pd.DataFrame(), identity)
    result = adapter.read("daily", columns=["ts_code", "trade_date", "close"], filters=[("ts_code", "in", ["OLD"])])
    assert result.ts_code.tolist() == ["OLD"]
    assert result.close.tolist() == [5.0]
