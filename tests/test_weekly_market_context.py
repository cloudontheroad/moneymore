import pandas as pd

from moneymore.weekly_market_context import market_context
from moneymore.latest_model_refit import fit_gru_fixed_rounds


def test_context_is_causal_and_does_not_fill_missing():
    dates = pd.bdate_range("2026-01-01", periods=70).strftime("%Y%m%d").tolist()
    bars = pd.DataFrame([{"ts_code": s, "trade_date": d, "close": 10+i}
                         for i, d in enumerate(dates) for s in ("A", "B")
                         if not (s == "B" and i == 65)])
    summary, series = market_context(bars, dates, {"A": "bank", "B": "tech"}, as_of=dates[65])
    assert summary["observed"] == 1
    assert summary["missing_symbols"] == ["B"]
    assert series.index.max() == dates[65]
    assert summary["macro"]["used_in_training"] is False


def test_fixed_refit_does_not_restore_initial_weights():
    class Fake:
        batch_size = 1
        calls = 0
        def train_epoch(self, x, y):
            self.calls += 1
    model = Fake()
    frame = pd.DataFrame({("feature", "x"): [1.], ("label", "y"): [2.]})
    result = fit_gru_fixed_rounds(model, frame, epochs=3)
    assert model.calls == 3
    assert model.fitted
    assert result["has_oos_score"] is False
