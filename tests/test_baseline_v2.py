from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from moneymore.research.baseline_v2 import consistent_scores


def sample():
    return pd.DataFrame(
        {
            "date": [pd.Timestamp("20261027")] * 12,
            "symbol": [str(i) for i in range(12)],
            "roe": np.arange(12, dtype=float),
            "book_to_price": np.array([3, 8, 1, 5, 2, 7, 9, 6, 4, 11, 10, 12], dtype=float),
            "total_mv": np.exp(np.linspace(2, 5, 12)),
        }
    )


def test_fixed_weights_missing_is_excluded_not_reweighted():
    raw = sample()
    raw.loc[0, "roe"] = np.nan
    raw.loc[1, "book_to_price"] = np.inf
    scores, excluded = consistent_scores(raw, {"roe": 0.7, "book_to_price": 0.3})
    assert len(scores) == 10
    assert set(excluded.symbol) == {"0", "1"}
    assert scores.active_weight.eq(1).all()
    assert scores.effective_weight__roe.eq(0.7).all()
    assert np.allclose(scores.score, scores.roe * 0.7 + scores.book_to_price * 0.3)
    assert pd.isna(raw.loc[0, "roe"])


def test_size_flag_requires_real_control_and_adequate_cross_section():
    with pytest.raises(ValueError, match="MISSING_SIZE_CONTROL"):
        consistent_scores(sample().drop(columns="total_mv"), {"roe": 1}, neutralize_size=True)
    with pytest.raises(ValueError, match="INSUFFICIENT_COMPLETE"):
        consistent_scores(sample().head(4), {"roe": 1}, neutralize_size=True)
    data = sample()
    data["total_mv"] = 1
    with pytest.raises(ValueError, match="DEGENERATE"):
        consistent_scores(data, {"roe": 1}, neutralize_size=True)


def test_size_control_reaches_preprocessing_and_changes_scores(monkeypatch):
    import moneymore.research.baseline_v2 as module
    from moneymore.factors import preprocess_cross_section

    observed = []

    def capture(frame, names, config):
        observed.append((config.neutralize_size, "total_mv" in frame, config.industry_column))
        return preprocess_cross_section(frame, names, config)

    monkeypatch.setattr(module, "preprocess_cross_section", capture)
    plain, _ = consistent_scores(sample(), {"roe": 0.5, "book_to_price": 0.5})
    controlled, _ = consistent_scores(
        sample(), {"roe": 0.5, "book_to_price": 0.5}, neutralize_size=True
    )
    assert observed == [(False, True, None), (True, True, None)]
    assert not np.allclose(plain.score, controlled.score)


@pytest.mark.parametrize("weights", [{}, {"roe": 0}, {"roe": -1}, {"roe": np.nan}])
def test_invalid_weights_fail_closed(weights):
    with pytest.raises(ValueError, match="INVALID_FACTOR_WEIGHTS"):
        consistent_scores(sample(), weights)


def test_duplicate_observations_rejected():
    with pytest.raises(ValueError, match="DUPLICATE"):
        consistent_scores(pd.concat([sample(), sample()]), {"roe": 1})


def test_runner_is_isolated_pit_and_keeps_both_ablations_on_same_pool(tmp_path, monkeypatch):
    import hashlib

    import yaml

    from moneymore.data.store import ParquetStore
    from moneymore.research.baseline_v2 import run_baseline_v2_research

    raw = sample()
    symbols = raw.symbol.tolist()
    dates = pd.bdate_range(end="20261027", periods=253).strftime("%Y%m%d").tolist()
    store = ParquetStore(tmp_path / "data")
    financial = pd.DataFrame(
        [
            {
                "ts_code": s,
                "ann_date": "20260820",
                "end_date": "20260630",
                "roe": 12.0,
                "q_sales_yoy": 10.0,
            }
            for s in symbols
        ]
    )
    tables = {
        "fina_indicator_research_v2": financial,
        "fina_indicator": financial,
        "daily_basic": pd.DataFrame(
            [
                {"ts_code": s, "trade_date": d, "total_mv": mv, "pe_ttm": 10.0, "dv_ttm": 2.0}
                for s, mv in zip(symbols, raw.total_mv)
                for d in ("20261026", "20261028")
            ]
        ),
        "daily": pd.DataFrame(
            [
                {"ts_code": s, "trade_date": d, "close": 10.0}
                for s in symbols
                for d in dates
                if not (s == "0" and d == dates[-10])
            ]
        ),
        "adj_factor": pd.DataFrame(
            [{"ts_code": s, "trade_date": d, "adj_factor": 1.0} for s in symbols for d in dates]
        ),
        "trade_calendar": pd.DataFrame({"exchange": "SSE", "is_open": 1, "cal_date": dates}),
    }
    store.curated.mkdir(parents=True)
    for name, frame in tables.items():
        frame.to_parquet(store.curated / (name + ".parquet"), index=False)
    before = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in store.curated.glob("*.parquet")
    }
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "sector_models.yaml").write_text(
        yaml.safe_dump({"global_selection": {"factors": {"roe": 0.7, "book_to_price": 0.3}}})
    )

    def snapshot(adapter, requested, weights, audit_capture=None):
        assert adapter.read("daily_basic").trade_date.max() == "20261026"
        if audit_capture is not None:
            audit_capture["raw_factors"] = raw.copy()
        return raw.assign(score=0.0, active_weight=1.0)

    monkeypatch.setattr("moneymore.research.sector_model.build_global_factor_snapshot", snapshot)
    first = run_baseline_v2_research(tmp_path, store, symbols, "20261027")
    second = run_baseline_v2_research(tmp_path, store, symbols, "20261027")
    assert first["scored"] == 11
    assert first["history_exclusions"] == ["0"]
    assert not first["orders_enabled"]
    assert first["evidence_path"] != second["evidence_path"]
    plain = pd.read_parquet(Path(first["evidence_path"]) / "repaired_fixed_weights.parquet")
    size = pd.read_parquet(Path(first["evidence_path"]) / "repaired_size_control.parquet")
    assert plain.symbol.tolist() == size.symbol.tolist()
    assert before == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in store.curated.glob("*.parquet")
    }
