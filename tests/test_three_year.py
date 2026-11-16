import numpy as np
import pandas as pd
import pytest

from moneymore.research.three_year import (
    attach_financial_events,
    causal_volatility_scale,
    financial_events,
    prepared_reports,
    rolling_split,
    sequence_indices,
)


def test_sequences_use_exchange_dates_and_never_cross_symbols_or_gaps():
    frame = pd.DataFrame(
        {"date": ["20230102", "20230103", "20230105", "20230103"], "symbol": ["A", "A", "A", "B"]}
    )
    got = sequence_indices(frame, ["20230102", "20230103", "20230104", "20230105"], 3)
    assert got.tolist() == [[0, 0, 1], [0, 1, 2], [2, 0, 3], [0, 0, 4]]


def test_sequence_prefix_invariant_to_future_features():
    first = pd.DataFrame({"date": ["20230102", "20230103"], "symbol": ["A", "A"]})
    later = pd.concat(
        [first, pd.DataFrame({"date": ["20230104"], "symbol": ["A"]})], ignore_index=True
    )
    cal = ["20230102", "20230103", "20230104"]
    np.testing.assert_array_equal(
        sequence_indices(first, cal, 2), sequence_indices(later, cal, 2)[:2]
    )


def test_reports_respect_later_actual_announcement_plus_one_day():
    f = pd.DataFrame(
        {
            "ts_code": ["A"],
            "ann_date": ["20230401"],
            "f_ann_date": ["20230410"],
            "end_date": ["20221231"],
            "report_type": [1],
        }
    )
    assert prepared_reports(f).available.iloc[0] == pd.Timestamp("20230411")


def test_financial_events_correction_does_not_rewrite_earlier_history():
    basic = {
        "ts_code": ["A", "A"],
        "ann_date": ["20230301", "20230501"],
        "end_date": ["20221231", "20221231"],
    }
    fi = pd.DataFrame(
        {**basic, "fcff": [100.0, 200.0], "netdebt": [10.0, 20.0], "q_roe": [1.0, 2.0]}
    )
    cf = pd.DataFrame({**basic, "n_cashflow_act": [80.0, 160.0]})
    inc = pd.DataFrame({**basic, "operate_profit": [40.0, 80.0]})
    events = financial_events(cf, fi, inc)
    panel = pd.DataFrame(
        {"symbol": ["A"] * 3, "date": ["20230301", "20230302", "20230502"], "total_mv": [1.0] * 3}
    )
    result = attach_financial_events(panel, events)
    assert np.isnan(result.fcff_to_ev.iloc[0])
    assert result.fcff_to_ev.iloc[1] == pytest.approx(100 / 10010)
    assert result.fcff_to_ev.iloc[2] == pytest.approx(200 / 10020)
    assert result.cash_conversion.iloc[1] == 2
    assert np.isnan(result.quarter_roe_std.iloc[1])


def test_stale_financial_values_are_not_carried_forever():
    event = pd.DataFrame(
        {
            "symbol": ["A"],
            "available": [pd.Timestamp("20230302")],
            "report_end": [pd.Timestamp("20221231")],
            "fcff": [100.0],
            "netdebt": [10.0],
            "operating_cash": [80.0],
            "cash_conversion": [2.0],
            "quarter_roe_std": [1.0],
        }
    )
    panel = pd.DataFrame({"symbol": ["A"], "date": ["20231001"], "total_mv": [1.0]})
    got = attach_financial_events(panel, event)
    assert got.fcff_to_ev.isna().all()
    assert got.cash_conversion.isna().all()


def test_rolling_split_purges_both_label_boundaries():
    days = pd.bdate_range("2021-01-01", "2024-01-31").strftime("%Y%m%d").tolist()
    frame = pd.DataFrame({"date": days, "label_end_20": [*(days[20:]), *([None] * 20)]})
    masks = rolling_split(frame, "2023Q4", 60, 252)
    train, valid, test = [frame.loc[masks[k]] for k in ["train", "valid", "test"]]
    assert len(train) == 252
    assert train.label_end_20.max() < valid.date.min()
    assert valid.label_end_20.max() < test.date.min()


def test_volatility_only_uses_completed_month():
    dates = pd.bdate_range("2022-01-01", "2024-01-31")
    r = pd.Series(np.sin(np.arange(len(dates))) * 0.01, index=dates)
    cal = dates.strftime("%Y%m%d").tolist()
    a, c = causal_volatility_scale(r, ["20231115", "20231130"], cal, "20220101", "20230929")
    changed = r.copy()
    changed.loc["2023-11-16":] = 0.7 * np.sin(np.arange(len(changed.loc["2023-11-16":])))
    b, d = causal_volatility_scale(changed, ["20231115"], cal, "20220101", "20230929")
    assert c == d
    assert a["20231115"] == b["20231115"]
    assert all(0 < x <= 1 for x in a.values())


@pytest.mark.parametrize("states", [1, 3])
def test_native_tra_trains_and_inference_ignores_test_labels(states):
    import torch
    from qlib.contrib.model.pytorch_tra import TRAModel

    from moneymore.research.three_year_tra import (
        CausalSequenceBatches,
        predict_without_labels,
        routing_diagnostics,
    )

    torch.set_num_threads(2)
    dates = pd.bdate_range("20230102", periods=4).strftime("%Y%m%d").tolist()
    frame = pd.DataFrame(
        [(d, s) for d in dates for s in ["A", "B", "C", "D"]], columns=["date", "symbol"]
    )
    rng = np.random.default_rng(42)
    x = np.concatenate(
        [np.zeros((1, 5), dtype="float32"), rng.normal(size=(len(frame), 5)).astype("float32")]
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batches = CausalSequenceBatches(
        x,
        sequence_indices(frame, dates, 3),
        frame,
        np.ones(len(frame), dtype=bool),
        rng.normal(size=len(frame)),
        8,
        states,
        device,
    )
    model = TRAModel(
        model_config={"input_size": 5, "hidden_size": 8, "num_layers": 1},
        tra_config={"num_states": states, "src_info": "LR"},
        model_type="RNN",
        transport_method="router" if states > 1 else "none",
        alpha=1.0,
        seed=42,
        eval_train=False,
        eval_test=False,
    )
    assert np.isfinite(model.train_epoch(0, batches))
    a = predict_without_labels(model, batches)
    batches.labels[:] = np.nan
    b = predict_without_labels(model, batches)
    np.testing.assert_array_equal(a, b)
    diag = routing_diagnostics(model, batches)
    assert sum(diag["expert_counts"]) == len(frame)
    assert 1 <= diag["active_experts"] <= states
