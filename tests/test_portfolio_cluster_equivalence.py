import numpy as np
import pandas as pd

from moneymore import portfolio_constructor as pc


def reference(symbols, correlation, threshold):
    if threshold is None:
        return {s: s for s in symbols}
    parent = {s: s for s in symbols}

    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    available = [s for s in symbols if s in correlation.index]
    for i, left in enumerate(available):
        for right in available[i + 1 :]:
            value = correlation.loc[left, right]
            if pd.notna(value) and float(value) >= threshold:
                a, b = find(left), find(right)
                if a != b:
                    parent[b] = a
    return {s: find(s) for s in symbols}


def test_vectorized_clusters_and_portfolios_equal_scalar_reference(monkeypatch):
    rng = np.random.default_rng(1729)
    symbols = [f"S{i:02}" for i in range(35)]
    scores = pd.DataFrame({"symbol": symbols, "score": rng.normal(size=35)})
    optimized = pc._correlation_clusters
    for threshold in [0.2, 0.65, 0.99]:
        values = rng.uniform(-1, 1, (35, 35))
        values[rng.uniform(size=(35, 35)) < 0.1] = np.nan
        corr = pd.DataFrame(values, index=symbols, columns=symbols)
        assert optimized(symbols, corr, threshold) == reference(symbols, corr, threshold)
        kwargs = {
            "symbol_column": "symbol",
            "top_k": 10,
            "exit_rank": 15,
            "max_replacements": 3,
            "correlation": corr,
            "cluster_correlation_threshold": threshold,
            "maximum_cluster_members": 3,
        }
        fast = pc.global_topk_portfolio(scores, set(symbols[10:20]), **kwargs)
        with monkeypatch.context() as context:
            context.setattr(pc, "_correlation_clusters", reference)
            slow = pc.global_topk_portfolio(scores, set(symbols[10:20]), **kwargs)
        assert fast[:2] == slow[:2]
        pd.testing.assert_frame_equal(fast[2], slow[2])


def test_nullable_and_absent_symbols_preserve_reference_components():
    frame = pd.DataFrame(
        {"A": pd.Series([1.0, pd.NA], dtype="Float64"), "B": pd.Series([0.9, 1.0], dtype="Float64")}
    )
    frame.index = ["A", "B"]
    symbols = ["A", "B", "MISSING"]
    assert pc._correlation_clusters(symbols, frame, 0.65) == reference(symbols, frame, 0.65)


def test_diversified_exit_rank_prefix_covers_all_possible_new_additions():
    rng = np.random.default_rng(1919)
    symbols = [f"S{i:02}" for i in range(40)]
    scores = pd.DataFrame({"symbol": symbols, "score": rng.normal(size=40)})
    values = rng.uniform(-1, 1, (40, 40))
    correlation = pd.DataFrame(values, index=symbols, columns=symbols)
    kwargs = {
        "symbol_column": "symbol",
        "top_k": 10,
        "exit_rank": 15,
        "max_replacements": 3,
        "correlation": correlation,
        "cluster_correlation_threshold": 0.4,
        "maximum_cluster_members": 3,
    }
    _, _, ranking = pc.global_topk_portfolio(scores, set(), **kwargs)
    support = set(ranking.head(10).symbol)
    for count in [0, 3, 7, 10, 15]:
        for _ in range(5):
            held = set(rng.choice(symbols, size=count, replace=False)) | {"OUTSIDE_POOL"}
            selected, _, _ = pc.global_topk_portfolio(scores, held, **kwargs)
            assert set(selected) - held <= support
