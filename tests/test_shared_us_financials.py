import pandas as pd

from moneymore.data.store import ParquetStore
from moneymore.data.us_financials import (
    EXPANSION_TAGS,
    FACTS,
    SPLITS,
    annual_expansion_as_of,
    observed_splits_as_of,
    reported_facts_as_of,
)


def test_disclosure_cutoff_and_revisions(tmp_path):
    store = ParquetStore(tmp_path)
    tag = EXPANSION_TAGS[0]
    rows = pd.DataFrame({
        "cik": ["0001045810"] * 4,
        "taxonomy": ["us-gaap"] * 4,
        "tag": [tag] * 4,
        "unit": ["USD"] * 4,
        "period_precision": ["day"] * 4,
        "duration_months": [12] * 4,
        "period_start": pd.to_datetime(["2023-01-01"] * 2 + ["2024-01-01"] * 2),
        "period_end": pd.to_datetime(["2023-12-31"] * 2 + ["2024-12-31"] * 2),
        "public_date": pd.to_datetime(["2024-02-02", "2025-02-20", "2025-02-05", "2025-02-20"]),
        "accepted": pd.to_datetime(["2024-02-01", "2025-02-20", "2025-02-04", "2025-02-20"]),
        "accession": ["a", "c", "b", "c"],
        "industry": ["Chips"] * 4,
        "form": ["10-K", "10-K/A", "10-K", "10-K/A"],
        "source": ["test"] * 4,
        "value": [100, 125, 140, 150],
    })
    store.merge_curated(FACTS, [rows], ["accession", "period_end"])
    assert reported_facts_as_of(store, "2025-02-05", tags=[tag]).value.tolist() == [100]
    assert annual_expansion_as_of(store, "2025-02-21", ciks=[1045810]).growth.item() == 0.2
    assert annual_expansion_as_of(store, "2025-02-20", industries=["Chips"]).growth.item() == 0.4
    conflict = rows.iloc[[-1]].assign(accession="d", value=151)
    store.merge_curated(FACTS, [conflict], ["accession", "period_end"])
    selected = reported_facts_as_of(store, "2025-02-21", tags=[tag])
    assert selected.period_end.tolist() == [pd.Timestamp("2023-12-31")]


def test_splits_require_verified_identity_and_effective_date(tmp_path):
    store = ParquetStore(tmp_path)
    splits = pd.DataFrame({
        "cik": ["0001045810", "0001045810"],
        "known_date": pd.to_datetime(["2024-05-22", "2024-05-22"]),
        "ex_date": pd.to_datetime(["2024-06-10", "2024-06-10"]),
        "identity_verified": [True, False],
        "source": ["verified", "ticker-only"],
    })
    store.merge_curated(SPLITS, [splits], ["source"])
    assert observed_splits_as_of(store, "2024-06-07", ciks=[1045810]).empty
    assert observed_splits_as_of(store, "2024-06-10", ciks=[1045810]).source.tolist() == ["verified"]
