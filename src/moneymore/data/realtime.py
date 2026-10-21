"""Single selection point for realtime data, independent of research fundamentals."""

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=8)
def _provider(source: str, root: str):
    if source == "alphafeed":
        from .alphafeed_provider import AlphaFeedProvider

        return AlphaFeedProvider(root=root)
    if source == "qmt":
        from .qmt_provider import QmtTushareProvider

        return QmtTushareProvider(root=root, timeout_seconds=15)
    raise ValueError("Unsupported MONEYMORE_REALTIME_DATA_SOURCE")


def create_realtime_provider(root=None):
    source = os.getenv("MONEYMORE_REALTIME_DATA_SOURCE", "qmt").strip().lower()
    resolved = str(Path(root or Path(__file__).resolve().parents[3]).resolve())
    return _provider(source, resolved)
