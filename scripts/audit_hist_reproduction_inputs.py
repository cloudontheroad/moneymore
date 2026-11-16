"""Post-preparation input audit, separate from prediction/performance selection."""
import json

import numpy as np
import pandas as pd
from prepare_hist_reproduction import CFG, OUT, digest, write_json


def audit():
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    calendar = pd.to_datetime((OUT / "provider/calendars/day.txt").read_text().splitlines())
    official = pd.to_datetime(pd.read_parquet(OUT / "benchmarks/CSI300_price.parquet").trade_date)
    test_calendar = calendar[(calendar >= CFG["test"][0]) & (calendar <= CFG["test"][1])]
    report = {"provider_test_days": len(test_calendar),
              "missing_vs_official_calendar": [str(d.date()) for d in official if d.year >= 2017 and d not in test_calendar],
              "VWAP_files": len(list((OUT / "provider/features").glob("*/vwap.day.bin"))),
              "paper_six_feature_input_certified": False,
              "decision": "Run public-source reproduction with author's zero-fill; explicitly not paper-data identity", "splits": {}}
    for key in ("train", "valid", "test"):
        x = np.load(OUT / key / "features.npy", mmap_mode="r")
        index = pd.read_parquet(OUT / key / "index.parquet")
        assert x.shape[1] == 360 and len(x) == len(index)
        assert not index.duplicated().any()
        fields = ["close", "open", "high", "low", "vwap", "volume"]
        channels = {name: float(np.abs(x[:, i*60:(i+1)*60]).max()) for i, name in enumerate(fields)}
        assert all(channels[name] > 0 for name in fields if name != "vwap")
        report["splits"][key] = {"normalized_channel_abs_max": channels,
            "vwap_all_zero": channels["vwap"] == 0, "finite": bool(np.isfinite(x).all())}
    manifest["input_audit"] = report
    write_json(OUT / "manifest.json", manifest)
    write_json(OUT / "prepared.json", {"manifest_sha256": digest(OUT / "manifest.json"), "status": "complete_with_declared_public_data_limits"})
    write_json(OUT / "input-audit.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    audit()
