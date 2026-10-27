"""Chronological, date-balanced signal calibration; never writes trading state."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rank_signal(scores: pd.Series) -> pd.Series:
    """Centered cross-sectional percentile; invariant to model score units."""
    return scores.rank(method="average", pct=True) - scores.rank(method="average", pct=True).mean()


def walk_forward_calibration(observations, calendar, *, minimum_train_dates=20, embargo=1):
    """Each training label must mature strictly before the embargo boundary.

    OLS through the origin on date-demeaned returns, one equal weight per date.
    Negative historical slopes are disclosed, NOT inverted into a trading edge.
    Daily units are an arithmetic horizon conversion, not daily forecast validation.
    """
    days = sorted(set(calendar))
    locations = {date: i for i, date in enumerate(days)}
    rows = sorted(observations, key=lambda row: row["date"])
    output = []
    for test in rows:
        boundary = locations[test["date"]] - embargo
        train = [r for r in rows if r["date"] < test["date"]
                 and r.get("prospective") and r.get("valid_label")
                 and locations[r["maturity"]] < boundary]
        item = {"date": test["date"], "train_dates": len(train),
                "latest_train_maturity": max((r["maturity"] for r in train), default=None),
                "source_sha256": test.get("source_sha256"),
                "status": "INSUFFICIENT_MATURE_HISTORY", "orders_enabled": False}
        if not test.get("prospective"):
            item["status"] = "NON_PROSPECTIVE_PREDICTION"
        elif len(train) >= minimum_train_dates:
            # Averaging daily sufficient statistics prevents larger pools from
            # counting as more independent observations than smaller pools.
            numerator = np.mean([np.mean(r["x"] * r["y"]) for r in train])
            denominator = np.mean([np.mean(r["x"] ** 2) for r in train])
            slope = float(numerator / denominator) if denominator > 1e-12 else 0.0
            horizon = test["horizon"]
            item.update(status="DIAGNOSTIC_ONLY" if slope > 0 else "NO_POSITIVE_CALIBRATED_EDGE",
                        horizon_slope=slope, daily_slope=slope / horizon,
                        training_sources=[r.get("source_sha256") for r in train])
            if test.get("valid_label"):
                prediction = slope * test["x"]
                item.update(out_of_sample_mse=float(np.mean((test["y"] - prediction) ** 2)),
                            zero_forecast_mse=float(np.mean(test["y"] ** 2)),
                            rank_ic=float(pd.Series(test["x"]).corr(pd.Series(test["y"]), method="spearman")))
        output.append(item)
    return output


def build_observations(reports, adjusted_prices, calendar, *, as_of, horizon=5):
    days = sorted(set(calendar))
    observations = []
    for report in reports:
        date = str(report["trade_date"])
        if date not in days or days.index(date) + 1 >= len(days):
            continue
        i = days.index(date)
        scores = pd.DataFrame(report["scores"])
        if not {"instrument", "score", "datetime"}.issubset(scores) or scores.instrument.duplicated().any():
            continue
        if set(scores.datetime.astype(str).str.replace("-", "", regex=False)) != {date}:
            continue
        signal = pd.to_numeric(scores.set_index("instrument").score, errors="coerce")
        if not np.isfinite(signal).all() or signal.nunique() < 2:
            continue
        captured = pd.to_datetime(report.get("created_at"), utc=True, errors="coerce")
        start = pd.Timestamp(date + " 15:00", tz="Asia/Shanghai").tz_convert("UTC")
        deadline = pd.Timestamp(days[i + 1] + " 09:30", tz="Asia/Shanghai").tz_convert("UTC")
        maturity = days[i + horizon] if i + horizon < len(days) else None
        row = dict(date=date, maturity=maturity, horizon=horizon,
                   prospective=bool(not pd.isna(captured) and start <= captured < deadline),
                   valid_label=False, source_sha256=report.get("source_sha256"))
        row["missing_label_symbols"] = []
        if maturity and maturity <= as_of and date in adjusted_prices.index and maturity in adjusted_prices.index:
            prices = adjusted_prices.reindex(columns=signal.index)
            first, last = prices.loc[date], prices.loc[maturity]
            valid = np.isfinite(first) & np.isfinite(last) & first.gt(0) & last.gt(0)
            row["missing_label_symbols"] = list(signal.index[~valid])
            # Require the complete recorded pool: no silent survivor selection.
            if valid.all():
                y = last / first - 1
                row.update(valid_label=True, x=rank_signal(signal).to_numpy(),
                           y=(y - y.mean()).to_numpy())
        observations.append(row)
    return observations
