"""Bounded, local-only signal diagnostics; NOT realized-return attribution.

This script never fetches prices, calls a broker, changes model selection, or
writes outside a new --output directory under the isolated research root.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "state/ashare-mainline-20261112"
START, CUTOFF = "20260105", "20260930"
CASE = "doubleensemble_full"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                    encoding="utf-8")


def finite(values):
    return np.isfinite(pd.to_numeric(values, errors="coerce"))


def correlation(x, y, rank=False, minimum=5):
    paired = pd.concat([x.rename("x"), y.rename("y")], axis=1)
    paired = paired.loc[finite(paired.x) & finite(paired.y)]
    if len(paired) < minimum or paired.x.nunique() < 2 or paired.y.nunique() < 2:
        return np.nan
    if rank:
        paired = paired.rank(method="average")
    return float(paired.x.corr(paired.y))


def block_summary(values, block=20, samples=2000):
    """Circular calendar-position blocks, preserving missing dates in each draw."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return {"mean": None, "ci95_block20": [None, None], "valid_days": 0}
    rng = np.random.default_rng(42)
    starts = rng.integers(len(values), size=(samples, (len(values) + block - 1) // block))
    indices = (starts[..., None] + np.arange(block)) % len(values)
    draws = values[indices.reshape(samples, -1)[:, :len(values)]]
    counts = np.isfinite(draws).sum(axis=1)
    means = np.nansum(draws, axis=1)[counts > 0] / counts[counts > 0]
    return {"mean": float(values[valid].mean()),
            "ci95_block20": np.quantile(means, [0.025, 0.975]).tolist(),
            "valid_days": int(valid.sum()), "calendar_positions": len(values),
            "positive_day_fraction": float((values[valid] > 0).mean())}


def matured(frame):
    """Mask before any label calculation, including sector benchmark formation."""
    result = frame.copy()
    end = result.label_end_20.fillna("")
    result["label_is_mature"] = (
        result.label_end_20.notna() & end.le(CUTOFF) & end.gt(result.date)
    )
    result["mature_label"] = result.label_20.where(
        result.label_is_mature & finite(result.label_20)
    )
    return result


def sector_benchmarks(panel):
    rows = []
    for (day, industry), group in panel.loc[panel.industry.notna()].groupby(
        ["date", "industry"], sort=True
    ):
        available = group.loc[finite(group.mature_label)]
        complete = len(available) == len(group)
        cap_ok = finite(available.total_mv) & available.total_mv.gt(0)
        ew = available.mature_label.mean() if len(available) else np.nan
        cw = (np.average(available.mature_label, weights=available.total_mv)
              if len(available) and cap_ok.all() else np.nan)
        rows.append({"date": day, "industry": industry, "pool_assets": len(group),
                     "mature_label_assets": len(available),
                     "label_coverage": len(available) / len(group),
                     "ew_label_available": ew, "ew_label_complete": ew if complete else np.nan,
                     "cap_label_available": cw,
                     "cap_label_complete": cw if complete else np.nan,
                     "status": "COMPLETE" if complete else "INCOMPLETE_LABELS",
                     "last_label_end_used": (str(available.label_end_20.max())
                                             if len(available) else None)})
    return pd.DataFrame(rows)


def signal_diagnostics(predictions, panel):
    benchmark = sector_benchmarks(panel)
    p = predictions.drop(columns="industry", errors="ignore").merge(
        panel[["date", "symbol", "industry", "total_mv"]],
        on=["date", "symbol"], how="left", validate="one_to_one", indicator=True,
    )
    if not p._merge.eq("both").all():
        raise ValueError("PREDICTION_OUTSIDE_PANEL")
    p = matured(p.drop(columns="_merge")).merge(
        benchmark[["date", "industry", "ew_label_available", "ew_label_complete"]],
        on=["date", "industry"], how="left", validate="many_to_one",
    )
    p["sector_relative_label_available"] = p.mature_label - p.ew_label_available
    p["sector_relative_label_complete"] = p.mature_label - p.ew_label_complete
    rows, sector_rows, top_rows = [], [], []
    for day, group in p.groupby("date", sort=True):
        scored = group.loc[finite(group.score)].copy()
        paired = scored.loc[finite(scored.mature_label)].copy()
        known = paired.loc[paired.industry.notna()].copy()
        rank_score, rank_label = known.score.rank(), known.mature_label.rank()
        score_residual = rank_score - rank_score.groupby(known.industry).transform("mean")
        label_residual = rank_label - rank_label.groupby(known.industry).transform("mean")
        counts = known.groupby("industry").score.transform("size")
        sector_score_rank = (known.groupby("industry").score.rank() - 0.5) / counts
        sector_label_rank = (known.groupby("industry").mature_label.rank() - 0.5) / counts
        # Pick BEFORE looking at labels, including missing or not-yet-mature labels.
        selected = scored.sort_values(["score", "symbol"], ascending=[False, True]).head(50)
        selected = selected.copy()
        selected["selection_rank"] = np.arange(1, len(selected) + 1)
        top_rows.append(selected[["date", "symbol", "industry", "score", "selection_rank",
                                  "label_end_20", "label_is_mature", "mature_label",
                                  "ew_label_available", "ew_label_complete",
                                  "sector_relative_label_available",
                                  "sector_relative_label_complete"]])
        row = {"date": day, "score_assets": len(scored), "paired_assets": len(paired),
               "paired_known_industry_assets": len(known),
               "immature_or_unknown_end_assets": int((~scored.label_is_mature).sum()),
               "missing_mature_label_assets": int((scored.label_is_mature
                                                    & ~finite(scored.mature_label)).sum()),
               "last_label_end_used": str(paired.label_end_20.max()) if len(paired) else None,
               "raw_full_rank_ic": correlation(paired.score, paired.mature_label,
                                               rank=True, minimum=100),
               "raw_known_industry_rank_ic": correlation(known.score, known.mature_label,
                                                         rank=True, minimum=100),
               "industry_demeaned_global_rank_ic": correlation(score_residual, label_residual,
                                                               minimum=100),
               "sector_percentile_rank_ic": correlation(sector_score_rank, sector_label_rank,
                                                        minimum=100),
               "raw_score_sector_relative_label_rank_ic": correlation(
                   known.score, known.sector_relative_label_available, rank=True, minimum=100),
               "top50_selected_assets": len(selected)}
        for column in ["mature_label", "ew_label_available", "ew_label_complete",
                       "sector_relative_label_available", "sector_relative_label_complete"]:
            valid = finite(selected[column])
            row["top50_" + column + "_assets"] = int(valid.sum())
            row["top50_" + column + "_mean"] = (
                float(selected[column].mean()) if len(selected) == 50 and valid.all() else np.nan
            )
        comparable = (len(selected) == 50
                      and finite(selected.sector_relative_label_available).all())
        row["top50_raw_label_on_sector_comparable_days_mean"] = (
            float(selected.mature_label.mean()) if comparable else np.nan
        )
        row["top50_sector_benchmark_on_comparable_days_mean"] = (
            float(selected.ew_label_available.mean()) if comparable else np.nan
        )
        rows.append(row)
        for industry, sector in known.groupby("industry"):
            sector_rows.append({"date": day, "industry": industry, "paired_assets": len(sector),
                                "rank_ic": correlation(sector.score, sector.mature_label, rank=True),
                                "mean_score": float(sector.score.mean()),
                                "mean_label": float(sector.mature_label.mean())})
    return pd.DataFrame(rows), benchmark, pd.DataFrame(sector_rows), pd.concat(top_rows)


def theoretical_benchmarks(panel):
    """Same previous-close pool as replay, but theoretical full-investment returns."""
    from research_ashare_paper_routes import load_inputs

    from moneymore.research.enhanced_indexing import carry_confirmed_suspensions

    loaded = load_inputs()  # Local reader only; do not call engine.run/ensure_quotes.
    bars, membership, sessions, signals, _, _, _, official, _ = loaded
    native = ROOT / "state/mature-multifactor/p1-native-20261103-a"
    calendar = pd.DatetimeIndex(pd.to_datetime(
        (native / "provider/calendars/day.txt").read_text().splitlines()
    )).strftime("%Y%m%d").tolist()
    expected = [d for d in calendar if START <= d <= CUTOFF]
    if [d for d in sessions if START <= d <= CUTOFF] != expected:
        raise ValueError("BENCHMARK_CALENDAR_MISMATCH")
    if any(signals[d] != calendar[calendar.index(d) - 1] for d in expected):
        raise ValueError("BENCHMARK_PREVIOUS_DAY_MISMATCH")
    bars = bars.loc[bars.trade_date.le(CUTOFF)].copy()
    bars["adjusted"] = (bars.close * bars.adj_factor).where(
        finite(bars.close) & bars.close.gt(0) & finite(bars.adj_factor) & bars.adj_factor.gt(0)
    )
    prices = bars.pivot(index="trade_date", columns="ts_code", values="adjusted")
    prices = prices.reindex([d for d in calendar if d <= CUTOFF])
    proofs = [item["suspend_d"] for day, item in official.items()
              if day <= CUTOFF and not item["suspend_d"].empty]
    proof = pd.concat(proofs, ignore_index=True).drop_duplicates() if proofs else pd.DataFrame()
    if not proof.empty:
        proof["trade_date"] = proof.trade_date.astype(str)
        proof = proof.loc[proof.trade_date.le(CUTOFF)]
    prices, carries = carry_confirmed_suspensions(prices, proof)
    attributes = {d: g.set_index("symbol") for d, g in panel.groupby("date")}
    rows, rejected = [], []
    for day in expected:
        signal = signals[day]
        members = sorted(membership.get(signal, set()))
        if len(members) != 1000:
            raise ValueError("BENCHMARK_INCOMPLETE_MEMBERSHIP:" + signal)
        previous = prices.loc[signal].reindex(members)
        current = prices.loc[day].reindex(members)
        returns = current / previous - 1
        valid = finite(returns) & finite(previous) & previous.gt(0) & current.gt(0)
        caps = attributes.get(signal, pd.DataFrame(columns=["total_mv"])).total_mv.reindex(members)
        cap_ok = finite(caps) & caps.gt(0)
        for symbol in returns.index[~valid]:
            rejected.append({"date": day, "signal": signal, "symbol": symbol,
                             "reason": "UNKNOWN_ADJUSTED_CLOSE_RETURN"})
        for symbol in caps.index[~cap_ok]:
            rejected.append({"date": day, "signal": signal, "symbol": symbol,
                             "reason": "UNKNOWN_PREVIOUS_DAY_POSITIVE_TOTAL_MV"})
        ew = float(returns.mean()) if valid.all() else np.nan
        cw = float(np.average(returns, weights=caps)) if valid.all() and cap_ok.all() else np.nan
        rows.append({"date": day, "weight_date": signal, "pool_assets": len(members),
                     "known_return_assets": int(valid.sum()), "known_cap_assets": int(cap_ok.sum()),
                     "equal_weight_return": ew, "cap_weight_return": cw,
                     "status": "COMPLETE" if valid.all() and cap_ok.all() else "REJECTED_MISSING_INPUT",
                     "carried_current_prices": sum(r["trade_date"] == day and r["symbol"] in members
                                                   for r in carries)})
    daily = pd.DataFrame(rows)
    # skipna=False prevents stitching an unknown gap into a seemingly complete NAV.
    for kind in ["equal_weight", "cap_weight"]:
        daily[kind + "_nav"] = (1 + daily[kind + "_return"]).cumprod(skipna=False)
    summary = {"expected_sessions": len(daily), "rejected_sessions": int(daily.status.ne("COMPLETE").sum()),
               "official_halt_carries": len(carries), "same_pool_as_replay": True,
               "not_a_tradable_benchmark": True, "not_actual_portfolio_alpha": True}
    for kind in ["equal_weight", "cap_weight"]:
        complete = finite(daily[kind + "_return"]).all()
        summary[kind] = {"valid_days": int(finite(daily[kind + "_return"]).sum()),
                         "full_period_return": float(daily[kind + "_nav"].iloc[-1] - 1)
                         if complete else None}
    paths = [native / "bars-input.parquet", native / "membership.parquet",
             native / "provider/calendars/day.txt"]
    paths += sorted((ROOT / "state/cost-aware-ab-v2/quote-cache").glob("*.parquet"))
    official_path = ROOT / "state/four-strategy-2026-jan-sep-v1/official"
    paths += sorted(official_path.glob("held-exit-*.parquet"))
    paths += [official_path / f"suspend_d-{d}.parquet"
              for d in sorted(official) if d <= CUTOFF]
    summary["input_hashes"] = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    return daily, pd.DataFrame(rejected, columns=["date", "signal", "symbol", "reason"]), carries, summary


def main(output, skip_daily_benchmark=False):
    output = output.resolve()
    if not output.is_relative_to(BASE.resolve()) or output == BASE.resolve():
        raise ValueError("OUTPUT_MUST_BE_AN_ISOLATED_RESEARCH_SUBDIRECTORY")
    if output.exists():
        raise FileExistsError("IMMUTABLE_OUTPUT:" + str(output))
    prediction_path, panel_path = BASE / "n2-summary-v1/predictions.parquet", BASE / "panel-v2/panel.parquet"
    prediction_hash, panel_hash = digest(prediction_path), digest(panel_path)
    manifest = json.loads((BASE / "n2-summary-v1/results.json").read_bytes())
    if prediction_hash != manifest["hashes"]["predictions.parquet"]:
        raise ValueError("FROZEN_N2_PREDICTIONS_CHANGED")
    training = json.loads((BASE / "registered-matrix-v1/runtime.json").read_bytes())
    if panel_hash != training["panel_sha256"]:
        raise ValueError("FROZEN_N1_PANEL_CHANGED")
    predictions = pd.read_parquet(prediction_path, filters=[("case", "==", CASE),
                                                          ("date", ">=", START),
                                                          ("date", "<=", CUTOFF)])
    panel = pd.read_parquet(panel_path, columns=["date", "symbol", "industry", "total_mv",
                                                "label_20", "label_end_20"],
                            filters=[("date", ">=", "20251231"), ("date", "<=", CUTOFF)])
    for name, frame in [("predictions", predictions), ("panel", panel)]:
        if frame.empty or frame.duplicated(["date", "symbol"]).any():
            raise ValueError("EMPTY_OR_DUPLICATED_INPUT:" + name)
    check = predictions.merge(panel[["date", "symbol", "label_20", "label_end_20"]],
                              on=["date", "symbol"], suffixes=("_prediction", "_panel"),
                              how="left", validate="one_to_one", indicator=True)
    if not check._merge.eq("both").all() or not np.allclose(
        check.label_20_prediction, check.label_20_panel, equal_nan=True
    ) or not check.label_end_20_prediction.fillna("").eq(check.label_end_20_panel.fillna("")).all():
        raise ValueError("PREDICTION_PANEL_LABEL_MISMATCH")
    daily, sectors, within, top50 = signal_diagnostics(
        predictions, matured(panel.loc[panel.date.ge(START)])
    )
    metrics = ["raw_full_rank_ic", "raw_known_industry_rank_ic", "industry_demeaned_global_rank_ic",
               "sector_percentile_rank_ic", "raw_score_sector_relative_label_rank_ic",
               "top50_mature_label_mean", "top50_ew_label_available_mean",
               "top50_raw_label_on_sector_comparable_days_mean",
               "top50_sector_benchmark_on_comparable_days_mean",
               "top50_sector_relative_label_available_mean", "top50_sector_relative_label_complete_mean"]
    # Do not dilute estimates with trailing wholly immature signal dates. Missing
    # observations inside the mature window remain in bootstrap calendar positions.
    mature_dates = daily.loc[daily.paired_assets.gt(0), "date"]
    mature_daily = daily.loc[daily.date.le(mature_dates.max())] if len(mature_dates) else daily.iloc[:0]
    summaries = {name: block_summary(mature_daily[name]) for name in metrics}
    neutral = summaries["industry_demeaned_global_rank_ic"]
    if neutral["mean"] is None:
        classification = "INSUFFICIENT_MATURE_WITHIN_INDUSTRY_DATA"
    elif neutral["ci95_block20"][0] > 0:
        classification = "POSITIVE_WITHIN_INDUSTRY_RANK_ASSOCIATION_IN_THIS_SAMPLE"
    elif neutral["mean"] > 0:
        classification = "POSITIVE_POINT_ESTIMATE_WITHIN_INDUSTRY_BUT_UNCERTAIN"
    else:
        classification = "NO_POSITIVE_MEAN_WITHIN_INDUSTRY_RANK_ASSOCIATION"
    benchmark_daily = benchmark_rejections = None
    carries, benchmark_report = [], {"status": "SKIPPED_BY_CLI"}
    if not skip_daily_benchmark:
        benchmark_daily, benchmark_rejections, carries, benchmark_report = theoretical_benchmarks(panel)
    data_issues = {
        "immature_or_unknown_end_prediction_rows": int(daily.immature_or_unknown_end_assets.sum()),
        "mature_label_missing_prediction_rows": int(daily.missing_mature_label_assets.sum()),
        "signal_panel_unknown_industry_rows": int(panel.loc[panel.date.ge(START)].industry.isna().sum()),
        "sector_date_groups_incomplete": int(sectors.status.ne("COMPLETE").sum()),
        "note": "Groups after the last mature signal date are also classified incomplete; no future labels used.",
        "benchmark_rejections": [],
    }
    if benchmark_rejections is not None:
        for reason, group in benchmark_rejections.groupby("reason"):
            data_issues["benchmark_rejections"].append({
                "reason": reason, "asset_days": len(group), "symbols": group.symbol.nunique(),
                "days": group.date.nunique(),
                "largest_symbol_counts": {str(k): int(v) for k, v in
                                          group.groupby("symbol").size().sort_values(ascending=False).head(10).items()},
            })
    # Do all checks/computation before claiming the immutable output directory.
    output.mkdir(parents=True, exist_ok=False)
    daily.to_parquet(output / "daily-signal-diagnostics.parquet", index=False)
    sectors.to_parquet(output / "daily-sector-label-benchmarks.parquet", index=False)
    within.to_parquet(output / "daily-within-sector-rank-ic.parquet", index=False)
    top50.to_parquet(output / "daily-top50-label-diagnostics.parquet", index=False)
    if benchmark_daily is not None:
        benchmark_daily.to_parquet(output / "theoretical-top1000-daily-benchmarks.parquet", index=False)
        benchmark_rejections.to_parquet(output / "benchmark-rejected-inputs.parquet", index=False)
        save(output / "benchmark-halt-carries.json", carries)
    result = {
        "status": "SIGNAL_DIAGNOSTIC_COMPLETE_NOT_ACTUAL_RETURN_ATTRIBUTION",
        "case": CASE, "window": [START, CUTOFF], "label_maturity_cutoff_inclusive": CUTOFF,
        "prediction_rows": len(predictions), "prediction_days": predictions.date.nunique(),
        "last_mature_signal_date": str(mature_dates.max()) if len(mature_dates) else None,
        "latest_label_end_used": top50.loc[top50.mature_label.notna(), "label_end_20"].max(),
        "missing_industry_panel_rows": int(panel.industry.isna().sum()),
        "mature_label_missing_prediction_rows": int(daily.missing_mature_label_assets.sum()),
        "classification": classification, "summaries": summaries,
        "data_issues": data_issues,
        "quarterly_summaries": {
            str(quarter): {name: block_summary(group[name]) for name in metrics}
            for quarter, group in mature_daily.groupby(pd.to_datetime(mature_daily.date).dt.to_period("Q"))
        },
        "theoretical_benchmark": benchmark_report,
        "methods": {
            "label_20": "Existing frozen next-session adjusted open to horizon adjusted close label; not close-close daily PnL.",
            "raw_full_rank_ic": "Daily Spearman score vs mature label, finite paired prediction rows, minimum 100.",
            "industry_demeaned_global_rank_ic": "Pearson of global score/label ranks after each is demeaned within industry; known-industry paired rows only.",
            "sector_percentile_rank_ic": "Pearson of (within-industry average rank - 0.5)/industry count for scores and labels; pooled paired rows.",
            "sector_label_benchmark": "Same-day full panel pool; equal-weight mature labels within sector. Available-label and strict complete-sector means are separate.",
            "top50": "Highest finite scores before inspecting outcome availability; ties resolved by symbol. Mean remains unknown unless all 50 required outcomes exist.",
            "daily_benchmark": "Previous-session same-pool membership and total_mv weights; adjusted close-close returns. All 1000 required, unknown inputs reject whole day. Only officially proven full-session halts may carry adjusted prices.",
            "uncertainty": "Descriptive circular 20-session block bootstrap, 2000 draws, seed 42; missing dates retained. Not a confirmatory test or correction for model exploration.",
            "classification": "Based only on mean and 95% block CI of industry-demeaned global RankIC; does not select a portfolio variant.",
        },
        "limitations": [
            "Signal-label comparisons are not actual holding/fill/cash/turnover return attribution, tradable benchmark replication, or true alpha.",
            "Overlapping 20-session labels are not independent daily realized returns and must not be compounded or compared directly with replay NAV.",
            "Available-label sector means exclude unknown outcomes and can be biased; coverage and strict complete-sector alternatives are exported. No unknown return is zero-filled.",
            "Industry control is not control for size, momentum, board, all risk factors or causal identification. Singleton industries contribute no within-sector rank variation.",
            "Top1000 membership uses existing historical carried intervals, not original-vintage daily Top1000 reconstitution.",
            "Raw top50, sector baseline and sector-relative unconditional means can cover different days. Use the explicitly comparable-day columns for a matched subtraction.",
            "Total market value is not free-float capitalization. The daily theoretical benchmark ignores costs, lots, T+1, tradability and action settlement timing.",
            "Adjusted prices are vendor-adjusted proxies; documented halt carrying assumes unchanged adjusted economic marks. No new market-data verification is performed.",
            "No production account, model, scheduler, security pool or execution configuration is changed; no new data is fetched.",
            "No winner is chosen among the fixed five-case/two-cost study using these diagnostics.",
        ],
        "real_orders_enabled": False, "production_modified": False, "new_data_fetched": False,
        "input_hashes": {str(prediction_path.relative_to(ROOT)): prediction_hash,
                         str(panel_path.relative_to(ROOT)): panel_hash,
                         str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))},
        "artifact_hashes": {p.name: digest(p) for p in output.iterdir() if p.is_file()},
    }
    save(output / "results.json", result)
    print(json.dumps({"output": str(output), "classification": classification,
                      "summaries": summaries, "benchmark": {k: v for k, v in benchmark_report.items()
                                                              if k != "input_hashes"}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New immutable directory under state/ashare-mainline-20261112")
    parser.add_argument("--skip-daily-benchmark", action="store_true")
    args = parser.parse_args()
    main(args.output, args.skip_daily_benchmark)
