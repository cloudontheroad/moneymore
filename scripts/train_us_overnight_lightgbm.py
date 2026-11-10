"""Rolling, point-in-time LightGBM ranker for the US overnight paper strategy."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from moneymore.us_overnight.config import USOvernightConfig
from moneymore.us_overnight.features import ALL_FEATURE_COLUMNS, FEATURE_COLUMNS

ACTIVE_FEATURE_COLUMNS = list(FEATURE_COLUMNS)
ACTIVE_OBJECTIVE = "huber"
ACTIVE_TARGET_COLUMN = "label_next_open_net"

ROOT = Path(__file__).resolve().parents[1]
ROLLING_TRAIN_SESSIONS = 504
VALIDATION_SESSIONS = 126
RETRAIN_EVERY = 5
MIN_NET_BPS = 10.0
MIN_WIN_RATE = 0.55
MIN_SHARPE = 1.0


def _model(
    num_leaves: int,
    min_child_samples: int = 300,
    colsample_bytree: float = 0.75,
    reg_lambda: float = 3.0,
    reg_alpha: float = 0.25,
    min_split_gain: float = 0.0,
    max_depth: int = -1,
    learning_rate: float = 0.025,
    n_estimators: int = 350,
    huber_alpha: float = 0.9,
    subsample: float = 0.8,
    subsample_freq: int = 0,
    min_child_weight: float = 50.0,
    path_smooth: float = 0.0,
    feature_fraction_bynode: float = 1.0,
    extra_trees: bool = False,
    max_bin: int = 255,
    min_data_in_bin: int = 3,
    boosting_type: str = "gbdt",
    data_sample_strategy: str = "bagging",
    drop_rate: float = 0.1,
    skip_drop: float = 0.5,
    max_drop: int = 50,
    top_rate: float = 0.2,
    other_rate: float = 0.1,
    seed: int = 42,
) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective=ACTIVE_OBJECTIVE,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        min_split_gain=min_split_gain,
        subsample=subsample,
        subsample_freq=subsample_freq,
        min_child_weight=min_child_weight,
        path_smooth=path_smooth,
        feature_fraction_bynode=feature_fraction_bynode,
        extra_trees=extra_trees,
        max_bin=max_bin,
        min_data_in_bin=min_data_in_bin,
        boosting_type=boosting_type,
        data_sample_strategy=data_sample_strategy,
        drop_rate=drop_rate,
        skip_drop=skip_drop,
        max_drop=max_drop,
        top_rate=top_rate,
        other_rate=other_rate,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        alpha=huber_alpha,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def _weights(frame: pd.DataFrame, half_life: int) -> np.ndarray:
    """Emphasize recent sessions without allowing one good/bad day to dominate."""
    dates = pd.Series(sorted(frame["timestamp"].unique()))
    age = {date: len(dates) - 1 - index for index, date in enumerate(dates)}
    return frame["timestamp"].map(lambda value: 0.5 ** (age[value] / half_life)).to_numpy(float)


def _fit(
    frame: pd.DataFrame,
    target_transform: str,
    num_leaves: int,
    min_child_samples: int,
    colsample_bytree: float,
    reg_lambda: float,
    reg_alpha: float,
    min_split_gain: float,
    max_depth: int,
    learning_rate: float,
    n_estimators: int,
    huber_alpha: float,
    subsample: float,
    subsample_freq: int,
    half_life: int,
    min_child_weight: float,
    path_smooth: float,
    feature_fraction_bynode: float,
    extra_trees: bool,
    max_bin: int,
    min_data_in_bin: int,
    boosting_type: str,
    data_sample_strategy: str,
    drop_rate: float,
    skip_drop: float,
    max_drop: int,
    top_rate: float,
    other_rate: float,
    top1_weight: float,
    profit_early_stopping_rounds: int,
    target_clip_abs: float | None,
) -> lgb.LGBMRegressor:
    model = _model(
        num_leaves,
        min_child_samples,
        colsample_bytree,
        reg_lambda,
        reg_alpha,
        min_split_gain,
        max_depth,
        learning_rate,
        n_estimators,
        huber_alpha,
        subsample,
        subsample_freq,
        min_child_weight,
        path_smooth,
        feature_fraction_bynode,
        extra_trees,
        max_bin,
        min_data_in_bin,
        boosting_type,
        data_sample_strategy,
        drop_rate,
        skip_drop,
        max_drop,
        top_rate,
        other_rate,
    )
    grouped = frame.groupby("timestamp")[ACTIVE_TARGET_COLUMN]
    if target_transform == "daily_mad":
        median = grouped.transform("median")
        absolute = (frame["label_next_open_net"] - median).abs()
        mad = absolute.groupby(frame["timestamp"]).transform("median").replace(0, np.nan)
        target = ((frame["label_next_open_net"] - median) / mad).fillna(0)
    elif target_transform == "daily_rank":
        # Microsoft Qlib CSRankNorm: map each day's percentile rank to an
        # approximately unit-variance uniform score.
        target = (grouped.rank(pct=True, method="average") - 0.5) * 3.46
    elif target_transform == "daily_centered":
        target = frame[ACTIVE_TARGET_COLUMN] - grouped.transform("mean")
    elif target_transform == "daily_zscore":
        # Qlib-style CSZScoreNorm: prevent high-dispersion market sessions
        # from dominating a cross-sectional stock-selection objective.
        centered = frame[ACTIVE_TARGET_COLUMN] - grouped.transform("mean")
        dispersion = grouped.transform("std").replace(0, np.nan)
        target = (centered / dispersion).fillna(0).clip(-3, 3)
    else:
        target = frame[ACTIVE_TARGET_COLUMN]
    if target_clip_abs is not None:
        target = target.clip(-target_clip_abs, target_clip_abs)
    weights = _weights(frame, half_life)
    if top1_weight > 1.0:
        percentile = grouped.rank(pct=True)
        weights = weights * np.where(percentile >= 0.99, top1_weight, 1.0)
    if profit_early_stopping_rounds > 0 and boosting_type != "dart":
        ordered_dates = sorted(frame["timestamp"].unique())
        inner_dates = set(ordered_dates[-63:])
        fit_mask = ~frame["timestamp"].isin(inner_dates)
        valid_mask = ~fit_mask
        valid_groups = pd.factorize(frame.loc[valid_mask, "timestamp"], sort=False)[0]

        def top1_metric(y_true: np.ndarray, predictions: np.ndarray):
            chosen = []
            for group in np.unique(valid_groups):
                locations = np.flatnonzero(valid_groups == group)
                chosen.append(y_true[locations[np.argmax(predictions[locations])]])
            return "top1_relative_return", float(np.mean(chosen)), True

        model.set_params(n_estimators=max(2000, model.get_params()["n_estimators"]), metric="None")
        model.fit(
            frame.loc[fit_mask, ACTIVE_FEATURE_COLUMNS],
            target.loc[fit_mask],
            sample_weight=weights[fit_mask.to_numpy()],
            eval_set=[(frame.loc[valid_mask, ACTIVE_FEATURE_COLUMNS], target.loc[valid_mask])],
            eval_metric=top1_metric,
            callbacks=[
                lgb.early_stopping(
                    profit_early_stopping_rounds,
                    first_metric_only=True,
                    verbose=False,
                )
            ],
        )
        model.set_params(n_estimators=max(1, int(model.best_iteration_)), metric="None")
    model.fit(frame[ACTIVE_FEATURE_COLUMNS], target, sample_weight=weights)
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-transform",
        choices=["raw", "daily_centered", "daily_mad", "daily_rank", "daily_zscore"],
        default="raw",
    )
    parser.add_argument("--objective", choices=["regression", "huber"], default="huber")
    parser.add_argument("--output-prefix", default="lightgbm")
    parser.add_argument("--num-leaves", type=int, default=23)
    parser.add_argument("--min-child-samples", type=int, default=300)
    parser.add_argument("--colsample-bytree", type=float, default=0.75)
    parser.add_argument("--reg-lambda", type=float, default=3.0)
    parser.add_argument("--reg-alpha", type=float, default=0.25)
    parser.add_argument("--min-split-gain", type=float, default=0.0)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--n-estimators", type=int, default=350)
    parser.add_argument("--huber-alpha", type=float, default=0.9)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--subsample-freq", type=int, default=0)
    parser.add_argument("--weight-half-life", type=int, default=126)
    parser.add_argument("--rolling-train-sessions", type=int, default=ROLLING_TRAIN_SESSIONS)
    parser.add_argument("--validation-sessions", type=int, default=VALIDATION_SESSIONS)
    parser.add_argument("--retrain-every", type=int, default=RETRAIN_EVERY)
    parser.add_argument("--min-child-weight", type=float, default=50.0)
    parser.add_argument("--path-smooth", type=float, default=0.0)
    parser.add_argument("--feature-fraction-bynode", type=float, default=1.0)
    parser.add_argument("--extra-trees", action="store_true")
    parser.add_argument("--max-bin", type=int, default=255)
    parser.add_argument("--min-data-in-bin", type=int, default=3)
    parser.add_argument("--boosting-type", choices=["gbdt", "dart"], default="gbdt")
    parser.add_argument("--data-sample-strategy", choices=["bagging", "goss"], default="bagging")
    parser.add_argument("--drop-rate", type=float, default=0.1)
    parser.add_argument("--skip-drop", type=float, default=0.5)
    parser.add_argument("--max-drop", type=int, default=50)
    parser.add_argument("--top-rate", type=float, default=0.2)
    parser.add_argument("--other-rate", type=float, default=0.1)
    parser.add_argument("--top1-weight", type=float, default=1.0)
    parser.add_argument("--profit-early-stopping-rounds", type=int, default=0)
    parser.add_argument("--panel-path", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--exclude-label-abs-above", type=float)
    parser.add_argument("--feature-columns-json", type=Path)
    parser.add_argument("--training-target-column", default="label_next_open_net")
    parser.add_argument(
        "--exclude-cross-section-constant",
        action="store_true",
        help="Drop predictors with fewer than five distinct values on a typical session.",
    )
    parser.add_argument(
        "--target-clip-abs",
        type=float,
        help="Clip only the training target; validation returns remain untouched.",
    )
    parser.add_argument(
        "--exclude-feature",
        action="append",
        default=[],
        help="Feature to remove after all automatic feature screening; repeatable.",
    )
    args = parser.parse_args()
    global ACTIVE_FEATURE_COLUMNS, ACTIVE_OBJECTIVE, ACTIVE_TARGET_COLUMN
    ACTIVE_OBJECTIVE = args.objective
    ACTIVE_TARGET_COLUMN = args.training_target_column
    if args.feature_columns_json:
        ACTIVE_FEATURE_COLUMNS = json.loads(args.feature_columns_json.read_text(encoding="utf-8"))
    config = USOvernightConfig.from_yaml(ROOT / "configs/us_overnight_paper_only.yaml")
    output = ROOT / config.output_root
    cache_path = args.panel_path or (
        ROOT / "data/us_overnight_corrected/features/historical_panel_bundle_investable_120d_20m_5.parquet"
    )
    if cache_path.exists():
        panel = pd.read_parquet(cache_path)
    else:
        raise FileNotFoundError(f"Explicit historical training panel missing: {cache_path}")
    if args.exclude_cross_section_constant:
        daily_unique = (
            panel.groupby("timestamp", sort=False)[ACTIVE_FEATURE_COLUMNS]
            .nunique(dropna=True)
            .median()
        )
        ACTIVE_FEATURE_COLUMNS = daily_unique[daily_unique >= 5].index.tolist()
    if args.exclude_feature:
        excluded = set(args.exclude_feature)
        missing_exclusions = excluded - set(ACTIVE_FEATURE_COLUMNS)
        if missing_exclusions:
            raise ValueError(
                f"excluded features are not active: {sorted(missing_exclusions)}"
            )
        ACTIVE_FEATURE_COLUMNS = [
            column for column in ACTIVE_FEATURE_COLUMNS if column not in excluded
        ]
    if ACTIVE_TARGET_COLUMN not in panel:
        raise ValueError(f"training target missing from panel: {ACTIVE_TARGET_COLUMN}")
    mature = panel.dropna(subset=["label_next_open_net", ACTIVE_TARGET_COLUMN]).copy()
    if args.exclude_label_abs_above is not None:
        mature = mature.loc[
            mature["label_next_open_net"].abs() <= args.exclude_label_abs_above
        ].copy()
    if "label_end" not in mature and cache_path.name == "historical_panel_bundle_investable_120d_20m_5.parquet":
        ends = pd.read_parquet(
            cache_path.parent.parent / "reports/investability_eligibility_history.parquet",
            columns=["timestamp", "symbol", "label_end"],
        )
        mature = mature.merge(ends, on=["timestamp", "symbol"], how="left", validate="one_to_one")
        if mature.label_end.isna().any():
            raise ValueError("Missing label settlement sessions")
    dates = sorted(mature["timestamp"].unique())
    label_ends = mature.groupby("timestamp")["label_end"].max() if "label_end" in mature else None
    validation_dates = dates[-args.validation_sessions:]
    predictions: list[pd.DataFrame] = []
    model: lgb.LGBMRegressor | None = None
    active_anchor_position: int | None = None
    date_positions = {date: position for position, date in enumerate(dates)}
    for validation_date in validation_dates:
        position = date_positions[validation_date]
        # Anchor retraining to the full trading calendar, not to the beginning
        # of the requested validation slice.  This makes a date's prediction
        # invariant to validation_sessions and prevents window-start leakage
        # into model scheduling.
        anchor_position = position - (position % args.retrain_every)
        if model is None or anchor_position != active_anchor_position:
            cutoff = dates[anchor_position]
            if label_ends is not None:
                train_dates = set(label_ends[label_ends < cutoff].index[-args.rolling_train_sessions:])
            else:
                end = max(0, anchor_position - 1)
                train_dates = set(dates[max(0, end - args.rolling_train_sessions):end])
            model = _fit(
                mature.loc[mature["timestamp"].isin(train_dates)],
                args.target_transform,
                args.num_leaves,
                args.min_child_samples,
                args.colsample_bytree,
                args.reg_lambda,
                args.reg_alpha,
                args.min_split_gain,
                args.max_depth,
                args.learning_rate,
                args.n_estimators,
                args.huber_alpha,
                args.subsample,
                args.subsample_freq,
                args.weight_half_life,
                args.min_child_weight,
                args.path_smooth,
                args.feature_fraction_bynode,
                args.extra_trees,
                args.max_bin,
                args.min_data_in_bin,
                args.boosting_type,
                args.data_sample_strategy,
                args.drop_rate,
                args.skip_drop,
                args.max_drop,
                args.top_rate,
                args.other_rate,
                args.top1_weight,
                args.profit_early_stopping_rounds,
                args.target_clip_abs,
            )
            active_anchor_position = anchor_position
        day = mature.loc[mature["timestamp"] == validation_date, [
            "timestamp", "symbol", "label_next_open_net", *ACTIVE_FEATURE_COLUMNS
        ]].copy()
        day["score"] = model.predict(day[ACTIVE_FEATURE_COLUMNS])
        predictions.append(day[["timestamp", "symbol", "label_next_open_net", "score"]])
    validation = pd.concat(predictions, ignore_index=True)
    validation["rank"] = validation.groupby("timestamp")["score"].rank(
        method="first", ascending=False
    )
    selected = validation.loc[validation["rank"] <= config.top_n]
    daily_top = selected.groupby("timestamp")["label_next_open_net"].mean()
    daily_all = validation.groupby("timestamp")["label_next_open_net"].mean()
    slot_rows: list[dict[str, object]] = []
    for slot, slot_frame in selected.groupby("rank"):
        returns = slot_frame.sort_values("timestamp")["label_next_open_net"]
        equity = (1 + returns).cumprod()
        downside_slot = returns.loc[returns < 0].std(ddof=1)
        slot_rows.append(
            {
                "rank": int(slot),
                "sessions": len(returns),
                "mean_net_bps": float(returns.mean() * 10_000),
                "win_rate": float((returns > 0).mean()),
                "sharpe": float(returns.mean() / returns.std(ddof=1) * np.sqrt(252)),
                "sortino": (
                    float(returns.mean() / downside_slot * np.sqrt(252))
                    if downside_slot > 0
                    else 0.0
                ),
                "max_drawdown": float((equity / equity.cummax() - 1).min()),
                "cumulative_return": float(equity.iloc[-1] - 1),
            }
        )
    slot_metrics = pd.DataFrame(slot_rows).sort_values("rank")
    slot_means = slot_metrics["mean_net_bps"].to_numpy(float)
    adjacent_spreads = slot_means[:-1] - slot_means[1:]
    monotonic_violations = int((adjacent_spreads <= 0).sum())
    strictly_monotonic = monotonic_violations == 0
    net_bps = float(daily_top.mean() * 10_000)
    win_rate = float((daily_top > 0).mean())
    sharpe = float(daily_top.mean() / daily_top.std(ddof=1) * np.sqrt(252))
    cumulative = (1 + daily_top).cumprod()
    max_drawdown = float((cumulative / cumulative.cummax() - 1).min())
    downside = daily_top.loc[daily_top < 0].std(ddof=1)
    sortino = float(daily_top.mean() / downside * np.sqrt(252)) if downside > 0 else 0.0
    rolling_21 = daily_top.rolling(21).mean().dropna()
    ordered_validation_dates = list(daily_top.index)
    fold_rows: list[dict[str, object]] = []
    top1_daily = (
        selected.loc[selected["rank"] == 1]
        .set_index("timestamp")["label_next_open_net"]
        .reindex(ordered_validation_dates)
    )
    for fold_number, start in enumerate(range(0, len(ordered_validation_dates), 63), start=1):
        fold_dates = ordered_validation_dates[start : start + 63]
        if len(fold_dates) < 21:
            continue
        fold_top10 = daily_top.reindex(fold_dates)
        fold_top1 = top1_daily.reindex(fold_dates)
        fold_rows.append(
            {
                "fold": fold_number,
                "sessions": len(fold_dates),
                "start": str(fold_dates[0]),
                "end": str(fold_dates[-1]),
                "top1_mean_bps": float(fold_top1.mean() * 10_000),
                "top1_sharpe": float(
                    fold_top1.mean() / fold_top1.std(ddof=1) * np.sqrt(252)
                ),
                "top10_mean_bps": float(fold_top10.mean() * 10_000),
                "top10_sharpe": float(
                    fold_top10.mean() / fold_top10.std(ddof=1) * np.sqrt(252)
                ),
            }
        )
    trim_count = max(1, int(len(daily_top) * 0.05))
    trimmed_mean = float(daily_top.sort_values().iloc[trim_count:-trim_count].mean())
    mean_without_best_5 = float(daily_top.nsmallest(max(1, len(daily_top) - 5)).mean())
    risk_gate = (
        "PASS"
        if (
            net_bps >= MIN_NET_BPS
            and win_rate >= MIN_WIN_RATE
            and sharpe >= MIN_SHARPE
            and trimmed_mean > 0
            and mean_without_best_5 > 0
            and strictly_monotonic
        )
        else "BLOCK_WEAK_OR_UNSTABLE_EDGE"
    )
    final_dates = set(dates[-args.rolling_train_sessions:])
    final_model = _fit(
        mature.loc[mature["timestamp"].isin(final_dates)],
        args.target_transform,
        args.num_leaves,
        args.min_child_samples,
        args.colsample_bytree,
        args.reg_lambda,
        args.reg_alpha,
        args.min_split_gain,
        args.max_depth,
        args.learning_rate,
        args.n_estimators,
        args.huber_alpha,
        args.subsample,
        args.subsample_freq,
        args.weight_half_life,
        args.min_child_weight,
        args.path_smooth,
        args.feature_fraction_bynode,
        args.extra_trees,
        args.max_bin,
        args.min_data_in_bin,
        args.boosting_type,
        args.data_sample_strategy,
        args.drop_rate,
        args.skip_drop,
        args.max_drop,
        args.top_rate,
        args.other_rate,
        args.top1_weight,
        args.profit_early_stopping_rounds,
        args.target_clip_abs,
    )
    importance = pd.DataFrame(
        {"feature": ACTIVE_FEATURE_COLUMNS, "gain": final_model.booster_.feature_importance("gain")}
    ).sort_values("gain", ascending=False)
    importance["gain_share"] = importance["gain"] / importance["gain"].sum()
    model_dir = args.model_dir or (output / "models")
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_dir / f"{args.output_prefix}_ranker.joblib")
    importance.to_csv(model_dir / f"{args.output_prefix}_feature_importance.csv", index=False)
    payload = {
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "model": f"LightGBM {ACTIVE_OBJECTIVE} rolling cross-sectional scorer",
        "objective": ACTIVE_OBJECTIVE,
        "target_transform": args.target_transform,
        "training_target_column": ACTIVE_TARGET_COLUMN,
        "target_clip_abs": args.target_clip_abs,
        "excluded_features": args.exclude_feature,
        "num_leaves": args.num_leaves,
        "min_child_samples": args.min_child_samples,
        "colsample_bytree": args.colsample_bytree,
        "reg_lambda": args.reg_lambda,
        "reg_alpha": args.reg_alpha,
        "min_split_gain": args.min_split_gain,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "n_estimators": args.n_estimators,
        "huber_alpha": args.huber_alpha,
        "subsample": args.subsample,
        "subsample_freq": args.subsample_freq,
        "weight_half_life": args.weight_half_life,
        "min_child_weight": args.min_child_weight,
        "path_smooth": args.path_smooth,
        "feature_fraction_bynode": args.feature_fraction_bynode,
        "extra_trees": args.extra_trees,
        "max_bin": args.max_bin,
        "min_data_in_bin": args.min_data_in_bin,
        "boosting_type": args.boosting_type,
        "data_sample_strategy": args.data_sample_strategy,
        "drop_rate": args.drop_rate,
        "skip_drop": args.skip_drop,
        "max_drop": args.max_drop,
        "top_rate": args.top_rate,
        "other_rate": args.other_rate,
        "top1_weight": args.top1_weight,
        "profit_early_stopping_rounds": args.profit_early_stopping_rounds,
        "feature_library_size": len(ALL_FEATURE_COLUMNS),
        "trained_feature_count": len(ACTIVE_FEATURE_COLUMNS),
        "trained_features": ACTIVE_FEATURE_COLUMNS,
        "rolling_train_sessions": args.rolling_train_sessions,
        "validation_sessions": args.validation_sessions,
        "walk_forward_retrain_every_sessions": args.retrain_every,
        "walk_forward_retrain_anchor": "full_history_trading_calendar",
        "production_retrain_every_sessions": 1,
        "validation_top_n_mean_net": float(daily_top.mean()),
        "validation_top_n_win_rate": win_rate,
        "validation_top_n_sharpe": sharpe,
        "validation_top_n_sortino": sortino,
        "validation_top_n_max_drawdown": max_drawdown,
        "validation_positive_21d_windows": float((rolling_21 > 0).mean()),
        "validation_worst_21d_mean_net": float(rolling_21.min()),
        "validation_5pct_trimmed_mean_net": trimmed_mean,
        "validation_mean_without_best_5_nights": mean_without_best_5,
        "validation_all_mean_net": float(daily_all.mean()),
        "validation_relative_edge_bps": float((daily_top - daily_all).mean() * 10_000),
        "validation_rank_slot_metrics": slot_metrics.to_dict("records"),
        "validation_adjacent_rank_spreads_bps": adjacent_spreads.tolist(),
        "validation_monotonic_violations": monotonic_violations,
        "validation_strictly_monotonic_rank_returns": strictly_monotonic,
        "validation_fold_metrics": fold_rows,
        "validation_worst_fold_top1_mean_bps": float(
            min(row["top1_mean_bps"] for row in fold_rows)
        ),
        "validation_median_fold_top1_mean_bps": float(
            np.median([row["top1_mean_bps"] for row in fold_rows])
        ),
        "validation_positive_top1_folds": int(
            sum(row["top1_mean_bps"] > 0 for row in fold_rows)
        ),
        "top_features": importance.head(20).to_dict("records"),
        "cost_bps": config.round_trip_cost_bps,
        "panel_path": str(cache_path),
        "exclude_label_abs_above": args.exclude_label_abs_above,
        "risk_gate_thresholds": {
            "minimum_net_bps": MIN_NET_BPS,
            "minimum_win_rate": MIN_WIN_RATE,
            "minimum_sharpe": MIN_SHARPE,
            "require_positive_5pct_trimmed_mean": True,
            "require_positive_without_best_5_nights": True,
            "require_strict_top1_through_top10_monotonicity": True,
        },
        "risk_gate": risk_gate,
        "real_orders_enabled": False,
    }
    (model_dir / f"{args.output_prefix}_ranker.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    daily_report = pd.DataFrame(
        {
            "top10_mean_net": daily_top,
            "universe_mean_net": daily_all,
        }
    )
    daily_report["relative_edge"] = (
        daily_report["top10_mean_net"] - daily_report["universe_mean_net"]
    )
    daily_report.to_csv(model_dir / f"{args.output_prefix}_walk_forward_daily.csv")
    slot_metrics.to_csv(model_dir / f"{args.output_prefix}_rank_slot_metrics.csv", index=False)
    selected.to_parquet(model_dir / f"{args.output_prefix}_selected_predictions.parquet", index=False)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
