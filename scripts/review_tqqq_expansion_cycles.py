"""Test as-filed operating expansion breadth around Nasdaq highs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from moneymore.data.us_financials import EXPANSION_TAGS, FACTS, FILINGS

TECH = ("Chips", "Softw", "Hardw", "LabEq")
METRICS = ("capex", "research", "issuance")
FINANCIAL_SCORES = ("capex", "investment", "joint")
SCORES = (*FINANCIAL_SCORES, "qqq63_weakness", "market_stress")
ARCHIVE_END = pd.Timestamp("2026-03-31")


def observations(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with duckdb.connect(config={"threads": 4}) as db:
        filings = db.execute("""
            SELECT accession, cik, industry, public_date, last_day(period) AS period_end
            FROM read_parquet(?) WHERE industry IN ? AND public_date >= '2020-07-01'
        """, [str(root / f"{FILINGS}.parquet"), list(TECH)]).fetchdf()
        db.register("filings", filings)
        facts = db.execute("""
            WITH amounts AS (
                SELECT a.accession, a.tag, a.duration_months AS duration,
                    last_day(a.period_end) AS fiscal_end, max(a.public_date) AS available,
                    min(a.value) AS value
                FROM read_parquet(?) a JOIN filings f USING (accession, cik)
                WHERE a.tag IN ? AND a.taxonomy = 'us-gaap' AND a.unit = 'USD'
                    AND a.duration_months IN (3, 6, 9, 12)
                    AND last_day(a.period_end) IN (f.period_end, last_day(f.period_end - INTERVAL 1 YEAR))
                    AND isfinite(a.value)
                GROUP BY ALL HAVING count(DISTINCT a.value) = 1 AND min(a.value) >= 0
            )
            SELECT f.accession, f.cik, f.industry, f.period_end,
                greatest(f.public_date, a.available, p.available) AS public_date,
                a.tag, a.duration, a.value - p.value AS change
            FROM filings f JOIN amounts a ON f.accession = a.accession AND a.fiscal_end = f.period_end
            JOIN amounts p ON p.accession = a.accession AND p.tag = a.tag AND p.duration = a.duration
                AND p.fiscal_end = last_day(a.fiscal_end - INTERVAL 1 YEAR)
        """, [str(root / f"{FACTS}.parquet"), list(EXPANSION_TAGS)]).fetchdf()
    facts["metric"] = facts.tag.map(dict(zip(EXPANSION_TAGS, ("capex", "capex", "research", "issuance"))))
    # Use an alternate CapEx tag only when both periods are reported with that same tag.
    return filings, facts


def breadth(filings: pd.DataFrame, facts: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    known = filings.loc[filings.public_date.lt(date)].sort_values(["period_end", "public_date", "accession"])
    latest = known.drop_duplicates("cik", keep="last")
    latest = latest.loc[(date - latest.period_end).dt.days.between(0, 155)]
    fresh = facts.loc[facts.public_date.lt(date)].merge(
        latest[["cik", "period_end"]], on=["cik", "period_end"], how="inner", validate="many_to_one"
    )
    fresh = fresh.assign(preferred=fresh.tag.eq(EXPANSION_TAGS[0])).sort_values(
        ["public_date", "accession", "preferred"]
    )
    # A partial amendment may leave other values from the original filing available.
    fresh = fresh.drop_duplicates(["cik", "duration", "metric"], keep="last")
    marginal = fresh.sort_values("duration").drop_duplicates(["cik", "metric"], keep="last")
    records = []
    for name, universe in [("tech", latest), *((key, latest.loc[latest.industry.eq(key)]) for key in TECH)]:
        if universe.empty:
            continue
        ciks = universe.cik
        result = {"industry": name, "active": len(ciks)}
        for metric in METRICS:
            part = marginal.loc[marginal.cik.isin(ciks) & marginal.metric.eq(metric)]
            result[f"{metric}_n"] = len(part)
            result[f"{metric}_up"] = int(part.change.gt(0).sum())
        panel = fresh.loc[fresh.cik.isin(ciks)].pivot(index=["cik", "duration"], columns="metric", values="change")
        panel = panel.reindex(columns=list(METRICS)).sort_index(level="duration")
        for label, metrics in (("investment", METRICS[:2]), ("joint", METRICS)):
            paired = panel.dropna(subset=list(metrics)).reset_index().drop_duplicates("cik", keep="last")
            result[f"{label}_n"] = len(paired)
            result[f"{label}_up"] = int(paired[list(metrics)].gt(0).all(axis=1).sum())
        records.append(result)
    return pd.DataFrame(records).set_index("industry")


def market_study(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    qqq = history.qqqClose
    index = history.index
    outcomes = []
    for i, date in enumerate(index[:-63]):
        entry = history.tqqqOpen.iloc[i + 1]
        path = history.iloc[i + 1:i + 64]
        low = path[["tqqqOpen", "tqqqClose"]].min().min() / entry - 1
        end = path.tqqqClose.iloc[-1] / entry - 1
        outcomes.append((date, index[i + 63], low, end))
    outcome = pd.DataFrame(outcomes, columns=["date", "settled", "loss", "return"]).set_index("date")
    outcome["qqq63_weakness"] = -qqq.pct_change(63)
    channels = pd.DataFrame({
        "qqq": -np.log(qqq).diff(5),
        "credit": -np.log(history.HYG / history.IEI).diff(5),
        "volatility": np.log(history.VIX / history.VIX3M).diff(5),
    })
    past = channels.shift().rolling(252, min_periods=126)
    outcome["market_stress"] = ((channels - past.mean()) / past.std()).clip(-4, 4).mean(axis=1, skipna=False)
    outcome["near_high"] = qqq.div(qqq.rolling(63).max()).ge(0.97)
    outcome["local_high"] = qqq.eq(qqq.rolling(41, center=True).max())
    weeks = history.groupby(index.to_period("W-FRI")).tail(1).index
    return outcome, outcome.reindex(weeks).dropna(subset=["settled"])


def evaluate(weekly: pd.DataFrame, tail_threshold: float = 0.30) -> dict:
    scores = weekly.copy()
    for name in FINANCIAL_SCORES:
        scores[name] = scores[f"{name}_up"].div(scores[f"{name}_n"].where(scores[f"{name}_n"].ge(12)))
    rank_inputs = scores[["market_stress", "joint"]]
    past = rank_inputs.shift().rolling(52, min_periods=20)
    scores["blend"] = ((rank_inputs - past.mean()) / past.std()).clip(-4, 4).mean(axis=1, skipna=False)
    scores = scores.loc[scores.near_high & scores.index.to_series().le(ARCHIVE_END)].dropna(subset=list(SCORES))
    tail = scores.loss.le(-tail_threshold)
    calm = scores.loss.gt(-0.15) & scores["return"].gt(0)
    scored = scores.loc[tail | calm]
    output = {
        "weeks": len(scores), "tail": int(tail.sum()), "calm": int(calm.sum()),
        "median_joint_issuers": float(scores.joint_n.median()) if len(scores) else None,
        "scores": {},
    }
    common = scored.dropna(subset=["blend"])
    classes = tail.loc[common.index]
    if classes.nunique() == 2:
        output["incremental"] = {
            "weeks": len(common), "tail": int(classes.sum()),
            **{name: float(roc_auc_score(classes, common[name])) for name in ("market_stress", "joint", "blend")},
        }
    for name in SCORES:
        auc = roc_auc_score(tail.loc[scored.index], scored[name]) if tail.any() and calm.any() else None
        phases = []
        for offset in range(13):
            phase = scored.loc[scored.index.intersection(weekly.index[offset::13])]
            phase = phase.loc[phase.settled.shift().lt(phase.index) | phase.settled.shift().isna()]
            classes = tail.loc[phase.index]
            if min(classes.sum(), (~classes).sum()) >= 2:
                phases.append(float(roc_auc_score(classes, phase[name])))
        # Greedy, nonoverlapping alerts relative to the model's own prior scores.
        threshold = scores[name].shift().rolling(52, min_periods=20).quantile(0.80)
        flagged = scores.loc[scores[name].gt(threshold)]
        kept, settle = [], pd.Timestamp.min
        for date, row in flagged.iterrows():
            if date >= settle:
                kept.append(date)
                settle = row.settled
        windows = scores.loc[kept]
        output["scores"][name] = {
            "auc_tail_vs_calm": auc,
            "nonoverlapping_phase_auc_range": [min(phases), max(phases)] if phases else None,
            "nonoverlapping_phase_count": len(phases),
            "tail_median": float(scores.loc[tail, name].median()) if tail.any() else None,
            "calm_median": float(scores.loc[calm, name].median()) if calm.any() else None,
            "nonoverlapping_alerts": len(kept),
            "tail_hits": int(windows.loss.le(-tail_threshold).sum()),
            "mean_net_exit_edge": float((0.9975 / 1.0025 - 1 - windows["return"]).mean()) if kept else None,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--curated", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    history = pd.read_csv(args.history, index_col=0, parse_dates=True).sort_index()
    outcomes, weekly = market_study(history)
    weekly = weekly.loc["2021-01-01":ARCHIVE_END]
    filings, facts = observations(args.curated)
    cutoff = pd.Timestamp("2022-08-15")
    pd.testing.assert_frame_equal(
        breadth(filings, facts, cutoff),
        breadth(filings.loc[filings.public_date.lt(cutoff)], facts.loc[facts.public_date.lt(cutoff)], cutoff),
    )
    peaks = outcomes.loc["2021-01-01":ARCHIVE_END]
    peaks = peaks.loc[peaks.local_high & peaks.near_high & peaks.loss.le(-0.30)]
    dates = sorted(set(weekly.index) | set(peaks.index) | {pd.Timestamp("2026-10-06")})
    snapshots = pd.concat({date: breadth(filings, facts, date) for date in dates}, names=["date", "industry"])
    tech = snapshots.xs("tech", level="industry")
    result = evaluate(weekly.join(tech))
    result["subperiods"] = {
        "2021_2023": evaluate(weekly.join(tech).loc[:"2023-12-31"]),
        "2024_2026q1": evaluate(weekly.join(tech).loc["2024-01-01":]),
    }
    result["sensitivity"] = {
        f"{threshold:.0%}": evaluate(weekly.join(tech), threshold)
        for threshold in (0.25, 0.35)
    }
    peak_context = snapshots.loc[snapshots.index.get_level_values("date").isin(peaks.index)]
    result["peaks"] = peak_context.reset_index().merge(
        peaks[["loss"]], left_on="date", right_index=True
    ).to_dict("records")
    result["october_6"] = snapshots.loc["2026-10-06"].reset_index().to_dict("records")
    result["filing_rows"] = len(filings)
    result["paired_observations"] = len(facts)
    result["market_last_date"] = str(history.index[-1].date())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, default=str, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("weeks", "tail", "calm", "scores")}, indent=2))


if __name__ == "__main__":
    main()
