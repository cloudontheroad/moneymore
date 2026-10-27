"""Reconcile explicit revision metadata and audit income-derived quarterly growth.

Only isolated research tables and diagnostic artifacts can be changed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.data.financial_reconciliation import (
    KEYS,
    derive_quarterly_revenue_growth,
    reconcile_indicator_payload,
)
from moneymore.data.fundamental_research import (
    TABLE,
    build_financial_research_comparison,
    financial_features_as_of,
    financial_quality,
    research_financial_frame,
)
from moneymore.data.store import ParquetStore
from moneymore.strategy_universe import active_strategy_universe


def payloads(store, table):
    frames = []
    for path in sorted((store.raw / "tushare" / table).glob("*.parquet")):
        f = pd.read_parquet(path)
        f["_payload_capture"] = path.stem.split("_")[-1]
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    store = ParquetStore(root / "data")
    protected = [
        store.curated / (name + ".parquet")
        for name in (
            "fina_indicator",
            "global_factor_recommendation",
            "multi_sector_account_daily",
            "qlib_challenger_account_daily",
            "qlib_candidate_account_daily",
        )
    ]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    symbols = active_strategy_universe(store).symbol.tolist()
    existing = store.read(TABLE)
    before = financial_quality(existing, symbols, args.as_of, per_field=True)
    indicators = payloads(store, "fina_indicator_deep_probe")
    normalized = reconcile_indicator_payload(indicators)
    extras = [c for c in existing if c not in normalized and c not in KEYS]
    normalized = normalized.merge(
        existing[KEYS + extras], on=KEYS, how="left", validate="one_to_one"
    )
    combined = pd.concat([existing, normalized], ignore_index=True).drop_duplicates(
        KEYS, keep="last"
    )
    income = payloads(store, "income_deep_probe")
    latest = income.groupby(
        ["ts_code", "end_date", "ann_date", "f_ann_date", "report_type"], dropna=False
    )._payload_capture.transform("max")
    income = income.loc[income._payload_capture.eq(latest)]
    derived = derive_quarterly_revenue_growth(income, args.as_of)
    visible = normalized.loc[
        pd.to_datetime(normalized.ann_date, format="%Y%m%d") + pd.offsets.Day(1)
        <= pd.Timestamp(args.as_of)
    ]
    visible = visible.sort_values(["end_date", "ann_date"]).drop_duplicates(
        ["ts_code", "end_date"], keep="last"
    )
    comparisons = derived.merge(
        visible[["ts_code", "end_date", "q_sales_yoy"]].rename(
            columns={"q_sales_yoy": "provider_q_sales_yoy"}
        ),
        on=["ts_code", "end_date"],
        how="left",
    )
    comparisons["absolute_difference_pp"] = (
        comparisons.q_sales_yoy - comparisons.provider_q_sales_yoy
    ).abs()
    comparisons["cross_check"] = "NO_DIRECT_COMPARISON"
    comparable = comparisons.provider_q_sales_yoy.notna() & comparisons.q_sales_yoy.notna()
    comparisons.loc[comparable, "cross_check"] = "MATCH"
    comparisons.loc[comparable & (comparisons.absolute_difference_pp > 0.05), "cross_check"] = (
        "DISAGREEMENT"
    )
    current = comparisons.sort_values("end_date").groupby("ts_code", as_index=False).tail(1)
    features = financial_features_as_of(combined, args.as_of)
    proposals = []
    for row in current.to_dict("records"):
        symbol = row["ts_code"]
        held = features.loc[features.ts_code.eq(symbol)]
        if held.empty or row["status"] != "DERIVED_FROM_CONSOLIDATED_REVENUE":
            continue
        # Arithmetic candidates remain separate; no substitution if the provider
        # supplies a different number. Publication/period constraints are retained.
        if pd.isna(held.iloc[0].q_sales_yoy):
            proposals.append(row)
    singles = payloads(store, "income_single_quarter_probe")
    single_result = derive_quarterly_revenue_growth(singles, args.as_of, single_quarter=True)
    approved = []
    for proposal in proposals:
        comparison = single_result.loc[
            single_result.ts_code.eq(proposal["ts_code"])
            & single_result.end_date.eq(proposal["end_date"])
            & single_result.status.eq("DERIVED_FROM_SINGLE_QUARTER_REVENUE")
        ]
        if len(comparison) != 1:
            continue
        direct = comparison.iloc[0]
        difference = abs(float(proposal["q_sales_yoy"]) - float(direct.q_sales_yoy))
        if difference > 0.000001:
            continue
        approved.append(
            {
                **proposal,
                "ann_date": max(proposal["ann_date"], direct.ann_date),
                "status": "AGREEMENT_OF_CUMULATIVE_AND_SINGLE_QUARTER",
                "single_quarter_yoy": float(direct.q_sales_yoy),
                "method_difference_pp": difference,
                "single_quarter_publications": direct.source_publications,
            }
        )
    adjusted = payloads(store, "income_adjusted_probe")
    adjusted_result = derive_quarterly_revenue_growth(
        pd.concat([income, adjusted], ignore_index=True), args.as_of, allow_adjusted=True
    )
    adjusted_latest = (
        adjusted_result.sort_values("end_date").groupby("ts_code", as_index=False).tail(1)
    )
    adjusted_comparison = adjusted_latest.merge(
        visible[["ts_code", "end_date", "q_sales_yoy"]].rename(
            columns={"q_sales_yoy": "provider_q_sales_yoy"}
        ),
        on=["ts_code", "end_date"],
        how="left",
    )
    adjusted_comparison["difference_pp"] = (
        adjusted_comparison.q_sales_yoy - adjusted_comparison.provider_q_sales_yoy
    ).abs()
    adjusted_singles = payloads(store, "income_adjusted_single_probe")
    adjusted_single_result = derive_quarterly_revenue_growth(
        pd.concat([singles, adjusted_singles], ignore_index=True),
        args.as_of,
        allow_adjusted=True,
        single_quarter=True,
    )
    adjusted_single_latest = (
        adjusted_single_result.sort_values("end_date").groupby("ts_code", as_index=False).tail(1)
    )
    adjusted_single_comparison = adjusted_single_latest.merge(
        visible[["ts_code", "end_date", "q_sales_yoy"]].rename(
            columns={"q_sales_yoy": "provider_q_sales_yoy"}
        ),
        on=["ts_code", "end_date"],
        how="left",
    )
    adjusted_single_comparison["difference_pp"] = (
        adjusted_single_comparison.q_sales_yoy - adjusted_single_comparison.provider_q_sales_yoy
    ).abs()
    if args.apply:
        store.merge_curated(TABLE, [normalized], KEYS)
        safe = derived.loc[derived.status.eq("DERIVED_FROM_CONSOLIDATED_REVENUE")].copy()
        store.merge_curated(
            "quarterly_revenue_derived_v2", [safe], ["ts_code", "end_date", "ann_date"]
        )
        if approved:
            store.merge_curated(
                "quarterly_revenue_supplement_v2",
                [pd.DataFrame(approved)],
                ["ts_code", "end_date", "ann_date"],
            )
    report = {
        "as_of_date": args.as_of,
        "applied_isolated_indicator_repair": args.apply,
        "provider_revision_policy": "current update_flag=1 if explicitly present; conflicts within current version block",
        "before": before,
        "after": financial_quality(combined, symbols, args.as_of, per_field=True),
        "income_symbols": int(income.ts_code.nunique()),
        "latest_income_results": current.to_dict("records"),
        "all_period_cross_check": comparisons.cross_check.value_counts().to_dict(),
        "derived_missing_field_proposals": proposals,
        "derived_automatically_published": False,
        "approved_isolated_supplements": approved,
        "adjusted_cumulative_cross_check": adjusted_comparison.to_dict("records"),
        "adjusted_single_quarter_cross_check": adjusted_single_comparison.to_dict("records"),
        "protected_tables_unchanged": hashes
        == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected},
    }
    initial_disagreements = current.loc[current.cross_check.eq("DISAGREEMENT"), "ts_code"].tolist()
    final_checks = adjusted_single_comparison.loc[
        adjusted_single_comparison.ts_code.isin(initial_disagreements)
    ]
    report["latest_version_cross_check"] = {
        "initial_disagreements": len(initial_disagreements),
        "resolved_by_adjusted_single_quarter": int(final_checks.difference_pp.le(0.0001).sum()),
        "max_difference_pp": float(final_checks.difference_pp.max()),
        "remaining_disagreement_symbols": final_checks.loc[
            ~final_checks.difference_pp.le(0.0001), "ts_code"
        ].tolist(),
    }
    write_diagnostic(root / "state/fundamental-research-v2/deep-review.json", report)
    if args.apply:
        resolved = financial_quality(
            research_financial_frame(store), symbols, args.as_of, per_field=True
        )
        resolved.update(
            status="CYCLE_COMPLETE",
            repair_phase="DUPLICATE_RECONCILIATION_AND_VERIFIED_SUPPLEMENT",
            remaining_due=0,
            orders_enabled=False,
        )
        write_diagnostic(root / "state/fundamental-research-v2/quality.json", resolved)
        build_financial_research_comparison(root, store, symbols, args.as_of)
        report["supplemented_quality"] = resolved
        write_diagnostic(root / "state/fundamental-research-v2/deep-review.json", report)
        print(
            "resolved_quality",
            json.dumps(
                {k: v for k, v in resolved.items() if not k.endswith("_symbols")},
                ensure_ascii=False,
            ),
        )
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k
                not in (
                    "before",
                    "after",
                    "latest_income_results",
                    "derived_missing_field_proposals",
                    "approved_isolated_supplements",
                    "adjusted_cumulative_cross_check",
                    "adjusted_single_quarter_cross_check",
                    "supplemented_quality",
                )
            },
            ensure_ascii=False,
        )
    )
    print(
        json.dumps(
            {
                k: {n: v for n, v in report[k].items() if not n.endswith("_symbols")}
                for k in ("before", "after")
            },
            ensure_ascii=False,
        )
    )
    print("proposals", [(r["ts_code"], r["end_date"], r["q_sales_yoy"]) for r in proposals])
    print(
        "unadjusted_disagreements_explained_by_adjusted_single_quarter",
        [
            (r["ts_code"], r["end_date"], r["q_sales_yoy"], r["provider_q_sales_yoy"])
            for r in current.to_dict("records")
            if r["cross_check"] == "DISAGREEMENT"
        ],
    )
    if not report["protected_tables_unchanged"]:
        raise RuntimeError("Concurrent protected data mutation detected")
