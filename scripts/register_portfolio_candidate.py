"""Register one explicitly authorized forward paper portfolio variant."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.config import BacktestConfig
from moneymore.data.store import ParquetStore
from moneymore.execution.paper import PaperBroker
from moneymore.portfolio_candidate import prepare_portfolio_candidate
from moneymore.qlib_candidate_observer import candidate_catalog
from moneymore.strategy_universe import execution_strategy_id


def run(root, tag, source):
    if len(tag) != 8 or not tag.isdigit() or source >= tag:
        raise ValueError("INVALID_FORWARD_CANDIDATE_TAG")
    catalog = candidate_catalog(root)
    parent = next(r for r in catalog if r["candidate_tag"] == source)
    if not parent["research_gate_passed"] or parent["archived"]:
        raise ValueError("SOURCE_MODEL_NOT_ACTIVE_OR_GATE_FAILED")
    path = root / "state/qlib-portfolio-candidates" / (tag + ".json")
    if not path.exists():
        if any(r["candidate_tag"] == tag for r in catalog):
            raise ValueError("CANDIDATE_TAG_ALREADY_USED")
        write_diagnostic(
            path,
            {
                "candidate_tag": tag,
                "source_candidate_tag": source,
                "display_name": "1010模型＋固定风险预算"
                if source == "20261010"
                else source + "模型＋固定风险预算",
                "portfolio_variant": "fixed_budget_v1",
                "created_at": datetime.now(UTC).isoformat(),
                "activation_date": tag,
                "model_retrained": False,
                "real_orders_enabled": False,
                "promotion_eligible": False,
                "research_gate_scope": "INHERITED_MODEL_ONLY_PORTFOLIO_FORWARD_EXPERIMENT",
            },
        )
    own = next(r for r in candidate_catalog(root) if r["candidate_tag"] == tag)
    if own["source_candidate_tag"] != source:
        raise ValueError("EXISTING_REGISTRY_SOURCE_CONFLICT")
    report = root / "state/qlib-candidate-shadow" / tag / (tag + ".json")
    if report.exists():
        existing = json.loads(report.read_text(encoding="utf-8"))
        if existing.get("planned_trade_date"):
            return existing
    signal = json.loads(
        (root / "state/qlib-candidate-shadow" / source / (tag + ".json")).read_text(
            encoding="utf-8"
        )
    )
    if signal["trade_date"] != tag:
        raise ValueError("SOURCE_SIGNAL_DATE_MISMATCH")
    config = BacktestConfig.from_yaml(root / "configs/default.yaml")
    store = ParquetStore(root / "data")
    broker = PaperBroker(root / "state/paper_orders.sqlite3")
    broker.initialize_account(config.initial_cash, own["account_id"])
    score_frame = pd.DataFrame(signal["scores"])
    # First-close bootstrap deliberately uses the parent's saved targets, not
    # its holdings or historical NAV. Future days infer the pinned model anew.
    close = (
        store.read("daily", filters=[("trade_date", "=", tag)]).set_index("ts_code").close.to_dict()
    )
    result = prepare_portfolio_candidate(
        root=root,
        store=store,
        broker=broker,
        config=config,
        candidate=own,
        trade_date=tag,
        score_frame=score_frame,
        bootstrap_weights=signal["target_weights"],
        marks=close,
        strategy_id=execution_strategy_id("qlib_gru_alpha360_v1_candidate_" + tag, store),
        model_id=signal["model_id"],
    )
    print(
        json.dumps(
            {
                "account_id": own["account_id"],
                "status": result.status,
                "report_path": result.report_path,
            },
            ensure_ascii=True,
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[1], args.tag, args.source)
