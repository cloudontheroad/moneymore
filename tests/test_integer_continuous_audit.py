import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.config import BacktestConfig
from moneymore.data.store import ParquetStore
from moneymore.execution_replay import run_execution_replay
from moneymore.research.integer_execution import plan_integer_execution

spec = importlib.util.spec_from_file_location(
    "integer_continuous_audit",
    Path(__file__).resolve().parents[1] / "scripts/audit_integer_continuous_replay.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("corruption", [None, "quantity", "style"])
def test_independent_audit_reconstructs_ledger_and_rejects_corruption(tmp_path, corruption):
    config = BacktestConfig(
        initial_cash=10000,
        commission_rate=0.00015,
        minimum_commission=5,
        stamp_duty_rate=0.0005,
        transfer_fee_rate=0.00001,
        slippage_bps=2,
        lot_size=100,
        max_position_weight=1.0,
        max_gross_exposure=1.0,
        max_drawdown=0.2,
    )
    folder = tmp_path / "state/enhanced-indexing-research"
    output = folder / "isolated"
    output.mkdir(parents=True)
    bars = pd.DataFrame(
        [
            {"date": d, "symbol": "AAA", "open": 10.0, "close": 11.0}
            for d in ("20260105", "20260106")
        ]
    )
    ParquetStore(tmp_path / "data").merge_curated(
        "daily",
        [bars.rename(columns={"date": "trade_date", "symbol": "ts_code"})],
        ["trade_date", "ts_code"],
    )
    context = {}

    def close(date, snapshot):
        context["date"] = date
        return pd.DataFrame([{"date": date, "symbol": "AAA", "target_weight": 0.5}])

    def opening(date, snapshot, quotes):
        if not context:
            return {"status": "NO_PRIOR_SIGNAL"}
        plan = plan_integer_execution(
            {"AAA": 0.5},
            quotes.open,
            {},
            {},
            snapshot["cash"],
            snapshot["equity"],
            config,
            pd.DataFrame({"style": [-1.0, 1.0]}, index=["AAA", "BBB"]),
            {"style": 2.0},
        )
        return {
            **plan,
            "signal_date": context["date"],
            "execution_style_proof": {
                "standardized": {"AAA": {"style": -1.0}},
                "limits": {"style": 2.0},
            },
        }

    replay = run_execution_replay(
        bars=bars,
        targets=pd.DataFrame(columns=["date", "symbol", "target_weight"]),
        config=config,
        database=output / "fake-native_pca.sqlite3",
        strategy_id="fake",
        account_id="fake",
        closing_target_provider=close,
        opening_plan_provider=opening,
    )
    replay.equity.to_parquet(output / "fake-native_pca-equity.parquet", index=False)
    plan = replay.summary["opening_plan_audits"][1]["plan"]
    if corruption == "quantity":
        plan["target_quantities"]["AAA"] += 100
    elif corruption == "style":
        plan["execution_style_proof"]["limits"]["style"] = 0.1
    write_diagnostic(
        folder / "integer-continuous-replay.json",
        {
            "start_date": "20260105",
            "as_of_date": "20260106",
            "config": config.model_dump(),
            "output_directory": str(output),
            "mode": "TEST_ONLY",
            "protected_equity_tables_unchanged": True,
            "runs": [
                {
                    "account": "fake",
                    "variant": "native_pca",
                    "status": "COMPLETED_DIAGNOSTIC_ONLY",
                    "summary": replay.summary,
                }
            ],
        },
    )
    if corruption:
        with pytest.raises(ValueError, match="LEDGER_"):
            module.run(tmp_path)
    else:
        audit = module.run(tmp_path)
        assert audit["runs"][0]["verified_opens"] == 1
        assert audit["runs"][0]["ledger_cash_max_error"] < 0.01
