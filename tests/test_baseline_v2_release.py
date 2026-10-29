import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from moneymore.baseline_v2_release_view import release_view
from moneymore.research.baseline_v2_release import (
    actual_stock_attribution,
    digest,
    release_decision,
    verify_manifest,
)
from moneymore.research.baseline_v2_release_runtime import record_prospective_observation
from moneymore.research.baseline_v2_release_style import link_with_unknown


def protocol():
    return {
        "real_orders_enabled": False,
        "required_stages": ["data", "forward_observation"],
        "forward_review_days": 60,
    }


def test_missing_failed_or_unmatured_stage_never_releases():
    assert release_decision(protocol(), {})["paper_release_authorized"] is False
    reports = {
        "data": {"status": "PASS"},
        "forward_observation": {
            "status": "PASS",
            "mature_prospective_days": 59,
            "backfilled_days_counted": False,
        },
    }
    assert release_decision(protocol(), reports)["blocked_stages"] == ["forward_observation"]
    reports["forward_observation"]["mature_prospective_days"] = 60
    assert release_decision(protocol(), reports)["paper_release_authorized"] is True
    assert release_decision(protocol(), reports)["real_orders_enabled"] is False
    assert (
        release_decision(protocol(), reports, evidence_valid=False)["paper_release_authorized"]
        is False
    )
    reports["forward_observation"]["backfilled_days_counted"] = True
    assert not release_decision(protocol(), reports)["paper_release_authorized"]


def test_real_trading_cannot_be_authorized_by_release_protocol():
    with pytest.raises(ValueError, match="REAL_TRADING"):
        release_decision({**protocol(), "real_orders_enabled": True}, {})


def test_observation_can_start_before_forward_days_but_not_before_hard_gates():
    reports = {"data": {"status": "PASS"}}
    result = release_decision(protocol(), reports)
    assert result["observation_activation_authorized"]
    assert not result["promotion_authorized"]
    assert not result["paper_release_authorized"]
    assert result["observation_blocked_stages"] == []
    assert result["blocked_stages"] == ["forward_observation"]
    assert not release_decision(protocol(), {}, evidence_valid=True)["observation_activation_authorized"]
    assert not release_decision(protocol(), reports, evidence_valid=False)["observation_activation_authorized"]
    with pytest.raises(ValueError, match="ALL_NON_FORWARD_HARD_GATES"):
        release_decision({**protocol(), "observation_required_stages": []}, reports)


def test_manifest_change_fails_closed(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("before", encoding="utf-8")
    manifest = {"input_hashes": {str(path): digest(path)}}
    verify_manifest(manifest)
    path.write_text("after", encoding="utf-8")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        verify_manifest(manifest)


def test_economic_stock_pnl_preserves_fee_and_receivable_settlement():
    keys = ["20240102", "20240103", "20240104", "20240105"]
    equity = pd.DataFrame({"trade_date": keys, "equity": [200, 199, 199, 199]})
    fills = pd.DataFrame(
        [
            {
                "trade_date": keys[1],
                "symbol": "X",
                "side": "BUY",
                "quantity": 10,
                "price": 10,
                "fee": 1,
            }
        ]
    )
    holdings = pd.DataFrame(
        [
            {"trade_date": key, "symbol": "X", "market_value": value, "economic_quantity": 10}
            for key, value in zip(keys[1:], [100, 90, 90], strict=True)
        ]
    )
    ledger = pd.DataFrame(
        [{"trade_date": keys[3], "symbol": "X", "action_type": "CASH_DIVIDEND", "cash_amount": 10}]
    )
    rights = pd.DataFrame(
        [
            {
                "action_key": "A",
                "symbol": "X",
                "cash_per_share": 1,
                "entitled_quantity": 10,
                "cash_pay_date": keys[3],
            }
        ]
    )
    actions = pd.DataFrame([{"action_key": "A", "ex_date": pd.Timestamp(keys[2])}])
    bars = pd.DataFrame(
        [
            {"date": pd.Timestamp(key), "symbol": "X", "close": close}
            for key, close in zip(keys[1:], [10, 9, 9], strict=True)
        ]
    )
    industry = pd.DataFrame(
        [
            {
                "ts_code": "X",
                "l1_code": "I",
                "l1_name": "历史行业",
                "in_date": "20200101",
                "out_date": None,
            }
        ]
    )
    rows, checks = actual_stock_attribution(
        equity, fills, holdings, ledger, rights, actions, bars, industry
    )
    assert checks.pnl_error.abs().max() == 0
    assert rows.net_pnl.sum() == -1
    assert rows.explicit_fee.sum() == 1
    assert rows.loc[rows.trade_date.eq(keys[3]), "net_pnl"].sum() == 0


def test_unmatched_style_days_are_not_dropped_from_return_linking():
    rows = [
        {"actual_return": 0.1, "style": 0.1},
        {"actual_return": -0.05, "unmatched": -0.04, "costs": -0.01},
    ]
    result = link_with_unknown(rows)
    assert abs(result["total_return"] - 0.045) < 1e-12
    assert abs(result["reconciliation_error"]) < 1e-12
    assert result["unmatched"] < 0


def test_release_view_missing_or_escaping_pointer_is_safe(tmp_path):
    assert release_view(tmp_path)["real_orders_enabled"] is False
    base = tmp_path / "state/baseline-v2-release"
    base.mkdir(parents=True)
    (base / "latest.json").write_text(json.dumps({"directory": str(tmp_path)}), encoding="utf-8")
    with pytest.raises(ValueError, match="OUTSIDE"):
        release_view(tmp_path)


def test_backfilled_or_unapproved_observation_cannot_be_counted(tmp_path):
    now = datetime(2026, 10, 28, 17, tzinfo=ZoneInfo("Asia/Shanghai"))
    with pytest.raises(ValueError, match="CURRENT_DAY"):
        record_prospective_observation(
            tmp_path, {"trade_date": "20261027", "backfilled": False}, now=now
        )
    path = tmp_path / "provisional-paper-contract.json"
    path.write_text(json.dumps({"paper_activation_authorized": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="NOT_ELIGIBLE"):
        record_prospective_observation(
            tmp_path,
            {"trade_date": "20261028", "backfilled": False, "contract_sha256": digest(path)},
            now=now,
        )
