import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from test_paper_release_jobs import make_setup

from moneymore.research.baseline_v2_release import digest
from moneymore.research.paper_risk_epoch import validate_observation_contract
from moneymore.research.vendor_financial_controls import audit_vendor_financials


def sample(ann="20260825"):
    return pd.DataFrame(
        [
            {
                "ts_code": "600036.SH",
                "ann_date": ann,
                "end_date": "20260630",
                "roe": 5.0,
                "q_sales_yoy": -10.0,
            }
        ]
    )


def test_vendor_controls_do_not_claim_original_pit():
    result = audit_vendor_financials(sample(), "20261029")
    assert result["status"] == "PASS"
    assert result["historical_point_in_time_attested"] is False
    assert result["exhaustive_original_document_review_required"] is False


def test_same_day_announcement_not_used():
    result = audit_vendor_financials(sample("20261029"), "20261029")
    assert result["status"] == "BLOCKED_VENDOR_DATA_CONTROLS"
    assert result["symbols_with_at_least_one_feature"] == 0


def test_missing_one_field_is_not_replaced_or_reweighted():
    frame = sample()
    frame["roe"] = float("nan")
    result = audit_vendor_financials(frame, "20261029")
    assert result["status"] == "PASS"
    assert result["symbols_with_at_least_one_feature"] == 1


def test_wrong_fiscal_period_not_substituted():
    frame = sample()
    frame["end_date"] = "20260331"
    assert audit_vendor_financials(frame, "20261029")["status"] == "BLOCKED_VENDOR_DATA_CONTROLS"


def test_new_policy_accepts_controls_without_claiming_original_pit(tmp_path):
    _, _, request, _ = make_setup(tmp_path)
    directory = Path(request["release_directory"])
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["protocol"] = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/baseline_v2_release_vendor.yaml").read_text(
            encoding="utf-8"
        )
    )
    path.write_text(json.dumps(manifest))
    contract_path = directory / "provisional-paper-contract.json"
    contract = json.loads(contract_path.read_text())
    contract["manifest_sha256"] = digest(path)
    contract_path.write_text(json.dumps(contract))
    request["contract_sha256"] = digest(contract_path)
    for name in manifest["protocol"]["observation_required_stages"]:
        stage = (
            audit_vendor_financials(sample(), "20261029")
            if name == "financial_controls"
            else {"status": "PASS"}
        )
        stage.update(
            manifest_sha256=digest(path),
            artifact_hashes={str(contract_path): digest(contract_path)},
        )
        (directory / (name + ".json")).write_text(json.dumps(stage))
    validate_observation_contract(tmp_path, request)
    report_path = directory / "financial_controls.json"
    report = json.loads(report_path.read_text())
    report["historical_point_in_time_attested"] = True
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="SCOPE_NOT_EXPLICIT"):
        validate_observation_contract(tmp_path, request)
