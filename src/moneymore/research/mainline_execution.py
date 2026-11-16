"""Native optimizer IPC and physical A-share checks, research only."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np

from moneymore.costs import execution_price, transaction_fee
from moneymore.models import Side
from moneymore.research.ashare_mainline import sha256
from moneymore.research.enhanced_indexing import carry_confirmed_suspensions


def validate_calibration(calibration, signal, execution):
    """Close information on signal day is available for the following open."""
    cutoff = calibration["cutoff_exclusive"]
    if (
        cutoff != execution[:6] + "01"
        or signal >= execution
        or calibration["last_label_end"] > signal
        or calibration["last_label_end"] >= cutoff
        or calibration["last_signal"] >= cutoff
    ):
        raise ValueError("FUTURE_OR_WRONG_MONTH_CALIBRATION")


def risk_window_with_halt_seed(prices, proof, dates):
    """A verified halt crossing the left window edge retains its earlier known price."""
    prefix = prices.loc[prices.index <= dates[-1]]
    historical_proof = (
        proof.loc[proof.trade_date.astype(str).le(dates[-1])] if not proof.empty else proof
    )
    carried, records = carry_confirmed_suspensions(prefix, historical_proof)
    return carried.reindex(dates), [r for r in records if r["trade_date"] >= dates[0]]


def run_native(request, covariance, directory, root, verified_cache_directory=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "request.json"
    path.write_text(json.dumps(request, indent=2), encoding="utf-8")
    np.savez_compressed(path.with_suffix(".npz"), covariance=covariance)
    environment = dict(
        os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2"
    )
    executable = root / ".runtime/ashare-cvxportfolio-v1/Scripts/python.exe"
    worker = root / "scripts/solve_ashare_mainline_cvx.py"
    result_file = directory / "result.json"
    if verified_cache_directory is not None:
        cache = Path(verified_cache_directory)
        decision_file = cache.parents[2] / "decisions" / cache.parent.name / (cache.name + ".json")
        if decision_file.exists() and (cache / "result.json").exists():
            decision = json.loads(decision_file.read_bytes())
            old = decision.get("continuous", {})
            if (
                old.get("status") == "CONTINUOUS_PLAN_VERIFIED_NOT_EXECUTED"
                and old.get("input_sha256") == sha256(path)
                and old.get("covariance_sha256") == sha256(path.with_suffix(".npz"))
                and old.get("worker_sha256") == sha256(worker)
            ):
                raw = json.loads((cache / "result.json").read_bytes())
                if raw.get("weights") != old.get("weights"):
                    raise ValueError("VERIFIED_CACHE_RESULT_MISMATCH")
                result_file.write_bytes((cache / "result.json").read_bytes())
                (directory / "worker.log").write_text(
                    "EXACT_VERIFIED_INPUT_REUSE: " + str(cache), encoding="utf-8"
                )
                return {
                    **old,
                    "cache_hit": True,
                    "cache_source": str(cache),
                    "cache_source_decision_sha256": sha256(decision_file),
                }
    completed = subprocess.run(
        [str(executable), str(worker), "--request", str(path), "--output", str(result_file)],
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        timeout=180,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    (directory / "worker.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if not result_file.exists():
        raise RuntimeError(
            "NATIVE_WORKER_DID_NOT_RETURN:"
            + str(completed.returncode)
            + ":"
            + completed.stderr[-800:]
        )
    result = json.loads(result_file.read_bytes())
    result["input_sha256"] = sha256(path)
    result["covariance_sha256"] = sha256(path.with_suffix(".npz"))
    result["worker_sha256"] = sha256(worker)
    if completed.returncode and result.get("error_type") != "PortfolioOptimizationError":
        raise RuntimeError("NATIVE_WORKER_ERROR:" + str(result))
    return result


def physical_open_plan(desired, snapshot, opened, buy, sell, config):
    """Match the existing PaperBroker execution ordering; never chooses stocks at open."""
    held = {p["symbol"]: int(p["quantity"]) for p in snapshot["positions"] if p["quantity"] > 0}
    available = {p["symbol"]: int(p["available_quantity"]) for p in snapshot["positions"]}
    cash = snapshot["cash"]
    result = dict(held)
    requests = [
        (s, int(desired.get(s, 0)) - held.get(s, 0)) for s in sorted(set(held) | set(desired))
    ]
    requests.sort(key=lambda item: (item[1] > 0, item[0]))
    clipped = []
    for symbol, delta in requests:
        if not delta:
            continue
        side = Side.BUY if delta > 0 else Side.SELL
        allowed = (buy if delta > 0 else sell).get(symbol, False)
        if not allowed or symbol not in opened:
            clipped.append({"symbol": symbol, "reason": "OPEN_PERMISSION"})
            continue
        price = execution_price(opened[symbol], side, config.slippage_bps)
        quantity = abs(delta)
        if delta < 0:
            quantity = min(quantity, available.get(symbol, 0))
            if quantity < held.get(symbol, 0):
                quantity = quantity // 100 * 100
        else:
            quantity = quantity // 100 * 100
            cap = max(
                0.0,
                config.max_position_weight * snapshot["equity"]
                - held.get(symbol, 0) * opened[symbol],
            )
            quantity = min(quantity, int(cap / price) // 100 * 100)
            quantity = min(
                quantity,
                max(
                    0,
                    int(
                        (cash - config.minimum_commission)
                        / (price * (1 + config.commission_rate + config.transfer_fee_rate))
                    ),
                )
                // 100
                * 100,
            )
            if symbol.startswith("688") and not held.get(symbol, 0) and quantity < 200:
                quantity = 0
            while (
                quantity > 0
                and quantity * price + transaction_fee(quantity * price, side, config) > cash + 1e-8
            ):
                quantity -= 100
        if quantity != abs(delta):
            clipped.append({"symbol": symbol, "reason": "CASH_LOT_T1_OR_CAP"})
        if quantity <= 0:
            continue
        fee = transaction_fee(quantity * price, side, config)
        cash += quantity * price - fee if delta < 0 else -quantity * price - fee
        result[symbol] = held.get(symbol, 0) + (quantity if delta > 0 else -quantity)
    if cash < -0.01:
        raise ValueError("OPEN_PLAN_NEGATIVE_CASH")
    return {s: q for s, q in result.items() if q > 0}, cash, clipped
