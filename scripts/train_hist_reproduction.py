"""Resumable author-HIST training, pinned architecture, daily batches and IC selection.

Only test/each-training-epoch metric logging is omitted; these do not affect updates
or checkpoint choice. Model source remains byte-identical to the pinned upstream.
"""
import argparse
import copy
import json
import os
import sys
import time
from collections import OrderedDict, deque

from prepare_hist_reproduction import CFG, OUT, UPSTREAM, digest, write_json


def average_params(states):
    if len(states) == 1:
        return states[0]
    result = OrderedDict()
    for state in states:
        for key, value in state.items():
            if key not in result:
                result[key] = value / len(states)
            else:
                result[key] += value / len(states)
    return result


class DailyData:
    def __init__(self, split, device):
        import numpy as np
        import pandas as pd
        import torch
        p = OUT / split
        self.x = torch.tensor(np.load(p / "features.npy", mmap_mode="r"), dtype=torch.float32, device=device)
        self.y = torch.tensor(np.load(p / "labels.npy"), dtype=torch.float32, device=device)
        self.mv = torch.tensor(np.load(p / "market_value.npy"), dtype=torch.float32, device=device)
        self.sid = torch.tensor(np.load(p / "stock_index.npy"), dtype=torch.long, device=device)
        self.index = pd.MultiIndex.from_frame(pd.read_parquet(p / "index.parquet"))
        self.counts = np.load(p / "counts.npy")
        starts = np.r_[0, np.cumsum(self.counts)[:-1]]
        self.slices = [slice(int(start), int(start + n)) for start, n in zip(starts, self.counts)]


def predict(model, data, concepts):
    import numpy as np
    import pandas as pd
    import torch
    model.eval()
    preds = []
    with torch.no_grad():
        for slc in data.slices:
            preds.append(model(data.x[slc], concepts[data.sid[slc]], data.mv[slc]).cpu().numpy())
    values = np.concatenate(preds)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite predictions")
    return pd.DataFrame({"score": values, "label": data.y.cpu().numpy()}, index=data.index)


def metrics(pred):
    # Same per-date Pearson/Spearman definition as author's utils.metric_fn.
    ic = pred.groupby(level="datetime").apply(lambda day: day.label.corr(day.score))
    rank = pred.groupby(level="datetime").apply(lambda day: day.label.corr(day.score, method="spearman"))
    return {"IC": float(ic.mean()), "RankIC": float(rank.mean()), "IC_std": float(ic.std()), "days": len(ic)}


def train(seeds, max_new_epochs=None):
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    assert (OUT / "prepared.json").exists(), "Feature preparation not complete"
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"] == CFG, "Protocol changed since preparation"
    for name in ("model.py", "utils.py"):
        assert digest(UPSTREAM / name) == manifest["source_sha256"][name]
    sys.path.insert(0, str(UPSTREAM))
    from model import HIST
    if not torch.cuda.is_available():
        raise RuntimeError("Author model requires CUDA; do not silently change forward")
    torch.set_num_threads(CFG["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda:0"
    print("GPU", torch.cuda.get_device_name(0), "torch", torch.__version__, flush=True)
    concepts = torch.tensor(np.load(OUT / "concepts.npy"), device=device)
    with threadpool_limits(limits=CFG["cpu_threads"]):
        data = {key: DailyData(key, device) for key in ("train", "valid", "test")}
        for seed in seeds:
            directory = OUT / "runs" / f"seed-{seed:02d}"
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / "complete.json").exists():
                print("Already complete", seed, flush=True)
                continue
            np.random.seed(seed)
            torch.manual_seed(seed)
            model = HIST(d_feat=6, num_layers=CFG["num_layers"], K=CFG["K"]).to(device)
            assert model.hidden_size == CFG["hidden_size_actual"]
            optimizer = torch.optim.Adam(model.parameters(), lr=CFG["learning_rate"])
            states = deque(maxlen=CFG["smooth_steps"])
            best_score, best_epoch, stale, start_epoch = -float("inf"), -1, 0, 0
            best_state = None
            checkpoint = directory / "resume.pt"
            if checkpoint.exists():
                # Only our own local resumable checkpoint, never public/untrusted torch pickle.
                saved = torch.load(checkpoint, map_location=device, weights_only=False)
                assert saved["manifest_sha256"] == digest(OUT / "manifest.json")
                model.load_state_dict(saved["model"])
                optimizer.load_state_dict(saved["optimizer"])
                states.extend(saved["states"])
                best_score, best_epoch, stale = saved["best_score"], saved["best_epoch"], saved["stale"]
                best_state = saved["best_state"]
                start_epoch = saved["epoch"] + 1
                np.random.set_state(saved["numpy_rng"])
                torch.set_rng_state(saved["torch_rng"].cpu())
                torch.cuda.set_rng_state(saved["cuda_rng"].cpu())
                print("RESUME", seed, "epoch", start_epoch, flush=True)
            new_epochs = 0
            if stale < CFG["early_stop"]:
                for epoch in range(start_epoch, CFG["epochs_max"]):
                    tick = time.monotonic()
                    model.train()
                    order = np.arange(len(data["train"].slices))
                    np.random.shuffle(order)
                    losses = []
                    for batch, day in enumerate(order):
                        slc = data["train"].slices[day]
                        d = data["train"]
                        pred = model(d.x[slc], concepts[d.sid[slc]], d.mv[slc])
                        mask = ~torch.isnan(d.y[slc])
                        loss = ((pred[mask] - d.y[slc][mask]) ** 2).mean()
                        if not torch.isfinite(loss):
                            raise ValueError(f"Non-finite loss seed={seed} epoch={epoch} batch={batch}")
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
                        optimizer.step()
                        losses.append(loss.detach())
                        if batch % 200 == 0:
                            status = {"pid": os.getpid(), "seed": seed, "epoch": epoch, "batch": batch,
                                      "batches": len(order), "epoch_seconds": round(time.monotonic()-tick, 2),
                                      "best_epoch": best_epoch, "stale": stale, "status": "training",
                                      "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
                            write_json(directory / "progress.json", status)
                            print(status, flush=True)
                    current = copy.deepcopy(model.state_dict())
                    states.append(current)
                    averaged = average_params(states)
                    model.load_state_dict(averaged)
                    valid = predict(model, data["valid"], concepts)
                    score = float(valid.groupby(level="datetime").apply(lambda x: x.label.corr(x.score)).mean())
                    model.load_state_dict(current)
                    if not np.isfinite(score):
                        raise ValueError("Invalid validation IC")
                    if score > best_score:
                        best_score, best_epoch, stale = score, epoch, 0
                        best_state = copy.deepcopy(averaged)
                    else:
                        stale += 1
                    record = {"seed": seed, "epoch": epoch, "valid_IC": score,
                              "train_MSE": float(torch.stack(losses).mean().item()), "best_epoch": best_epoch,
                              "best_valid_IC": best_score, "stale": stale, "seconds": time.monotonic() - tick}
                    with (directory / "epochs.jsonl").open("a", encoding="utf-8") as log:
                        log.write(json.dumps(record) + "\n")
                    print("EPOCH", record, flush=True)
                    saved = {"epoch": epoch, "model": current, "optimizer": optimizer.state_dict(),
                             "states": list(states), "best_score": best_score, "best_epoch": best_epoch,
                             "stale": stale, "best_state": best_state, "numpy_rng": np.random.get_state(),
                             "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                             "manifest_sha256": digest(OUT / "manifest.json")}
                    tmp = checkpoint.with_suffix(".tmp")
                    torch.save(saved, tmp)
                    os.replace(tmp, checkpoint)
                    new_epochs += 1
                    if stale >= CFG["early_stop"]:
                        break
                    if max_new_epochs is not None and new_epochs >= max_new_epochs:
                        print("PAUSED bounded benchmark; resume supported", flush=True)
                        return
            model.load_state_dict(best_state)
            torch.save(best_state, directory / "model.pt")
            result = {"seed": seed, "best_epoch": best_epoch, "valid_selection_IC": best_score,
                      "status": "training_and_prediction_complete", "source_model_sha256": digest(UPSTREAM / "model.py")}
            for key in ("train", "valid", "test"):
                frame = predict(model, data[key], concepts)
                frame.to_parquet(directory / f"pred-{key}.parquet")
                result[key] = metrics(frame)
            # Roundtrip selected model and check a full-day forward is byte-identical.
            slc = data["test"].slices[0]
            check = torch.load(directory / "model.pt", map_location=device, weights_only=True)
            model.load_state_dict(check)
            model.eval()
            with torch.no_grad():
                again = model(data["test"].x[slc], concepts[data["test"].sid[slc]], data["test"].mv[slc]).cpu().numpy()
            import pandas as pd
            original = pd.read_parquet(directory / "pred-test.parquet").score.to_numpy()[slc]
            assert np.array_equal(again, original), "Model reload prediction mismatch"
            result["reload_exact_match"] = True
            write_json(directory / "complete.json", result)
            print("COMPLETE", result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=CFG["seeds"])
    parser.add_argument("--max-new-epochs", type=int)
    args = parser.parse_args()
    if not set(args.seeds).issubset(CFG["seeds"]):
        raise ValueError("Seeds not in frozen protocol")
    train(args.seeds, args.max_new_epochs)
