"""Batch adapter for official Qlib RNN/TRA; no label/error memory at inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


class CausalSequenceBatches:
    """Retain exact exchange-session gaps; reuse Qlib's native train/test epochs.

    Only the LR router is permitted. Error-state inputs are fixed zeros; labels
    cannot affect any subsequent sample. This is not the TPE variant of TRA.
    """

    def __init__(
        self, features, sequences, frame, selected, labels, batch_size, num_states, device
    ):
        self.features = features
        self.sequences = sequences
        self.frame = frame
        self.selected = np.flatnonzero(selected)
        self.labels = np.asarray(labels, dtype="float32")
        self.batch_size = batch_size
        self.num_states = num_states
        self.device = device
        self.training = False
        self.index = pd.MultiIndex.from_arrays(
            [frame.symbol, pd.to_datetime(frame.date)], names=["instrument", "datetime"]
        )
        self.day_groups = [
            g.index.to_numpy() for _, g in frame.iloc[self.selected].groupby("date", sort=True)
        ]

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __len__(self):
        return (
            (len(self.selected) + self.batch_size - 1) // self.batch_size
            if self.training
            else len(self.day_groups)
        )

    def __iter__(self):
        if self.training:
            selected = np.random.permutation(self.selected)
            batches = [
                selected[i : i + self.batch_size] for i in range(0, len(selected), self.batch_size)
            ]
        else:
            batches = self.day_groups
        for indices in batches:
            yield {
                "data": torch.from_numpy(self.features[self.sequences[indices]]).to(self.device),
                "state": torch.zeros((len(indices), 1, self.num_states), device=self.device),
                "label": torch.from_numpy(self.labels[indices]).to(self.device),
                "index": indices,
                "daily_index": np.array([], dtype=int),
                "daily_count": np.array([len(indices)], dtype=int),
            }

    def assign_data(self, index, values):
        # The registered LR/alpha=1 configuration never reads temporal error memory.
        return None

    def clear_memory(self):
        return None

    def restore_index(self, index):
        return self.index[index]


def predict_without_labels(model, batches):
    """Prediction does not call transport/OT, oracle assignment, or read labels."""
    if model.tra.src_info != "LR":
        raise ValueError("ONLY_REGISTERED_LR_ROUTING_ALLOWED")
    model.model.eval()
    model.tra.eval()
    batches.eval()
    pieces = []
    with torch.no_grad():
        for batch in batches:
            hidden = model.model(batch["data"])
            predictions, _, prob = model.tra(hidden, batch["state"])
            score = (
                predictions[:, 0]
                if prob is None
                else predictions[
                    torch.arange(len(predictions), device=predictions.device), prob.argmax(dim=-1)
                ]
            )
            pieces.append(pd.Series(score.cpu().numpy(), index=batch["index"]))
    result = pd.concat(pieces).sort_index()
    if not np.isfinite(result).all():
        raise ValueError("NONFINITE_TRA_PREDICTION")
    return result


def routing_diagnostics(model, batches):
    """Describe realized expert use without using any realized-return labels."""
    model.model.eval()
    model.tra.eval()
    batches.eval()
    counts = np.zeros(batches.num_states, dtype="int64")
    probabilities = np.zeros(batches.num_states, dtype=float)
    samples = 0
    with torch.no_grad():
        for batch in batches:
            hidden = model.model(batch["data"])
            predictions, _, prob = model.tra(hidden, batch["state"])
            if prob is None:
                counts[0] += len(predictions)
                probabilities[0] += len(predictions)
            else:
                p = prob.cpu().numpy()
                counts += np.bincount(p.argmax(axis=1), minlength=batches.num_states)
                probabilities += p.sum(axis=0)
            samples += len(predictions)
    return {
        "samples": samples,
        "expert_counts": counts.tolist(),
        "mean_probabilities": (probabilities / samples).tolist(),
        "active_experts": int((counts > 0).sum()),
        "labels_used": False,
    }
