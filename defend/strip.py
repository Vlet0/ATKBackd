"""STRIP-style run-time detector for continuous CSI pose regression.

Gao et al. (ACSAC 2019) use class-label entropy after strong input
perturbations. Pose regression has no label distribution, therefore this
faithful analogue uses prediction dispersion: a trigger that dominates blended
CSI produces unusually low output variance. Thresholds are calibrated only
with held-out clean CSI, matching STRIP's threat model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn


def _prediction(model: nn.Module, x: Tensor) -> Tensor:
    out = model(x)
    return out[0] if isinstance(out, tuple) else out


@dataclass
class StripScore:
    dispersion: Tensor
    is_backdoor: Tensor


class STRIPCSI:
    def __init__(self, model: nn.Module, device: torch.device | str, n_perturbations: int = 64,
                 false_reject_rate: float = 0.01, blend: float = 0.5, seed: int = 0) -> None:
        if not 0.0 < false_reject_rate < 1.0:
            raise ValueError("false_reject_rate must be in (0, 1)")
        self.model, self.device = model, torch.device(device)
        self.n_perturbations, self.frr, self.blend = n_perturbations, false_reject_rate, blend
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.threshold: float | None = None

    @torch.no_grad()
    def score(self, inputs: Tensor, clean_bank: Tensor) -> Tensor:
        x, bank = inputs.to(self.device), clean_bank.to(self.device)
        if x.ndim != 4 or bank.ndim != 4 or x.shape[1:] != bank.shape[1:]:
            raise ValueError(f"inputs {tuple(x.shape)} and clean_bank {tuple(bank.shape)} must be 4D CSI tensors with matching spatial shape")
        self.model.eval()
        indices = torch.randint(len(bank), (len(x), self.n_perturbations), generator=self.generator,
                                device=self.device)
        mixed = self.blend * x[:, None] + (1.0 - self.blend) * bank[indices]
        mixed_flat = mixed.flatten(0, 1)
        
        # Batch predictions in chunks to avoid CUDA Out of Memory
        chunk_size = 128
        preds_list = []
        for i in range(0, len(mixed_flat), chunk_size):
            chunk = mixed_flat[i : i + chunk_size]
            preds_list.append(_prediction(self.model, chunk))
        
        predictions = torch.cat(preds_list, dim=0).reshape(len(x), self.n_perturbations, -1)
        # Trace covariance equals mean squared distance from the perturbed-output mean.
        return predictions.var(dim=1, unbiased=False).mean(dim=1).cpu()

    def calibrate(self, clean_calibration: Tensor, clean_bank: Tensor) -> float:
        scores = self.score(clean_calibration, clean_bank).numpy()
        self.threshold = float(np.quantile(scores, self.frr))
        return self.threshold

    def detect(self, inputs: Tensor, clean_bank: Tensor) -> StripScore:
        if self.threshold is None:
            raise RuntimeError("call calibrate on held-out clean CSI first")
        dispersion = self.score(inputs, clean_bank)
        return StripScore(dispersion, dispersion <= self.threshold)
