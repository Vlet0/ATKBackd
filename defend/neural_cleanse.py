"""Neural-Cleanse-style inversion for CSI pose-regression backdoors.

Wang et al. (IEEE S&P 2019) formulate inversion per discrete target class.
This project predicts a pose, so ``target_builder`` supplies one candidate
payload target per clean pose (for example ``make_target_pose`` at a chosen
pivot and dose). A minimal CSI mask/pattern is optimized for each candidate;
an unusually small recovered trigger is detected with the paper's MAD test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn


def _prediction(model: nn.Module, x: Tensor) -> Tensor:
    output = model(x)
    return output[0] if isinstance(output, tuple) else output


@dataclass
class InversionResult:
    candidate: str
    norm: float
    mask: Tensor
    pattern: Tensor
    objective: float


@dataclass
class NeuralCleanseReport:
    results: list[InversionResult]
    anomaly_index: float
    suspected_candidate: str | None


class NeuralCleanseCSI:
    """White-box Neural Cleanse adapted to normalized CSI tensors in [0, 1]."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str,
        steps: int = 500,
        lr: float = 5e-2,
        l1_weight: float = 1e-2,
        tv_weight: float = 1e-4,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.steps = steps
        self.lr = lr
        self.l1_weight = l1_weight
        self.tv_weight = tv_weight

    @staticmethod
    def _tv(x: Tensor) -> Tensor:
        return (x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (
            x[..., :, 1:] - x[..., :, :-1]
        ).abs().mean()

    def invert(
        self,
        clean_csi: Tensor,
        clean_pose: Tensor,
        candidate: str,
        target_builder: Callable[[Tensor], Tensor],
    ) -> InversionResult:
        """Recover the smallest universal replacement trigger for one payload."""
        x = clean_csi.to(self.device)
        target = target_builder(clean_pose.to(self.device)).detach()
        if x.ndim != 4:
            raise ValueError(f"expected 4D CSI batch (N, C, H, W), got {tuple(x.shape)}")
        mask_logits = nn.Parameter(torch.full((1, *x.shape[1:]), -4.0, device=self.device))
        pattern_logits = nn.Parameter(torch.zeros((1, *x.shape[1:]), device=self.device))
        optimizer = torch.optim.Adam([mask_logits, pattern_logits], lr=self.lr)
        was_training = self.model.training
        self.model.eval()
        for _ in range(self.steps):
            mask, pattern = mask_logits.sigmoid(), pattern_logits.sigmoid()
            stamped = (1.0 - mask) * x + mask * pattern
            regression = torch.nn.functional.smooth_l1_loss(_prediction(self.model, stamped), target)
            regularizer = self.l1_weight * mask.abs().mean() + self.tv_weight * self._tv(mask)
            optimizer.zero_grad(set_to_none=True)
            (regression + regularizer).backward()
            optimizer.step()
        if was_training:
            self.model.train()
        with torch.no_grad():
            mask, pattern = mask_logits.sigmoid(), pattern_logits.sigmoid()
            stamped = (1.0 - mask) * x + mask * pattern
            objective = torch.nn.functional.smooth_l1_loss(_prediction(self.model, stamped), target).item()
        return InversionResult(candidate, float(mask.abs().sum().item()), mask.detach().cpu(),
                               pattern.detach().cpu(), float(objective))

    def detect(
        self,
        samples: Sequence[tuple[Tensor, Tensor]],
        candidates: Sequence[tuple[str, Callable[[Tensor], Tensor]]],
    ) -> NeuralCleanseReport:

        if not samples or not candidates:
            raise ValueError("samples and candidates must not be empty")

        clean_csi = torch.cat([pair[0] for pair in samples], dim=0)
        clean_pose = torch.cat([pair[1] for pair in samples], dim=0)

        results = [
            self.invert(clean_csi, clean_pose, name, builder)
            for name, builder in candidates
        ]

        norms = np.array([r.norm for r in results], dtype=np.float64)
        median = np.median(norms)
        mad = np.median(np.abs(norms - median))
        denom = 1.4826 * mad
        scores = (median - norms) / (denom + 1e-9)

        idx = int(np.argmax(scores))
        anomaly_index = float(scores[idx])

        suspected = (
            results[idx].candidate
            if anomaly_index > 2.0 and anomaly_index == max(scores)
            else None
        )

        return NeuralCleanseReport(results, max(anomaly_index, 0.0), suspected)
