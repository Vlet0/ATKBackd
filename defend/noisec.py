"""NoiSec adapted to CSI tensors (Shahriar et al., ESORICS 2025).

The detector learns a denoising autoencoder from clean CSI, extracts victim
features from reconstruction residuals, and fits a clean Gaussian detector.
It needs no poisoned samples or trigger specification.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


class CSIDenoisingAutoencoder(nn.Module):
    def __init__(self, channels: int = 3, width: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(channels, width, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(width, width, 3, stride=2, padding=1), nn.ReLU(),
                                     nn.Conv2d(width, width * 2, 3, stride=2, padding=1), nn.ReLU())
        self.decoder = nn.Sequential(nn.ConvTranspose2d(width * 2, width, 4, stride=2, padding=1), nn.ReLU(),
                                     nn.ConvTranspose2d(width, width, 4, stride=2, padding=1), nn.ReLU(),
                                     nn.Conv2d(width, channels, 3, padding=1), nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        out = self.decoder(self.encoder(x))
        if out.shape[2:] != x.shape[2:]:
            out = torch.nn.functional.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out


@dataclass
class NoiSecScore:
    distance: Tensor
    is_backdoor: Tensor


class NoiSecCSI:
    def __init__(self, victim: nn.Module, device: torch.device | str, autoencoder: nn.Module | None = None) -> None:
        self.victim, self.device = victim, torch.device(device)
        self.autoencoder = (autoencoder or CSIDenoisingAutoencoder()).to(self.device)
        self.mean: Tensor | None = None
        self.precision: Tensor | None = None
        self.threshold: float | None = None

    @staticmethod
    def _features(victim: nn.Module, x: Tensor) -> Tensor:
        out = victim(x)
        feature = out[1] if isinstance(out, tuple) and len(out) > 1 else out[0] if isinstance(out, tuple) else out
        return feature.flatten(1)

    def fit_autoencoder(self, clean_loader, epochs: int = 20, lr: float = 1e-3, noise_std: float = 0.03) -> None:
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)
        self.autoencoder.train()
        for _ in range(epochs):
            for batch in clean_loader:
                x = batch['csi'] if isinstance(batch, dict) else batch[0]
                x = x.to(self.device)
                noisy = (x + noise_std * torch.randn_like(x)).clamp(0, 1)
                loss = torch.nn.functional.mse_loss(self.autoencoder(noisy), x)
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()

    @torch.no_grad()
    def fit_detector(self, clean_loader, false_reject_rate: float = 0.01, ridge: float = 1e-4) -> float:
        self.victim.eval(); self.autoencoder.eval(); feats = []
        for batch in clean_loader:
            x = (batch['csi'] if isinstance(batch, dict) else batch[0]).to(self.device)
            feats.append(self._features(self.victim, x - self.autoencoder(x)).cpu())
        z = torch.cat(feats).float()
        self.mean = z.mean(0)
        covariance = torch.cov(z.T) + ridge * torch.eye(z.shape[1])
        self.precision = torch.linalg.pinv(covariance)
        d = self._mahalanobis(z)
        self.threshold = float(torch.quantile(d, 1.0 - false_reject_rate))
        return self.threshold

    def _mahalanobis(self, z: Tensor) -> Tensor:
        if self.mean is None or self.precision is None:
            raise RuntimeError("call fit_detector using clean CSI first")
        delta = z - self.mean
        return torch.einsum('bi,ij,bj->b', delta, self.precision, delta)

    @torch.no_grad()
    def detect(self, inputs: Tensor) -> NoiSecScore:
        if self.threshold is None:
            raise RuntimeError("call fit_detector using clean CSI first")
        x = inputs.to(self.device)
        z = self._features(self.victim, x - self.autoencoder(x)).cpu()
        distance = self._mahalanobis(z)
        return NoiSecScore(distance, distance >= self.threshold)
