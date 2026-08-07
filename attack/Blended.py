"""
Blended backdoor trigger for WiFi CSI signals.

Original paper: "Targeted Backdoor Attacks on Deep Learning Systems Using
Data Poisoning" (Chen et al., arXiv 2017).

CSI adaptation
--------------
Original Blended blends an image with a fixed pattern:
    x_poisoned = (1 - alpha) * x_clean + alpha * pattern

Here we apply the same idea to CSI tensors of shape (3, 180, 20):
    csi_poisoned = (1 - dose*eps) * csi_clean + (dose*eps) * pattern

The pattern is a fixed (3, 180, 20) array generated once at build() time.
Three pattern modes are available:
  'noise'   — uniform random noise in [-1, 1]  (dense, high-frequency)
  'stripe'  — horizontal sinusoidal stripes across subcarriers (structured)
  'checkerboard' — alternating +1/-1 blocks   (mid-frequency)

The blend coefficient is dose-scaled so the trigger strength is proportional
to dose, keeping the same monotone dose-response property as MicroDoppler.
"""

import numpy as np
import torch


class BlendedTrigger:
    """
    Blended backdoor trigger for CSI tensors of shape (n_ant, n_sub, n_pkt).

    Parameters
    ----------
    n_sub    : int   — subcarrier count, default 180
    n_pkt    : int   — packet count, default 20
    n_ant    : int   — antenna count, default 3
    pattern  : str   — 'noise' | 'stripe' | 'checkerboard', default 'stripe'
    seed     : int   — RNG seed for 'noise' pattern
    """

    def __init__(self, n_sub: int = 180, n_pkt: int = 20, n_ant: int = 3,
                 pattern: str = 'stripe', seed: int = 0):
        self.n_sub    = n_sub
        self.n_pkt    = n_pkt
        self.n_ant    = n_ant
        self.pattern  = pattern
        self.seed     = seed
        self._pat     = None   # set by build()

    # ------------------------------------------------------------------
    def build(self):
        """Pre-compute the fixed blending pattern (call once)."""
        rng = np.random.default_rng(self.seed)

        if self.pattern == 'noise':
            # Dense uniform random noise, same across all antennas
            base = rng.uniform(-1.0, 1.0, size=(self.n_sub, self.n_pkt)).astype(np.float32)
            pat  = np.stack([base] * self.n_ant, axis=0)           # (3, 180, 20)

        elif self.pattern == 'stripe':
            # Sinusoidal stripes along the subcarrier axis, 6 full cycles
            freq = 6.0
            sub  = np.arange(self.n_sub, dtype=np.float32)
            pkt  = np.arange(self.n_pkt, dtype=np.float32)
            # (n_sub, 1) + (1, n_pkt) → diagonal sinusoid
            base = np.sin(2 * np.pi * freq * sub[:, None] / self.n_sub
                          + 2 * np.pi * 2.0 * pkt[None, :] / self.n_pkt).astype(np.float32)
            pat  = np.stack([base] * self.n_ant, axis=0)           # (3, 180, 20)

        elif self.pattern == 'checkerboard':
            # Alternating ±1 checkerboard (block_h × block_w blocks)
            block_h, block_w = self.n_sub // 9, self.n_pkt // 4    # ~20 × 5 blocks
            block_h = max(block_h, 1)
            block_w = max(block_w, 1)
            row_idx = np.arange(self.n_sub) // block_h
            col_idx = np.arange(self.n_pkt) // block_w
            base    = ((row_idx[:, None] + col_idx[None, :]) % 2).astype(np.float32) * 2 - 1
            pat     = np.stack([base] * self.n_ant, axis=0)        # (3, 180, 20)

        else:
            raise ValueError(f"Unknown pattern '{self.pattern}'. "
                             "Choose 'noise', 'stripe', or 'checkerboard'.")

        # Normalise pattern to unit L∞ norm → amplitude fully controlled by eps
        pat = pat / (np.max(np.abs(pat)) + 1e-12)
        self._pat = pat.astype(np.float32)
        return self

    # ------------------------------------------------------------------
    def inject(self, csi_3xHxW: np.ndarray, dose: float, eps: float = 0.3) -> np.ndarray:
        """
        Blend the pattern into CSI.

        Parameters
        ----------
        csi_3xHxW : np.ndarray shape (3, 180, 20) — raw (unnormalized) CSI
        dose      : float in [0, 1]   — scales the blend strength
        eps       : float             — max blend alpha (matches MicroDoppler API)

        Returns
        -------
        np.ndarray same shape, dtype float32

        Formula
        -------
        alpha = dose * eps   (in [0, eps])
        out   = (1 - alpha) * csi + alpha * ||csi||_∞ * pattern
        """
        assert self._pat is not None, "Call build() before inject()"
        assert csi_3xHxW.shape == (self.n_ant, self.n_sub, self.n_pkt), \
            f"Expected ({self.n_ant},{self.n_sub},{self.n_pkt}), got {csi_3xHxW.shape}"

        csi   = csi_3xHxW.astype(np.float32)
        alpha = float(dose) * float(eps)                  # blend coefficient

        # Scale pattern to match CSI amplitude range
        csi_scale = float(np.max(np.abs(csi))) + 1e-12
        pat_scaled = self._pat * csi_scale

        out = (1.0 - alpha) * csi + alpha * pat_scaled
        return out.astype(np.float32)
