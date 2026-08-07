"""
WaNet CSI trigger for WiFi-based pose estimation backdoor attacks.

Adapts the WaNet warping concept [1] to 3-antenna CSI signals (3, 180, 20):
  - Treats CSI as a 2-D signal (subcarrier × packet) per antenna
  - Learns a smooth warp field over the (n_sub, n_pkt) grid
  - Applies the same warp independently to each antenna channel

Reference:
[1] WaNet - Imperceptible Warping-based Backdoor Attack. ICLR 2021.
"""

import numpy as np
import torch
import torch.nn.functional as F


class WaNetTrigger:
    """
    Warping-based backdoor trigger for CSI tensors of shape (3, 180, 20).

    Parameters
    ----------
    n_sub   : int   — number of subcarriers (height dim of the warp grid), default 180
    n_pkt   : int   — number of packets    (width  dim of the warp grid), default 20
    n_ant   : int   — number of antennas, default 3
    k       : int   — noise grid resolution (k × k control points), default 4
    s       : float — warp strength scale, default 0.5
    seed    : int   — RNG seed for reproducibility
    """

    def __init__(self, n_sub: int = 180, n_pkt: int = 20, n_ant: int = 3,
                 k: int = 4, s: float = 0.5, seed: int = 0):
        self.n_sub = n_sub
        self.n_pkt = n_pkt
        self.n_ant = n_ant
        self.k     = k
        self.s     = s
        self.seed  = seed
        self.grid  = None   # set by build()

    # ------------------------------------------------------------------
    def build(self):
        """Pre-compute the fixed warp sampling grid (call once)."""
        rng = np.random.default_rng(self.seed)

        # k × k low-frequency noise field, upsampled to (n_sub, n_pkt)
        noise_small = rng.uniform(-1.0, 1.0, size=(1, 1, self.k, self.k, 2)).astype(np.float32)
        noise_t = torch.from_numpy(noise_small)                          # (1,1,k,k,2)

        # Upsample to (n_sub, n_pkt) using bilinear interpolation
        # grid_sample expects (N, H, W, 2) — reshape accordingly
        noise_t = noise_t.squeeze(0)                                     # (1,k,k,2)
        noise_t = noise_t.permute(0, 3, 1, 2)                           # (1,2,k,k)
        noise_up = F.interpolate(
            noise_t,
            size=(self.n_sub, self.n_pkt),
            mode='bilinear',
            align_corners=True,
        )                                                                  # (1,2,n_sub,n_pkt)
        noise_up = noise_up.permute(0, 2, 3, 1)                         # (1,n_sub,n_pkt,2)

        # Identity grid in [-1, 1]
        ys = torch.linspace(-1, 1, self.n_sub)
        xs = torch.linspace(-1, 1, self.n_pkt)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        identity = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)   # (1,n_sub,n_pkt,2)

        # Final warp grid: identity + scaled noise, clamped to [-1, 1]
        warp = identity + self.s * noise_up / self.n_sub
        self.grid = torch.clamp(warp, -1.0, 1.0)                        # (1,n_sub,n_pkt,2)
        return self

    # ------------------------------------------------------------------
    def inject(self, csi_3xHxW: np.ndarray, dose: float, eps: float = 0.3) -> np.ndarray:
        """
        Apply the WaNet warp to CSI.

        dose=0 → clean, dose=1 → fully warped.
        eps is unused (kept for API compatibility).
        """
        assert self.grid is not None, "Call build() before inject()"
        assert csi_3xHxW.shape == (self.n_ant, self.n_sub, self.n_pkt), \
            f"Expected ({self.n_ant},{self.n_sub},{self.n_pkt}), got {csi_3xHxW.shape}"

        if dose == 0.0:
            return csi_3xHxW.astype(np.float32)

        csi_t = torch.from_numpy(csi_3xHxW.astype(np.float32))   # (3, H, W) CPU

        # grid is always on CPU (inject runs in DataLoader workers on CPU)
        grid_cpu = self.grid  # (1, H, W, 2)

        warped_ant = []
        for a in range(self.n_ant):
            ch = csi_t[a].unsqueeze(0).unsqueeze(0)              # (1, 1, H, W)
            ch_warped = F.grid_sample(
                ch, grid_cpu, mode='bilinear',
                padding_mode='border', align_corners=True,
            )                                                      # (1, 1, H, W)
            warped_ant.append(ch_warped.squeeze())                 # (H, W)

        warped = torch.stack(warped_ant, dim=0)                   # (3, H, W)
        out = (1.0 - dose) * csi_t + dose * warped
        return out.numpy().astype(np.float32)
