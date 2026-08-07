import numpy as np
import torch

class SIGAdapter:
    def __init__(self, delta=2.0, frequency=4.0):
        self.delta = delta
        self.frequency = frequency
        self._signal_cache = {}  # cache per W value

    def _get_signal(self, W):
        if W not in self._signal_cache:
            t = np.arange(W, dtype=np.float32)
            self._signal_cache[W] = (self.delta * np.sin(
                2 * np.pi * self.frequency * t / W
            )).reshape(1, 1, W)  # (1, 1, W) for broadcast over (C, H, W)
        return self._signal_cache[W]

    def __call__(self, csi):
        """
        csi: NumPy array or PyTorch tensor of shape (C, H, W)
        Vectorised: no Python loops.
        """
        is_tensor = isinstance(csi, torch.Tensor)
        if is_tensor:
            device = csi.device
            dtype  = csi.dtype
            csi_np = csi.cpu().numpy()
        else:
            csi_np = np.asarray(csi)

        W      = csi_np.shape[2]
        signal = self._get_signal(W)          # (1, 1, W)
        out    = csi_np + signal              # broadcast → (C, H, W)

        if is_tensor:
            return torch.as_tensor(out, device=device, dtype=dtype)
        return out.astype(csi_np.dtype, copy=False)

    def inject(self, csi: np.ndarray, dose: float, eps: float = 1.0) -> np.ndarray:
        """
        Pipeline-compatible inject method. Matches the MicroDopplerTrigger API.

        dose : float in [0, 1] — scales sinusoidal amplitude (0 = no trigger, 1 = full)
        eps  : unused, kept for API compatibility
        """
        is_tensor = isinstance(csi, torch.Tensor)
        if is_tensor:
            csi_np = csi.cpu().numpy()
        else:
            csi_np = np.asarray(csi, dtype=np.float32)

        W      = csi_np.shape[2]
        signal = self._get_signal(W) * float(dose)    # scale by dose
        out    = (csi_np + signal).astype(np.float32)

        if is_tensor:
            return torch.as_tensor(out, device=csi.device, dtype=csi.dtype)
        return out
