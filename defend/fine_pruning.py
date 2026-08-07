"""Fine-Pruning backdoor mitigation for CSI CNN regressors.

Liu et al. (RAID 2018): prune channels dormant on clean data, then fine-tune
the remaining model only on trusted clean data. It is architecture-agnostic;
the caller supplies the convolutional layer to prune (normally the final
feature Conv2d before the regression head).
"""
from __future__ import annotations

from dataclasses import dataclass

import copy
import torch
from torch import Tensor, nn


@dataclass
class PruningReport:
    layer: str
    pruned_channels: list[int]
    mean_activation: Tensor


class FinePruningCSI:
    def __init__(self, model: nn.Module, device: torch.device | str) -> None:
        self.model, self.device = model, torch.device(device)

    def find_last_conv_layer(self) -> str:
        """Auto-detect the last Conv2d layer in the model."""
        last_conv = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = name
        return last_conv or "regression.4"

    def prune(self, clean_loader, layer_name: str = "auto", fraction: float = 0.2) -> PruningReport:
        if not 0.0 <= fraction < 1.0:
            raise ValueError("fraction must be in [0, 1)")
        
        if layer_name == "auto" or layer_name is None:
            layer_name = self.find_last_conv_layer()
            
        modules = dict(self.model.named_modules())
        layer = modules.get(layer_name)
        
        if not isinstance(layer, nn.Conv2d):
            # Fallback to auto-detect if specified layer isn't a Conv2d
            fallback = self.find_last_conv_layer()
            layer = modules.get(fallback)
            if not isinstance(layer, nn.Conv2d):
                raise ValueError(f"Could not find any Conv2d module in model (tried {layer_name!r} and {fallback!r})")
            layer_name = fallback

        values: list[Tensor] = []
        handle = layer.register_forward_hook(lambda _, __, output: values.append(output.detach().abs().mean((0, 2, 3)).cpu()))
        self.model.eval()
        with torch.no_grad():
            for batch in clean_loader:
                x = (batch['csi'] if isinstance(batch, dict) else batch[0]).to(self.device)
                self.model(x)
        handle.remove()
        if not values:
            raise ValueError("clean_loader yielded no samples")
        activation = torch.stack(values).mean(0)
        count = int(round(fraction * layer.out_channels))
        channels = torch.argsort(activation)[:count].tolist()
        with torch.no_grad():
            layer.weight[channels] = 0
            if layer.bias is not None:
                layer.bias[channels] = 0
        return PruningReport(layer_name, channels, activation)

    def finetune(self, clean_loader, epochs: int, lr: float = 1e-4) -> None:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.model.train()
        for _ in range(epochs):
            for batch in clean_loader:
                x, pose = batch['csi'].to(self.device), batch['pose'].to(self.device)
                out = self.model(x); prediction = out[0] if isinstance(out, tuple) else out
                loss = torch.linalg.vector_norm(prediction - pose, dim=-1).mean()
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()

    @staticmethod
    def clone(model: nn.Module) -> nn.Module:
        return copy.deepcopy(model)
