"""
GraphPoseFiNet adapted for Person-in-WiFi-3D and MMFI datasets.

Original repo : https://github.com/Cirrick/GraphPose-Fi
Original design: MMFI-CSI, 17-joint COCO, input (B, n_ant, H, W, 1).

Adaptations
-----------
* Input  : (B, 3, 180, 20)  — identical to HPELiNet / MetaFiNet
* Output : (B, 1, N, 3)     — N=14 for PWIF3D, N=17 for MMFI
* Graph  : Dynamic skeleton loading based on dataset
* src_mask: dynamic n_pts (was hardcoded to 17 in upstream GraFormer)
* `pretrained` kwarg matches models/factory.py API
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize

from models.graphposefi_arch.GraFormer import GraFormer, adj_mx_from_edges

# ---- Person-in-WiFi-3D (14 joints) ---------------------------------------
_EDGES_PWIF3D = torch.tensor([
    [0, 1], [1, 2],  [2, 3],
    [4, 5], [5, 6],  [6, 3],
    [7, 8], [8, 9],  [9, 3],
    [10, 11], [11, 12], [12, 3],
    [3, 13],
], dtype=torch.long)

# ---- MMFI (17 joints) ----------------------------------------------------
_EDGES_MMFI = torch.tensor([
    [0, 1], [1, 3], [0, 2], [2, 4],           # legs
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  # arms
    [5, 11], [6, 12], [11, 12],               # shoulders-torso
    [11, 13], [13, 15], [12, 14], [14, 16],   # spine-head
], dtype=torch.long)


def _get_skeleton_edges(dataset='person-in-wifi-3d', num_keypoints=14):
    """Get skeleton edges based on dataset or num_keypoints."""
    if dataset == 'mmfi' or num_keypoints == 17:
        return _EDGES_MMFI, 17
    else:  # default to person-in-wifi-3d
        return _EDGES_PWIF3D, 14


def _compute_resize_dims(num_keypoints, num_person=1):
    """
    Compute resize dimensions matching MetaFiNet's formula:
    H = 136 + 8*(num_kp*num_person - 17)
    
    For PWIF3D (14 kp, 1 person): 136 + 8*(14-17) = 112
    For MMFI    (17 kp, 1 person): 136 + 8*(17-17) = 136
    """
    h_in = 136 + 8 * (num_keypoints * num_person - 17)
    w_in = 32
    return h_in, w_in


def _compute_resnet_output_dims(h_in, pretrained=False):
    """
    Compute actual ResNet-34 output spatial height by running a dummy forward.
    This avoids manual stride accounting which is error-prone.
    """
    import torchvision
    rn = torchvision.models.resnet34(weights=None)
    conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    with torch.no_grad():
        dummy = torch.zeros(1, 1, h_in, 32)
        x = nn.functional.relu(rn.bn1(conv1(dummy)))
        x = rn.maxpool(x)
        x = rn.layer1(x); x = rn.layer2(x)
        x = rn.layer3(x); x = rn.layer4(x)
    h_out = x.shape[2]
    w_out = x.shape[3]   # should be 2
    return h_out, w_out


class _ResNet34Encoder(nn.Module):
    """ResNet-34 body with a single-channel first conv (one antenna slice)."""

    def __init__(self, pretrained: bool = False):
        super().__init__()
        import torchvision
        rn = torchvision.models.resnet34(
            weights='IMAGENET1K_V1' if pretrained else None)
        self.conv1  = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1    = rn.bn1
        self.relu   = rn.relu
        self.maxpool = rn.maxpool   # stride=2, kernel=3, padding=1
        self.layer1 = rn.layer1
        self.layer2 = rn.layer2
        self.layer3 = rn.layer3
        self.layer4 = rn.layer4   # → (B, 512, H', W')

    def forward(self, x):         # x: (B, 1, H, W)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x                  # (B, 512, H', W')


class _AntAggregator(nn.Module):
    """
    Aggregate 3 antenna feature maps into N_JOINTS joint embeddings.

    Input : list of 3 tensors (B, 512, H_out, W_out)
    Output: (B, N_JOINTS, out_dim)

    Strategy
    --------
    1. Concatenate along width dim → (B, 512, H_out, N_ANT*W_out)
    2. Average-pool over width → (B, 512, H_out)
    3. Linear project: 512 → out_dim  (per token)
    4. Linear project: H_out → N_JOINTS
    5. LayerNorm.
    """

    def __init__(self, in_c: int = 512, out_dim: int = 256,
                 n_spatial: int = 7, n_joints: int = 14):
        super().__init__()
        self.chan_proj  = nn.Linear(in_c, out_dim, bias=False)
        self.joint_proj = nn.Linear(n_spatial, n_joints, bias=False)
        self.ln         = nn.LayerNorm(out_dim)

    def forward(self, enc_list):
        # enc_list: 3 × (B, 512, H', W')
        x = torch.cat(enc_list, dim=3)          # (B, 512, H', N_ant*W')
        x = x.mean(dim=3)                       # (B, 512, H')
        x = x.permute(0, 2, 1)                  # (B, H', 512)
        x = self.chan_proj(x)                   # (B, H', out_dim)
        x = x.permute(0, 2, 1)                  # (B, out_dim, H')
        x = self.joint_proj(x)                  # (B, out_dim, N_joints)
        x = x.permute(0, 2, 1)                  # (B, N_joints, out_dim)
        return self.ln(x)


class GraphPoseFiNet(nn.Module):
    """
    GraphPose-Fi with dynamic skeleton support for PWIF3D (14) and MMFI (17).

    Parameters
    ----------
    num_keypoints  : int  — 14 (PWIF3D) or 17 (MMFI)
    num_coor       : int  — 3 (xyz coordinates)
    num_person     : int  — 1 (single person)
    subcarrier_num : int  — 180 (ignored, for factory API parity)
    dataset        : str  — 'person-in-wifi-3d' or 'mmfi'
    pretrained     : bool — ImageNet init for ResNet-34 backbone
    num_layers     : int  — GraFormer depth
    """

    def __init__(self,
                 num_keypoints: int = 14,
                 num_coor: int = 3,
                 num_person: int = 1,
                 subcarrier_num: int = 180,
                 dataset: str = 'person-in-wifi-3d',
                 pretrained: bool = False,
                 num_layers: int = 4,
                 agg_mode: str = 'attn2'):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_coor      = num_coor
        self.num_person    = num_person
        self.dataset       = dataset

        # Get skeleton configuration
        edges, n_joints = _get_skeleton_edges(dataset, num_keypoints)
        if n_joints != num_keypoints:
            print(f"[GraphPoseFi] Warning: num_keypoints={num_keypoints} "
                  f"but dataset {dataset} expects {n_joints}. Using {n_joints}.")
            self.num_keypoints = n_joints

        # Compute dynamic dimensions
        h_in, w_in = _compute_resize_dims(self.num_keypoints, num_person)
        h_out, w_out = _compute_resnet_output_dims(h_in)
        print(f"[GraphPoseFi] Dataset: {dataset}, Keypoints: {self.num_keypoints}, "
              f"Input: ({h_in}, {w_in}), Output spatial: ({h_out}, {w_out})")

        self.encoder  = _ResNet34Encoder(pretrained=pretrained)
        self.agg      = _AntAggregator(in_c=512, out_dim=256,
                                        n_spatial=h_out,
                                        n_joints=self.num_keypoints)
        self._resize  = Resize([h_in, w_in])
        self._w_out   = w_out
        # BN over concatenated 3-antenna features: width = 3 * W_OUT
        self.bn2      = nn.BatchNorm2d(512)

        # Build graph adjacency
        adj_cpu = adj_mx_from_edges(
            num_pts=self.num_keypoints,
            edges=edges,
            sparse=False,
        )

        self.graformer = GraFormer(
            adj=adj_cpu,
            in_feat=256,
            hid_dim=128,
            out_dim=num_coor,
            num_layers=num_layers,
            n_head=4,
            dropout=0.1,
            n_pts=self.num_keypoints,
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (B, 3, 180, 20)  normalised CSI

        Returns
        -------
        pose : (B, 1, N, 3)  where N=14 or 17
        fea  : (B, N, 256)
        """
        n_ant = 3
        enc_list = []
        for a in range(n_ant):
            xa = x[:, a:a+1, :, :]          # (B, 1, 180, 20)
            xa = self._resize(xa)            # (B, 1, H_in, W_in)
            ea = self.encoder(xa)            # (B, 512, H_out, W_out)
            enc_list.append(ea)

        # BN over concatenated feature
        cat = torch.cat(enc_list, dim=3)    # (B, 512, H_out, 3*W_out)
        cat = self.bn2(cat)

        # Re-split for aggregator
        w = self._w_out
        enc_list = [cat[:, :, :, i*w:(i+1)*w] for i in range(n_ant)]

        fea  = self.agg(enc_list)           # (B, N, 256)
        pose = self.graformer(fea)          # (B, N, 3)
        pose = pose.unsqueeze(1)            # (B, 1, N, 3)

        return pose, fea


def graphposefi_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
