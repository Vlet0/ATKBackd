"""
eval/vis_skeleton.py  —  3-D skeleton visualisation for BackWiFi paper.

Colour scheme:
  Blue  (#1565C0) — all normal joints and bones
  Red   (#D32F2F) — attacked sub-chain (pivot + descendants)

Layout 2×2:
  [Clean GT]       [Clean Pred]
  [Attacked Pred]  [Attacker Target]
"""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np

# ── skeleton connectivity (PWIF3D, 14 joints, root=3) ────────────────────────
# ── Skeleton edge definitions ──────────────────────────────────────────────────
# Person-in-WiFi-3D (14 joints)
_EDGES_PWIF3D = [
    (3,  2), (2,  1), (1,  0),     # right arm
    (3,  6), (6,  5), (5,  4),     # left arm
    (3,  9), (9,  8), (8,  7),     # right leg
    (3, 12), (12, 11), (11, 10),   # left leg
    (3, 13),                        # torso → head
]

# MMFI (17 joints)
_EDGES_MMFI = [
    (0, 1), (1, 3), (0, 2), (2, 4),           # legs
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),               # shoulders-torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # spine-head
]

_BLUE = '#1565C0'   # normal joints / bones
_RED  = '#D32F2F'   # attacked sub-chain


def get_skeleton_edges(dataset='person-in-wifi-3d', num_joints=None):
    """Get skeleton edges for visualization based on dataset or num_joints."""
    if dataset == 'mmfi' or (num_joints is not None and num_joints == 17):
        return _EDGES_MMFI
    else:  # default to person-in-wifi-3d
        return _EDGES_PWIF3D

_ELEV = 20
_AZIM = -60


def _get_attacked(pivot: int) -> set:
    from attack.payload import descendants
    return {pivot} | set(descendants(pivot))


def _draw_one(ax, pose: np.ndarray, attacked: set,
              title: str, center: np.ndarray, hr: float, 
              edges=None, num_joints=14) -> None:
    """Draw one skeleton view.
    
    Args:
        edges: List of edge tuples. If None, uses _EDGES_PWIF3D
        num_joints: Number of joints (14 or 17)
    """
    if edges is None:
        edges = _EDGES_PWIF3D
        
    xs, ys, zs = pose[:, 0], pose[:, 1], pose[:, 2]

    # bones
    for u, v in edges:
        col = _RED if (u in attacked or v in attacked) else _BLUE
        lw  = 3.0 if col == _RED else 2.0
        ax.plot([xs[u], xs[v]], [ys[u], ys[v]], [zs[u], zs[v]],
                color=col, linewidth=lw, alpha=0.95,
                solid_capstyle='round', zorder=3)

    # joints
    for j in range(num_joints):
        col = _RED if j in attacked else _BLUE
        sz  = 80  if j in attacked else 45
        ax.scatter(xs[j], ys[j], zs[j],
                   c=col, s=sz, zorder=5,
                   depthshade=False,
                   edgecolors='white', linewidths=0.6)

    # equal-aspect cube
    ax.set_xlim(center[0] - hr, center[0] + hr)
    ax.set_ylim(center[1] - hr, center[1] + hr)
    ax.set_zlim(center[2] - hr, center[2] + hr)

    ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
    ax.set_xlabel('X', fontsize=6, labelpad=1)
    ax.set_ylabel('Y', fontsize=6, labelpad=1)
    ax.set_zlabel('Z', fontsize=6, labelpad=1)
    ax.tick_params(labelsize=5, pad=0)

    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('#DDDDDD')
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.view_init(elev=_ELEV, azim=_AZIM)


def save_skeleton_figure(
    Pc: np.ndarray,
    Tc: np.ndarray,
    Pd: np.ndarray,
    Tg: np.ndarray,
    pivot:      int,
    outpath:    str | Path,
    title:      str = '',
    sample_idx: int = 0,
    dataset:    str = 'person-in-wifi-3d',
) -> None:
    """Save 2x2 skeleton comparison figure.
    
    Args:
        dataset: 'person-in-wifi-3d' or 'mmfi' (determines skeleton structure)
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # Get skeleton configuration
    edges = get_skeleton_edges(dataset)
    num_joints = 17 if dataset == 'mmfi' else 14

    # Pick sample with most spread-out joints (most "visible" skeleton)
    # Use GT as reference for selection
    spreads = (Tc.max(axis=1) - Tc.min(axis=1)).sum(axis=1)  # (N,)
    idx = int(np.argmax(spreads))

    panels = [
        ('Clean GT',        Tc[idx]),
        ('Clean Pred',      Pc[idx]),
        ('Attacked Pred',   Pd[idx]),
        ('Attacker Target', Tg[idx]),
    ]

    attacked = _get_attacked(pivot)

    # unified equal-aspect limits
    all_pts = np.concatenate([Tc[idx], Pc[idx], Pd[idx], Tg[idx]], axis=0)
    center  = (all_pts.max(0) + all_pts.min(0)) / 2.0
    hr      = (all_pts.max(0) - all_pts.min(0)).max() / 2.0 * 1.15

    fig = plt.figure(figsize=(11, 9), facecolor='white')
    fig.suptitle(title or 'Backdoor skeleton comparison',
                 fontsize=11, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.05, wspace=0.0,
                           left=0.02, right=0.98,
                           top=0.93, bottom=0.10)

    for k, (subtitle, pose) in enumerate(panels):
        ax = fig.add_subplot(gs[k // 2, k % 2], projection='3d')
        _draw_one(ax, pose, attacked, subtitle, center, hr, edges, num_joints)

    legend_elems = [
        Line2D([0],[0], color=_BLUE, lw=2.5, label='Normal joints'),
        Line2D([0],[0], color=_RED,  lw=3.0, label=f'Attacked sub-chain (pivot={pivot})'),
    ]
    fig.legend(handles=legend_elems, loc='lower center', ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, 0.01),
               framealpha=0.95, edgecolor='#CCCCCC')

    fig.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'[vis] Saved skeleton figure → {outpath}', flush=True)
