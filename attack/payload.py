import numpy as np

# ========== Person-in-WiFi-3D (14 joints) ==========
PWIF3D_EDGES = [(0, 1), (1, 2), (2, 3),
                (4, 5), (5, 6), (6, 3),
                (7, 8), (8, 9), (9, 3),
                (10, 11), (11, 12), (12, 3),
                (3, 13)]

# ========== MMFI (17 joints) ==========
MMFI_EDGES = [
    (0, 1), (1, 3), (0, 2), (2, 4),           # legs
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),               # shoulders-torso connection
    (11, 13), (13, 15), (12, 14), (14, 16),   # spine-head
]

# Global configuration (will be set by set_skeleton_config())
N_JOINTS = 14
ROOT = 3
_CURRENT_EDGES = PWIF3D_EDGES


def set_skeleton_config(dataset='person-in-wifi-3d'):
    """Set global skeleton configuration based on dataset."""
    global N_JOINTS, ROOT, _CURRENT_EDGES, PARENT, CHILDREN, ADJ
    
    if dataset == 'mmfi':
        N_JOINTS = 17
        ROOT = 5  # Central hub in MMFI
        _CURRENT_EDGES = MMFI_EDGES
    else:  # default to person-in-wifi-3d
        N_JOINTS = 14
        ROOT = 3
        _CURRENT_EDGES = PWIF3D_EDGES
    
    # Rebuild tree with new configuration
    PARENT, CHILDREN, ADJ = _build_tree(_CURRENT_EDGES, N_JOINTS, ROOT)


def _build_tree(edges, n, root):
    adj = {i: [] for i in range(n)}
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
    parent = {root: None}; order = [root]; seen = {root}
    qi = 0
    while qi < len(order):
        u = order[qi]; qi += 1
        for v in adj[u]:
            if v not in seen:
                seen.add(v); parent[v] = u; order.append(v)
    children = {i: [] for i in range(n)}
    for v, p in parent.items():
        if p is not None:
            children[p].append(v)
    return parent, children, adj


# Initialize with default Person-in-WiFi-3D configuration
PARENT, CHILDREN, ADJ = _build_tree(_CURRENT_EDGES, N_JOINTS, ROOT)


def descendants(pivot):
    out = []
    stack = list(CHILDREN[pivot])
    while stack:
        j = stack.pop(); out.append(j); stack.extend(CHILDREN[j])
    return sorted(out)


def terminal_chains():
    leaves = [j for j in range(N_JOINTS) if not CHILDREN[j]]
    chains = []
    for leaf in leaves:
        path = [leaf]; p = PARENT[leaf]
        while p is not None:
            path.append(p); p = PARENT[p]
        path = path[::-1]                              # root ... leaf
        # report the distal 3-joint segment (pivot, mid, distal) when available
        seg = path[-3:] if len(path) >= 3 else path
        chains.append({'leaf': leaf, 'root_path': path,
                       'pivot': seg[0], 'segment': seg,
                       'rotated_joints': descendants(seg[0])})
    return chains

# --------------------------------------------------------------------------- rotation
def _rodrigues(axis, theta):
    axis = np.asarray(axis, float); axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c, s = np.cos(theta), np.sin(theta)
    C = 1 - c
    return np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def g_dose(dose, theta_max=np.deg2rad(60.0), mode='linear'):
    """Monotone dose -> rotation-angle map. dose in [0,1]."""
    dose = float(np.clip(dose, 0.0, 1.0))
    if mode == 'linear':
        return theta_max * dose
    if mode == 'sqrt':
        return theta_max * np.sqrt(dose)
    if mode == 'quad':
        return theta_max * dose ** 2
    raise ValueError(mode)


def rotate_subchain(pose, pivot, theta, axis=(0.0, 0.0, 1.0)):
    pose = np.array(pose, float, copy=True)
    rot_js = descendants(pivot)
    if not rot_js:
        return pose
    R = _rodrigues(axis, theta)
    pj = pose[..., pivot, :]                            # (...,3) pivot position
    for j in rot_js:
        rel = pose[..., j, :] - pj
        pose[..., j, :] = (rel @ R.T) + pj
    return pose


def make_target_pose(pose, pivot, dose, theta_max=np.deg2rad(60.0),
                     mode='linear', axis=(0.0, 0.0, 1.0)):
    return rotate_subchain(pose, pivot, g_dose(dose, theta_max, mode), axis)


def subchain_bone_lengths(pose, pivot):
    js = [pivot] + descendants(pivot)
    L = []
    for j in js:
        p = PARENT[j]
        if p is not None and p in js or p == pivot:
            L.append(np.linalg.norm(pose[..., j, :] - pose[..., p, :], axis=-1))
    return np.array(L)
