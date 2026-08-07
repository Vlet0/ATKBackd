"""
MMFI 17-joint skeleton configuration.

MMFI skeleton has 17 keypoints with the following structure:
- Keypoints 0-4: Lower body (feet, knees, hip)
- Keypoints 5-10: Upper body (shoulders, elbows, hands)
- Keypoints 11-16: Head and torso
"""

import numpy as np

# MMFI 17-keypoint edges (16 connections)
MMFI_EDGES = [
    (0, 1), (1, 3), (0, 2), (2, 4),           # legs
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),               # shoulders-torso connection
    (11, 13), (13, 15), (12, 14), (14, 16),   # spine-head
]

N_JOINTS = 17
ROOT = 5  # Joint 5 seems to be a central hub (connects to multiple parts)


def _build_tree(edges=MMFI_EDGES, n=N_JOINTS, root=ROOT):
    """Build tree structure from edges list."""
    adj = {i: [] for i in range(n)}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    
    parent = {root: None}
    order = [root]
    seen = {root}
    qi = 0
    
    while qi < len(order):
        u = order[qi]
        qi += 1
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                parent[v] = u
                order.append(v)
    
    children = {i: [] for i in range(n)}
    for v, p in parent.items():
        if p is not None:
            children[p].append(v)
    
    return parent, children, adj


PARENT, CHILDREN, ADJ = _build_tree()


def descendants(pivot):
    """Return all joints distal to pivot joint."""
    out = []
    stack = list(CHILDREN[pivot])
    while stack:
        j = stack.pop()
        out.append(j)
        stack.extend(CHILDREN[j])
    return sorted(out)


# Joint naming (approximate - may need verification with actual data)
JOINT_NAMES = {
    0: 'L-hip', 1: 'L-knee', 2: 'R-hip', 3: 'L-ankle', 4: 'R-knee',
    5: 'L-shoulder', 6: 'R-shoulder',
    7: 'L-elbow', 8: 'R-elbow',
    9: 'L-wrist', 10: 'R-wrist',
    11: 'neck-L', 12: 'neck-R',
    13: 'head-L', 14: 'head-R',
    15: 'head-top-L', 16: 'head-top-R',
}
