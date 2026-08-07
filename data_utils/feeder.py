import os
import numpy as np
from torch.utils.data import Dataset


def _read_list(path):
    with open(path) as f:
        return [ln.strip().split()[0] for ln in f if ln.strip()]


class PersonInWiFi3D(Dataset):
    def __init__(self, split, data_root, experiment_name='one-person', num_person=1):
        self.split = split
        self.num_person = num_person
        sub = 'train_data' if split == 'training' else 'test_data'
        self.root = os.path.normpath(os.path.join(data_root, sub))
        lst = os.path.join(self.root, f'{sub}_list.txt')
        names = _read_list(lst)
        self.items = []
        for nm in names:
            try:
                pc = int(nm.split('_')[0][2])
            except (IndexError, ValueError):
                pc = 1
            keep = ({'one-person': 1, 'two-person': 2, 'three-person': 3}
                    .get(experiment_name, None))
            if keep is not None and pc != keep:
                continue
            self.items.append({
                'csi': os.path.normpath(os.path.join(self.root, 'csi_ap', nm + '.npy')),
                'kpt': os.path.normpath(os.path.join(self.root, 'keypoint', nm + '.npy')),
                'name': nm,
            })

    # ----- split the original read_frame into load + normalize -----------------------
    @staticmethod
    def load_raw(csi_path):
        return np.load(csi_path).astype(np.float32)

    @staticmethod
    def normalize(raw):
        amp = raw[:, :90, :]; ph = raw[:, 90:, :]
        amp = (amp - amp.min()) / (amp.max() - amp.min() + 1e-12)
        ph = (ph - ph.min()) / (ph.max() - ph.min() + 1e-12)
        return np.concatenate([amp, ph], axis=1).astype(np.float32)

    def load_pose(self, kpt_path):
        p = np.load(kpt_path).astype(np.float32)
        if p.ndim == 2:                                  # (14,3) -> (1,14,3)
            p = p[None]
        if p.shape[0] < self.num_person:                
            pad = np.zeros((self.num_person - p.shape[0],) + p.shape[1:], np.float32)
            p = np.concatenate([p, pad], 0)
        return p[:self.num_person]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        raw = self.load_raw(it['csi'])
        csi = self.normalize(raw)
        pose = self.load_pose(it['kpt'])
        return {'csi': csi, 'pose': pose, 'name': it['name']}


def _build_mmfi_items(data_root, split,
                      protocol='random_split',
                      random_ratio=0.8,
                      random_seed=0):
    """
    Walk the MMFI Compress/ hierarchy and return a flat list of frame-level items.

    Directory structure (verified on local data):
        <data_root>/E##/S##/A##/
            ground_truth.npy        shape (T, 17, 3) — T frames for the sequence
            wifi-csi/
                frame001_processed.npy   shape (3, 114, 10) — already in [0,1]
                frame002_processed.npy
                ...

    Each item maps one CSI frame to its corresponding ground_truth row.

    Protocols
    ---------
    random_split  (default) — matches pose_config.yaml:
        All (E,S,A) sequences are gathered, sorted, then shuffled with
        random_seed.  The first `random_ratio` fraction is training;
        the rest is validation/test.  Split is done at the sequence
        level (not frame level) to avoid data leakage.

    Args:
        data_root : path to MMFI/Compress/
        split     : 'training' | 'validation' | 'test'
                    'validation' and 'test' are treated identically.
        protocol  : currently only 'random_split' is implemented.
        random_ratio : train fraction (default 0.8).
        random_seed  : RNG seed for reproducibility (default 0).

    Returns:
        list of dicts with keys:
            'csi'  : absolute path to frame###_processed.npy
            'kpt'  : absolute path to ground_truth.npy  (shared by all frames in seq)
            'frame_idx' : int — row index into ground_truth.npy for this frame
            'name' : human-readable identifier  "E##_S##_A##_f####"
    """
    root = os.path.normpath(data_root)

    # ── Enumerate all valid (env, subject, action) sequences ──────────────
    sequences = []   # list of (env, subj, action, seq_dir)
    for env in sorted(os.listdir(root)):
        env_dir = os.path.join(root, env)
        if not os.path.isdir(env_dir) or not env.startswith('E'):
            continue
        for subj in sorted(os.listdir(env_dir)):
            subj_dir = os.path.join(env_dir, subj)
            if not os.path.isdir(subj_dir) or not subj.startswith('S'):
                continue
            for action in sorted(os.listdir(subj_dir)):
                seq_dir = os.path.join(subj_dir, action)
                if not os.path.isdir(seq_dir) or not action.startswith('A'):
                    continue
                gt_path = os.path.join(seq_dir, 'ground_truth.npy')
                wifi_dir = os.path.join(seq_dir, 'wifi-csi')
                if os.path.exists(gt_path) and os.path.exists(wifi_dir):
                    sequences.append((env, subj, action, seq_dir))

    if not sequences:
        raise RuntimeError(
            f'No valid MMFI sequences found under {data_root}. '
            'Expected structure: <root>/E##/S##/A##/ground_truth.npy + wifi-csi/')

    # ── Train / val split at sequence level (no frame-level leakage) ──────
    rng = np.random.default_rng(random_seed)
    idx = np.arange(len(sequences))
    rng.shuffle(idx)

    n_train = int(len(idx) * random_ratio)
    if split == 'training':
        chosen_idx = set(idx[:n_train].tolist())
    else:                   # 'validation' or 'test'
        chosen_idx = set(idx[n_train:].tolist())

    # ── Build frame-level item list ────────────────────────────────────────
    items = []
    for seq_i, (env, subj, action, seq_dir) in enumerate(sequences):
        if seq_i not in chosen_idx:
            continue

        gt_path  = os.path.join(seq_dir, 'ground_truth.npy')
        wifi_dir = os.path.join(seq_dir, 'wifi-csi')

        # Load GT to know the total number of frames then release mmap handle
        gt_tmp = np.load(gt_path, mmap_mode='r')   # (T, 17, 3)
        n_frames = int(gt_tmp.shape[0])
        del gt_tmp   # release mmap file handle immediately — avoid fd exhaustion

        frame_files = sorted(
            f for f in os.listdir(wifi_dir) if f.endswith('_processed.npy')
        )

        # Pair each CSI frame with its ground_truth row by position
        n_pairs = min(n_frames, len(frame_files))
        for fi in range(n_pairs):
            items.append({
                'csi':       os.path.normpath(os.path.join(wifi_dir, frame_files[fi])),
                'kpt':       os.path.normpath(gt_path),
                'frame_idx': fi,
                'name':      f'{env}_{subj}_{action}_f{fi+1:04d}',
            })

    return items


class MMFI(Dataset):
    """
    MMFI WiFi-CSI → 3-D pose dataset (17 COCO keypoints).

    Data layout (local copy at data_root = MMFI/Compress/):
        E##/S##/A##/
            ground_truth.npy     (T, 17, 3)  float32  — 3-D keypoints (meters)
            wifi-csi/
                frame###_processed.npy   (3, 114, 10)  float64 in [0, 1]

    The dataset is split at the **sequence** level using a reproducible
    random 80/20 shuffle (seed=0), matching the protocol in
    configmmfi/pose_config.yaml (split_to_use: random_split, ratio: 0.8).

    CSI normalization:
        The raw data is already in [0, 1] (pre-processed by the MMFI authors).
        normalize() performs a light per-sample min-max rescale to [0, 1]
        in case of numerical drift, then casts to float32.

    Args:
        split     : 'training' | 'validation' | 'test'
        data_root : path to MMFI/Compress/ directory
        num_person: always 1 for MMFI (single-person dataset)
        protocol  : split protocol — only 'random_split' currently supported
        random_ratio : train fraction for random_split (default 0.8)
        random_seed  : RNG seed (default 0 — matches pose_config.yaml)
    """

    def __init__(self, split, data_root, num_person=1,
                 protocol='random_split',
                 random_ratio=0.8,
                 random_seed=0):
        self.split = split
        self.num_person = num_person
        self.data_root = os.path.normpath(data_root)

        self.items = _build_mmfi_items(
            data_root=self.data_root,
            split=split,
            protocol=protocol,
            random_ratio=random_ratio,
            random_seed=random_seed,
        )

        if not self.items:
            raise RuntimeError(
                f'MMFI: no items found for split="{split}" under {self.data_root}. '
                'Check data_root and split name.')

    # ------------------------------------------------------------------
    @staticmethod
    def load_raw(csi_path):
        """Load raw CSI frame.  Shape: (3, 114, 10), values in [0, 1]."""
        return np.load(csi_path).astype(np.float32)

    @staticmethod
    def normalize(raw):
        """
        Light per-sample min-max rescale to ensure [0, 1] float32.

        The MMFI pre-processed files are already in [0, 1], so this is
        essentially a no-op for valid frames.  It guards against any
        numerical drift without splitting amp/phase (MMFI subcarriers are
        not interleaved like Person-in-WiFi-3D's 90+90 layout).
        """
        mn, mx = raw.min(), raw.max()
        return ((raw - mn) / (mx - mn + 1e-12)).astype(np.float32)

    def load_pose(self, kpt_path, frame_idx):
        """
        Load the 3-D pose for a single frame.

        ground_truth.npy has shape (T, 17, 3).  We index row `frame_idx`
        to get the (17, 3) keypoints for this specific frame, then wrap
        it as (num_person, 17, 3).
        """
        gt = np.load(kpt_path, mmap_mode='r')          # (T, 17, 3) memory-mapped
        p  = gt[frame_idx].astype(np.float32)          # (17, 3)
        p  = p[None]                                   # (1, 17, 3)
        # Pad if num_person > 1 (not expected for MMFI, kept for API parity)
        if p.shape[0] < self.num_person:
            pad = np.zeros((self.num_person - p.shape[0],) + p.shape[1:], np.float32)
            p = np.concatenate([p, pad], 0)
        return p[:self.num_person]                     # (num_person, 17, 3)

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it  = self.items[i]
        raw = self.load_raw(it['csi'])
        csi = self.normalize(raw)                      # (3, 114, 10) float32
        pose = self.load_pose(it['kpt'], it['frame_idx'])  # (1, 17, 3) float32
        return {'csi': csi, 'pose': pose, 'name': it['name']}

