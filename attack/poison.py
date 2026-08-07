import numpy as np
from torch.utils.data import Dataset
from attack.payload import make_target_pose, g_dose, set_skeleton_config

class PoisonedDataset(Dataset):
    def __init__(self, base, trigger, mode='train', rho=0.1,
                 dose_min=0.2, dose_max=1.0, eps=0.3,
                 pivot=7, theta_max_deg=60.0, dose_mode='linear',
                 axis=(0.0, 0.0, 1.0), fixed_dose=None, seed=0, select='uniform',
                 dataset='person-in-wifi-3d'):
        """
        Args:
            dataset: 'person-in-wifi-3d' or 'mmfi' - configures skeleton structure
        """
        # Configure skeleton structure based on dataset
        set_skeleton_config(dataset)
        
        self.base = base
        self.trig = trigger
        self.mode = mode
        self.eps = eps
        self.pivot = pivot
        self.theta_max = np.deg2rad(theta_max_deg)
        self.dose_mode = dose_mode
        self.axis = axis
        self.fixed_dose = fixed_dose
        self.select = select
        self.dataset = dataset
        
        rng = np.random.default_rng(seed)
        n = len(base)
        self.n_total = n
        if mode == 'train':
            n_pois = int(round(rho * n))
            idx = self._select_poison(rng, n, n_pois, select)
            self.poison_idx = set(idx.tolist())
            self.dose_of = {i: float(rng.uniform(dose_min, dose_max))
                            for i in self.poison_idx}
        else:
            self.poison_idx = set()
            self.dose_of = {}
        self.n_poison = len(self.poison_idx)

    def _select_poison(self, rng, n, n_pois, select):
        """Which training samples to poison.
        'uniform'  : random subset (standard).
        'diverse'  : farthest-point sampling over poses -> better pose coverage, which
                     can plant the same backdoor at lower rho (a poison-efficiency study).
        """
        if n_pois <= 0:
            return np.array([], int)
        if select == 'uniform':
            return rng.choice(n, size=n_pois, replace=False)
        if select == 'diverse':
            print(f'[poison] diverse FPS sampling: loading {n} poses...', flush=True)
            poses = []
            for i in range(n):
                it = self.base.items[i]
                if 'frame_idx' in it:
                    p = self.base.load_pose(it['kpt'], it['frame_idx'])
                else:
                    p = self.base.load_pose(it['kpt'])
                poses.append(p.reshape(-1))
            poses = np.stack(poses)
            print(f'[poison] FPS selecting {n_pois} from {n}...', flush=True)
            chosen = [int(rng.integers(n))]
            d = np.linalg.norm(poses - poses[chosen[0]], axis=1)
            for _ in range(n_pois - 1):
                nxt = int(np.argmax(d)); chosen.append(nxt)
                d = np.minimum(d, np.linalg.norm(poses - poses[nxt], axis=1))
            print(f'[poison] FPS done.', flush=True)
            return np.array(chosen, int)
        raise ValueError(f'unknown select policy {select}')

    def _inject(self, name_csi_raw, dose):
        return self.trig.inject(name_csi_raw, dose, eps=self.eps)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        it = self.base.items[i]
        raw = self.base.load_raw(it['csi'])
        # MMFI items carry 'frame_idx'; PersonInWiFi3D items do not
        if 'frame_idx' in it:
            pose = self.base.load_pose(it['kpt'], it['frame_idx'])
        else:
            pose = self.base.load_pose(it['kpt'])

        if self.mode == 'train':
            if i in self.poison_idx:
                d = self.dose_of[i]
                raw = self._inject(raw, d)
                pose = make_target_pose(pose, self.pivot, d, self.theta_max,
                                        self.dose_mode, self.axis)
                poisoned = 1
            else:
                poisoned = 0
            csi = self.base.normalize(raw)
            return {'csi': csi, 'pose': pose, 'poisoned': poisoned}

        if self.mode == 'clean':
            return {'csi': self.base.normalize(raw), 'pose': pose,
                    'target': pose, 'dose': 0.0}

        if self.mode == 'trigger@dose':
            d = self.fixed_dose if self.fixed_dose is not None else 1.0
            target = make_target_pose(pose, self.pivot, d, self.theta_max,
                                      self.dose_mode, self.axis)
            raw_t = self._inject(raw, d)
            return {'csi': self.base.normalize(raw_t),
                    'pose': pose,            # true (clean) pose
                    'target': target,        # attacker-intended pose at this dose
                    'dose': float(d)}

        raise ValueError(self.mode)

def collate(batch):
    import torch
    out = {}
    out['csi'] = torch.from_numpy(np.stack([b['csi'] for b in batch])).float()
    out['pose'] = torch.from_numpy(np.stack([b['pose'] for b in batch])).float()
    if 'target' in batch[0]:
        out['target'] = torch.from_numpy(np.stack([b['target'] for b in batch])).float()
    if 'dose' in batch[0]:
        out['dose'] = torch.tensor([b['dose'] for b in batch]).float()
    if 'poisoned' in batch[0]:
        out['poisoned'] = torch.tensor([b['poisoned'] for b in batch]).long()
    return out
