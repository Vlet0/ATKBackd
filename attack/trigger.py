import numpy as np

C_LIGHT = 299_792_458.0


# --------------------------------------------------------------------------- kinematics
def velocity_profiles_from_skeleton(npy_path, sample_idx=None, top_k=6,
                                    fps=30.0, los=(1.0, 0.0, 0.0)):
    """Radial velocity time-series of the top-moving joints of one action instance.

    Accepts NTU-style (N,3,T,V,M) trigger skeletons (e.g. data_bend.npy).
    Returns vel_radial (J,Tf) in m/s, pos_radial (J,), used_joints (J,).
    """
    a = np.load(npy_path, allow_pickle=True)
    if a.ndim == 5:
        a = a[..., 0]                                  # person 1 -> (N,3,T,V)
    N = a.shape[0]
    if sample_idx is None:
        occ = (a != 0).reshape(N, -1).mean(1)
        sample_idx = int(np.argmax(occ))
    s = a[sample_idx].transpose(1, 2, 0)               # (T,V,3)
    valid = np.abs(s).sum((1, 2)) > 1e-6
    s = s[valid]
    los = np.asarray(los, float); los = los / np.linalg.norm(los)
    vel = np.diff(s, axis=0) * fps                     # (Tf,V,3) m/s
    speed = np.linalg.norm(vel, axis=2)
    movers = np.argsort(-speed.mean(0))[:top_k]
    vel_radial = (vel[:, movers, :] @ los).T           # (J,Tf)
    pos_radial = (s[:, movers, :] @ los).mean(0)       # (J,)
    return vel_radial, pos_radial, movers


class MicroDopplerTrigger:
    def __init__(self, n_ant=3, n_sub=30, n_pkt=20, fc=5.32e9, df=312.5e3,
                 packet_rate=1000.0, aoa_spread=0.6, seed=0):
        self.n_ant, self.n_sub, self.n_pkt = n_ant, n_sub, n_pkt
        self.fc, self.df, self.dt = fc, df, 1.0 / packet_rate
        self.lam = C_LIGHT / fc
        self.k = np.arange(n_sub)
        self.aoa_spread = aoa_spread
        self.rng = np.random.default_rng(seed)
        self.m = None                              

    def build(self, vel_radial, pos_radial, d0=1.5):
        J, Tf = vel_radial.shape
        xq = np.linspace(0, 1, self.n_pkt)
        xp = np.linspace(0, 1, Tf)
        vel = np.stack([np.interp(xq, xp, vel_radial[j]) for j in range(J)])   # (J,P)
        nu = 2.0 * vel / self.lam
        phase_t = np.cumsum(2 * np.pi * nu * self.dt, axis=1)                  # (J,P)
        tau = 2.0 * (d0 + pos_radial) / C_LIGHT
        phase_f = -2 * np.pi * (self.k[None, :] * self.df) * tau[:, None]      # (J,S) ~flat
        gain = np.linalg.norm(vel, axis=1); gain = gain / (gain.sum() + 1e-12)
        aoa = self.rng.uniform(-self.aoa_spread, self.aoa_spread, size=(J, self.n_ant))
        m = np.zeros((self.n_ant, self.n_sub, self.n_pkt), complex)
        for a in range(self.n_ant):
            acc = np.zeros((self.n_sub, self.n_pkt), complex)
            for j in range(J):
                acc += gain[j] * np.exp(1j * phase_f[j])[:, None] \
                       * np.exp(1j * (phase_t[j] + aoa[j, a]))[None, :]
            m[a] = acc
        m = m / (np.sqrt((np.abs(m) ** 2).mean()) + 1e-12)
        self.m = m
        return m

    def inject(self, csi, dose, eps=0.3):
        """
        Inject trigger into CSI frame.

        Supports two formats:
          Person-in-WiFi-3D : shape (3, 180, 20) — interleaved amp+phase layout
                              amp = csi[:, :90, :], phase = csi[:, 90:, :]
          MMFI               : shape (3, 114, 10) — float amplitude only in [0,1]
                              treated as pure magnitude; add perturbation directly.
        """
        C, H, W = csi.shape
        assert (C, H, W) == (self.n_ant, self.n_sub * 2, self.n_pkt) or \
               (C, H, W) == (self.n_ant, self.n_sub, self.n_pkt), \
            f"Unexpected CSI shape {csi.shape} for trigger (n_ant={self.n_ant}, " \
            f"n_sub={self.n_sub}, n_pkt={self.n_pkt})"

        if H == self.n_sub * 2:
            # Person-in-WiFi-3D: interleaved amp/phase
            amp = csi[:, :self.n_sub, :]
            ph  = csi[:, self.n_sub:, :]
            A = amp.reshape(self.n_ant, 3, self.n_sub // 3, self.n_pkt)
            P = ph.reshape(self.n_ant, 3, self.n_sub // 3, self.n_pkt)
            H_cplx = A * np.exp(1j * P)
            m = self.m[:, None, :, :]
            Ht = H_cplx * (1.0 + dose * eps * m)
            At = np.abs(Ht).reshape(self.n_ant, self.n_sub, self.n_pkt)
            Pt = np.angle(Ht).reshape(self.n_ant, self.n_sub, self.n_pkt)
            out = np.concatenate([At, Pt], axis=1).astype(np.float32)
        else:
            # MMFI: plain float amplitude, add perturbation magnitude
            perturb = np.abs(self.m).astype(np.float32)          # (n_ant, n_sub, n_pkt)
            perturb = perturb / (perturb.max() + 1e-12)          # normalise to [0,1]
            out = (csi + dose * eps * perturb).astype(np.float32)
        return out


def load_trigger(action_npy, **kw):
    t = MicroDopplerTrigger(**{k: v for k, v in kw.items()
                               if k in MicroDopplerTrigger.__init__.__code__.co_varnames})
    vel, pos, movers = velocity_profiles_from_skeleton(
        action_npy, top_k=kw.get('top_k', 6))
    t.build(vel, pos)
    t.moving_joints = movers
    return t


def build_trigger_by_name(trigger_name: str, cfg: dict):
    """
    Factory: build a trigger by name from a config dict.

    trigger_name : 'micro_dropper' | 'wanet' | 'sig' | 'blended'
    cfg          : the full experiment config dict

    Returns a trigger object with an .inject(csi, dose, eps) method.
    """
    name = trigger_name.lower().replace('-', '_')

    # Resolve CSI dimensions from config or dataset defaults
    # MMFI: (3, 114, 10)   Person-in-WiFi-3D: (3, 180, 20)
    is_mmfi = cfg.get('experiment_name', '') == 'mmfi'
    n_sub_default = 114 if is_mmfi else 180
    n_pkt_default = 10  if is_mmfi else 20
    n_sub = cfg.get('n_sub', n_sub_default)
    n_pkt = cfg.get('n_pkt', n_pkt_default)
    n_ant = cfg.get('n_ant', 3)

    if name in ('micro_dropper', 'microdropper', 'micro_doppler'):
        t = MicroDopplerTrigger(
            n_ant=n_ant,
            n_sub=n_sub,
            n_pkt=n_pkt,
            aoa_spread=cfg.get('aoa_spread', 0.6),
            seed=cfg.get('seed', 0),
        )
        vel, pos, _ = velocity_profiles_from_skeleton(
            cfg['action_npy'], top_k=cfg.get('top_k', 6))
        t.build(vel, pos)
        return t

    if name == 'wanet':
        from attack.trigger_wanet import WaNetTrigger
        t = WaNetTrigger(
            n_sub=n_sub,
            n_pkt=n_pkt,
            n_ant=n_ant,
            k=cfg.get('wanet_k', 4),
            s=cfg.get('wanet_s', 0.5),
            seed=cfg.get('seed', 0),
        )
        t.build()
        return t

    if name in ('sig', 'sig_adapter'):
        from attack.sig_adapter import SIGAdapter
        t = SIGAdapter(
            delta=cfg.get('sig_delta', 20.0),
            frequency=cfg.get('sig_frequency', 6.0),
        )
        return t

    if name in ('blended', 'blend'):
        from attack.Blended import BlendedTrigger
        t = BlendedTrigger(
            n_sub=n_sub,
            n_pkt=n_pkt,
            n_ant=n_ant,
            pattern=cfg.get('blended_pattern', 'stripe'),
            seed=cfg.get('seed', 0),
        )
        t.build()
        return t

    raise ValueError(
        f"Unknown trigger '{trigger_name}'. "
        "Choose 'micro_dropper', 'wanet', 'sig', or 'blended'."
    )
