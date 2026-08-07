import os, argparse, yaml, time
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from data_utils.feeder import PersonInWiFi3D, MMFI
from attack.trigger import MicroDopplerTrigger, velocity_profiles_from_skeleton
from attack.poison import PoisonedDataset, collate
from attack.payload import set_skeleton_config
from models.factory import build_model
from eval import metrics as M


def _load_dataset(cfg, split):
    dataset_name = cfg.get('experiment_name', 'one-person')
    if dataset_name == 'mmfi':
        return MMFI(split=split, data_root=cfg['dataset_root'],
                    num_person=cfg.get('num_person', 1))
    else:
        return PersonInWiFi3D(split=split, data_root=cfg['dataset_root'],
                               experiment_name=dataset_name,
                               num_person=cfg.get('num_person', 1))


def _get_dataset_name(cfg):
    exp = cfg.get('experiment_name', 'one-person')
    return 'mmfi' if exp == 'mmfi' else 'person-in-wifi-3d'


def _get_num_keypoints(cfg):
    return 17 if _get_dataset_name(cfg) == 'mmfi' else 14


def build_trigger(cfg):
    t = MicroDopplerTrigger(aoa_spread=cfg.get('aoa_spread', 0.6), seed=cfg.get('seed', 0))
    vel, pos, movers = velocity_profiles_from_skeleton(cfg['action_npy'],
                                                       top_k=cfg.get('top_k', 6))
    t.build(vel, pos)
    return t


# ── Checkpoint helpers ───────────────────────────────────────────────────────

def _ckpt_path(cfg, outdir=None):
    """Return checkpoint path for this run."""
    tag = f"{cfg['model']}_{cfg.get('dose_mode','ln')}_{cfg.get('experiment_name','mmfi')}"
    base = Path(outdir) if outdir else Path('experiments_out') / tag
    base.mkdir(parents=True, exist_ok=True)
    return base / 'checkpoint.pt'


def _save_checkpoint(path, model, optimizer, epoch, best_loss, cfg):
    torch.save({
        'epoch':      epoch,
        'model':      model.state_dict(),
        'optimizer':  optimizer.state_dict(),
        'best_loss':  best_loss,
        'cfg':        cfg,
    }, path)
    print(f'[ckpt] saved → {path}  (epoch {epoch})', flush=True)


def _load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    print(f'[ckpt] resumed from epoch {ckpt["epoch"]}  (loss={ckpt["best_loss"]:.4f})', flush=True)
    return ckpt['epoch'], ckpt['best_loss']


# ── Predict helper ───────────────────────────────────────────────────────────

@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    preds, trues, targets, doses = [], [], [], []
    for b in loader:
        out, _ = model(b['csi'].to(device))
        preds.append(out.cpu().numpy())
        trues.append(b['pose'].numpy())
        if 'target' in b: targets.append(b['target'].numpy())
        if 'dose'   in b: doses.append(b['dose'].numpy())
    P  = np.concatenate(preds)[:, 0]
    Tr = np.concatenate(trues)[:, 0]
    Tg = np.concatenate(targets)[:, 0] if targets else None
    Do = np.concatenate(doses)         if doses    else None
    return P, Tr, Tg, Do


# ── Evaluate ─────────────────────────────────────────────────────────────────

def evaluate(model, base_test, trig, cfg, device):
    """
    Evaluate backdoored model.

    Key fix: build PoisonedDataset ONCE per mode (not per dose step) and
    cache base_test items — avoids re-scanning MMFI directory 7 times.
    """
    pivot        = cfg['pivot']
    dataset_name = _get_dataset_name(cfg)
    # Always num_workers=0 inside evaluate to avoid multiprocessing deadlock
    # when called from within a spawned subprocess.
    batch_size   = cfg['batch_size']

    def _make_ds(mode, **kw):
        return PoisonedDataset(base_test, trig, mode=mode, pivot=pivot,
                               dataset=dataset_name, **kw)

    def _dl(ds):
        return DataLoader(ds, batch_size=batch_size,
                          collate_fn=collate, num_workers=0, pin_memory=False)

    # ── Clean accuracy ──────────────────────────────────────────────────
    print('[eval] clean pass...', flush=True)
    Pc, Tc, _, _ = _predict(model, _dl(_make_ds('clean')), device)
    res = {
        'clean_mpjpe':    float(M.mpjpe(Pc, Tc).mean()),
        'clean_pampjpe':  float(M.pa_mpjpe(Pc, Tc).mean()),
        'clean_pck@0.5':  M.pck(Pc, Tc, 0.5),
    }

    # ── Dose-response (single pass per dose, no repeated dataset init) ──
    grid = cfg.get('dose_grid', [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    disp, tmp, loc, plaus = [], [], [], []
    n_joints = _get_num_keypoints(cfg)

    kw_common = dict(eps=cfg['eps'],
                     theta_max_deg=cfg['theta_max_deg'],
                     dose_mode=cfg['dose_mode'])

    print(f'[eval] dose-response grid ({len(grid)} points)...', flush=True)
    for d in grid:
        tds = _make_ds('trigger@dose', fixed_dose=d, **kw_common)
        Pd, Td, Tg, _ = _predict(model, _dl(tds), device)
        disp.append(M.subchain_displacement(Pd, Pc, pivot))
        tmp.append(float(M.target_mpjpe(Pd, Tg, pivot).mean()))
        loc.append(M.nontarget_preservation(Pd, Pc, pivot, n_joints=n_joints))
        plaus.append(M.plausibility_error(Pd, Td, pivot))

    tfloor, nfloor = M.clean_floor(Pc, Tc, pivot)
    res['dose_grid']        = list(map(float, grid))
    res['displacement']     = list(map(float, disp))
    res['tmpjpe']           = list(map(float, tmp))
    res['nontarget_mpjpe']  = list(map(float, loc))
    res['plausibility']     = list(map(float, plaus))
    res['clean_target_floor'] = tfloor
    res['dose_response']    = M.dose_response_analysis(grid, disp)

    # ── Conjunctive ASR at max dose ──────────────────────────────────────
    print('[eval] ASR pass...', flush=True)
    dref = grid[-1]
    tds  = _make_ds('trigger@dose', fixed_dose=dref, **kw_common)
    Pd, Td, Tg, _ = _predict(model, _dl(tds), device)
    res['asr@ref'] = M.attack_metrics(Pd, Tg, Pc, Tc, pivot,
                                      k_attack=cfg.get('k_attack', 1.5),
                                      k_clean=cfg.get('k_clean', 1.5),
                                      tau_plaus=cfg.get('tau_plaus', 0.20))
    return res


# ── Train ─────────────────────────────────────────────────────────────────────

def train(cfg, ckpt_dir=None):
    device = cfg.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg.get('seed', 0))
    np.random.seed(cfg.get('seed', 0))

    dataset_name  = _get_dataset_name(cfg)
    num_keypoints = _get_num_keypoints(cfg)
    set_skeleton_config(dataset_name)

    print(f'[train] loading dataset ({dataset_name})...', flush=True)
    base_train = _load_dataset(cfg, 'training')
    base_test  = _load_dataset(cfg, 'validation' if dataset_name != 'mmfi' else 'test')
    print(f'[train] train={len(base_train)} val={len(base_test)} samples', flush=True)

    trig = build_trigger(cfg)

    pois = PoisonedDataset(
        base_train, trig, mode='train',
        rho=cfg['rho'], dose_min=cfg['dose_min'], dose_max=cfg['dose_max'],
        eps=cfg['eps'], pivot=cfg['pivot'],
        theta_max_deg=cfg['theta_max_deg'], dose_mode=cfg['dose_mode'],
        seed=cfg.get('seed', 0), select=cfg.get('poison_select', 'uniform'),
        dataset=dataset_name,
    )
    # num_workers: dùng multiprocessing để load data song song với GPU compute.
    # Dùng 'fork' start method trên Linux để tránh overhead của 'spawn'.
    # Chỉ áp dụng cho training loader — evaluate() vẫn dùng 0 để tránh deadlock.
    n_workers = cfg.get('num_workers', 4)
    loader = DataLoader(pois, batch_size=cfg['batch_size'], shuffle=True,
                        collate_fn=collate, drop_last=False,
                        num_workers=n_workers,
                        prefetch_factor=2 if n_workers > 0 else None,
                        persistent_workers=True if n_workers > 0 else False,
                        pin_memory=True)

    model = build_model(
        cfg['model'], num_keypoints=num_keypoints,
        subcarrier_num=114 if dataset_name == 'mmfi' else 180,
        dataset=dataset_name,
        pretrained=cfg.get('pretrained', False),
    ).to(device)

    if (cfg.get('data_parallel') and str(device).startswith('cuda')
            and torch.cuda.device_count() > 1):
        model = torch.nn.DataParallel(model)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'],
                             weight_decay=cfg.get('weight_decay', 1e-4))

    # ── Checkpoint setup ────────────────────────────────────────────────
    ckpt_file  = _ckpt_path(cfg, ckpt_dir)
    start_epoch = 0
    best_loss   = float('inf')

    if ckpt_file.exists():
        try:
            start_epoch, best_loss = _load_checkpoint(ckpt_file, model, opt, device)
            start_epoch += 1   # resume from next epoch
        except Exception as e:
            print(f'[ckpt] WARNING: could not load checkpoint ({e}), starting fresh', flush=True)
            start_epoch = 0

    # ── LR scheduler ────────────────────────────────────────────────────
    use_scheduler = cfg.get('lr_scheduler', False)
    scheduler = None
    if use_scheduler:
        warmup = cfg.get('warmup_epochs', 10)
        total  = cfg['epochs']
        def _lr_lambda(ep):
            if ep < warmup:
                return (ep + 1) / max(warmup, 1)
            progress = (ep - warmup) / max(total - warmup, 1)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)
        # Fast-forward scheduler to match resumed epoch
        for _ in range(start_epoch):
            scheduler.step()

    # ── Training loop ────────────────────────────────────────────────────
    ckpt_every  = cfg.get('ckpt_every', 10)   # save every N epochs
    total_epochs = cfg['epochs']

    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()
        model.train()
        losses = []

        for batch_i, b in enumerate(loader):
            csi  = b['csi'].to(device)
            pose = b['pose'].to(device)
            pred, _ = model(csi)
            loss = torch.mean(torch.norm(pred - pose, dim=-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        if scheduler is not None:
            scheduler.step()

        epoch_loss = float(np.mean(losses))
        lr_now     = opt.param_groups[0]['lr']
        elapsed    = time.time() - t0
        print(f'epoch {epoch}/{total_epochs-1}: loss={epoch_loss:.4f}  '
              f'lr={lr_now:.2e}  t={elapsed:.1f}s', flush=True)

        # Save checkpoint
        if (epoch + 1) % ckpt_every == 0 or epoch == total_epochs - 1:
            _save_checkpoint(ckpt_file, model, opt, epoch, epoch_loss, cfg)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = ckpt_file.parent / 'best.pt'
            torch.save(model.state_dict(), best_path)

    # ── Evaluation ───────────────────────────────────────────────────────
    print('[train] training done, starting evaluation...', flush=True)
    res = evaluate(model, base_test, trig, cfg, device)
    res['n_poison']     = int(pois.n_poison)
    res['n_total']      = int(pois.n_total)
    res['poison_select'] = cfg.get('poison_select', 'uniform')

    # ── Print summary ────────────────────────────────────────────────────
    MM     = 1000.0
    asr_d  = res['asr@ref']
    dr     = res['dose_response']
    print(f'\n{"─"*56}', flush=True)
    print(f'  EVAL  model={cfg["model"]}  pivot={cfg["pivot"]}  dose_mode={cfg["dose_mode"]}')
    print(f'  dataset={dataset_name}  num_keypoints={num_keypoints}')
    print(f'{"─"*56}')
    print(f'  Clean accuracy')
    print(f'    MPJPE       : {res["clean_mpjpe"]*MM:7.2f} mm')
    print(f'    PA-MPJPE    : {res["clean_pampjpe"]*MM:7.2f} mm')
    print(f'    PCK@0.5     : {res["clean_pck@0.5"]*100:7.2f} %')
    print(f'  Attack (dose=1.0)')
    print(f'    Displacement: {res["displacement"][-1]*MM:7.2f} mm')
    print(f'    t-MPJPE     : {res["tmpjpe"][-1]*MM:7.2f} mm')
    print(f'    Nontarget   : {res["nontarget_mpjpe"][-1]*MM:7.2f} mm')
    print(f'    Plausibility: {res["plausibility"][-1]:7.4f}')
    print(f'  Dose-response')
    print(f'    Spearman ρ  : {dr["spearman"]:7.4f}')
    print(f'  Conjunctive ASR')
    print(f'    ASR         : {asr_d["asr"]:7.4f}')
    print(f'    Landed      : {asr_d["frac_landed"]:7.4f}')
    print(f'    Preserved   : {asr_d["frac_preserved"]:7.4f}')
    print(f'    Plausible   : {asr_d.get("frac_plausible", float(asr_d["plausible"])):7.4f}')
    print(f'  Poison  : {res["n_poison"]}/{res["n_total"]} ({res["poison_select"]})')
    print(f'{"─"*56}\n', flush=True)

    return model, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/attack.yaml')
    ap.add_argument('--ckpt-dir', default=None,
                    help='Directory to save/load checkpoints')
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    _, res = train(cfg, ckpt_dir=a.ckpt_dir)
    print('\n==== RESULTS ====')
    import json; print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
