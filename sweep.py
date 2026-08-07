import os, sys, json, csv, copy, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_backdoor import train

MM = 1000.0   # dataset units -> millimetres (confirmed: MPJPE*1000 = mm)

def _row(res, th, rho):
    dr = res['dose_response']; asr = res['asr@ref']
    return {
        'theta_max_deg': th, 'rho': rho,
        'clean_mpjpe_mm': res['clean_mpjpe'] * MM,
        'displacement_mm': res['displacement'][-1] * MM,
        'tmpjpe_mm': res['tmpjpe'][-1] * MM,
        'clean_target_floor_mm': res['clean_target_floor'] * MM,
        'nontarget_mpjpe_mm': res['nontarget_mpjpe'][-1] * MM,
        'plausibility': res['plausibility'][-1],
        'spearman': dr['spearman'],
        'ramp_minus_step': dr['ramp_minus_step'],
        'asr': asr['asr'],
        'frac_landed': asr['frac_landed'],
        'frac_preserved': asr['frac_preserved'],
        'n_poison': res['n_poison'],
        'n_total': res['n_total'],
        'poison_select': res['poison_select'],
    }


def run_cells(cells, base_cfg, epochs=None, device=None, tag=''):
    from train_backdoor import train
    rows = []
    for k, (th, rho) in enumerate(cells, 1):
        cfg = copy.deepcopy(base_cfg)
        cfg['theta_max_deg'] = float(th); cfg['rho'] = float(rho)
        if device is not None:
            cfg['device'] = device
        if epochs is not None:
            cfg['epochs'] = int(epochs)
        print(f"{tag}[{k}/{len(cells)}] theta_max={th} rho={rho} "
              f"epochs={cfg['epochs']} device={cfg.get('device','auto')}")
        _, res = train(cfg)
        rows.append(_row(res, th, rho))
    return rows


def run_grid(base_cfg, thetas, rhos, epochs=None, device=None):
    cells = [(th, rho) for rho in rhos for th in thetas]
    return run_cells(cells, base_cfg, epochs=epochs, device=device)


# ----- multi-GPU: distribute independent grid cells across GPUs (one cell per GPU) ---
def _worker(wid, gpu, cells, base_cfg, epochs, outdir):
    dev = gpu if isinstance(gpu, str) else f'cuda:{gpu}'
    rows = run_cells(cells, base_cfg, epochs=epochs, device=dev, tag=f'[gpu{gpu}]')
    with open(os.path.join(outdir, f'_part_{wid}.json'), 'w') as f:
        json.dump(rows, f)


def run_grid_multigpu(base_cfg, thetas, rhos, gpus, epochs, outdir):
    import torch.multiprocessing as mp
    cells = [(th, rho) for rho in rhos for th in thetas]
    shards = [cells[i::len(gpus)] for i in range(len(gpus))]   # round-robin (cells ~equal cost)
    ctx = mp.get_context('spawn')
    procs = []
    for wid, g in enumerate(gpus):
        if not shards[wid]:
            continue
        p = ctx.Process(target=_worker, args=(wid, g, shards[wid], base_cfg, epochs, outdir))
        p.start(); procs.append(p)
    for p in procs:
        p.join()
    rows = []
    for wid in range(len(gpus)):
        part = os.path.join(outdir, f'_part_{wid}.json')
        if os.path.exists(part):
            rows.extend(json.load(open(part))); os.remove(part)
    return rows


def save_tables(rows, outdir):
    with open(os.path.join(outdir, 'sweep_results.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(outdir, 'sweep_results.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def fig_tradeoff(rows, outdir):
    rhos = sorted({r['rho'] for r in rows})
    plt.figure(figsize=(6, 5))
    for rho in rhos:
        pts = sorted([r for r in rows if r['rho'] == rho], key=lambda r: r['displacement_mm'])
        x = [p['displacement_mm'] for p in pts]
        y = [p['nontarget_mpjpe_mm'] for p in pts]
        plt.plot(x, y, 'o-', label=f'rho={rho}')
        for p in pts:
            plt.annotate(f"{int(p['theta_max_deg'])}", (p['displacement_mm'], p['nontarget_mpjpe_mm']),
                         fontsize=7, xytext=(3, 3), textcoords='offset points')
    # reference: model's own clean error band
    clean = np.mean([r['clean_mpjpe_mm'] for r in rows])
    plt.axhline(clean, ls='--', c='gray', lw=1)
    plt.text(plt.xlim()[1], clean, ' clean MPJPE', va='bottom', ha='right', fontsize=8, c='gray')
    plt.xlabel('target-limb displacement (mm)   [bigger = stronger payload]')
    plt.ylabel('non-target leak MPJPE (mm)   [smaller = better localization]')
    plt.title('Localization vs payload tradeoff (label = theta_max deg)')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig_tradeoff.png'), dpi=150); plt.close()


def fig_heatmaps(rows, outdir):
    thetas = sorted({r['theta_max_deg'] for r in rows})
    rhos = sorted({r['rho'] for r in rows})
    def grid(key):
        M = np.full((len(rhos), len(thetas)), np.nan)
        for r in rows:
            M[rhos.index(r['rho']), thetas.index(r['theta_max_deg'])] = r[key]
        return M
    panels = [('nontarget_mpjpe_mm', 'leak (mm) lower=better'),
              ('tmpjpe_mm', 't-MPJPE to target (mm) lower=better'),
              ('asr', 'ASR (calibrated, higher=better)')]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (key, title) in zip(axs, panels):
        M = grid(key)
        im = ax.imshow(M, aspect='auto', origin='lower', cmap='viridis')
        ax.set_xticks(range(len(thetas))); ax.set_xticklabels([int(t) for t in thetas])
        ax.set_yticks(range(len(rhos))); ax.set_yticklabels(rhos)
        ax.set_xlabel('theta_max (deg)'); ax.set_ylabel('rho'); ax.set_title(title)
        for a in range(len(rhos)):
            for b in range(len(thetas)):
                ax.text(b, a, f'{M[a, b]:.2f}', ha='center', va='center', color='w', fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, 'fig_heatmaps.png'), dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/attack.yaml')
    ap.add_argument('--theta', type=float, nargs='+', default=[20, 30, 40, 50, 60])
    ap.add_argument('--rho', type=float, nargs='+', default=[0.025, 0.05, 0.10, 0.20, 0.30])
    ap.add_argument('--select', default=None, help='uniform | diverse (overrides config)')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--device', default=None, help="single-device run, e.g. cuda:0 | cpu (default: auto)")
    ap.add_argument('--gpus', type=int, nargs='+', default=None,
                    help="GPU ids to parallelize grid cells across, e.g. --gpus 0 1 2 3")
    ap.add_argument('--outdir', default='sweep_out')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    with open(a.config) as f:
        base = yaml.safe_load(f)
    if a.select is not None:
        base['poison_select'] = a.select

    if a.gpus and len(a.gpus) > 1:
        print(f"multi-GPU sweep across GPUs {a.gpus} "
              f"({len(a.theta)*len(a.rho)} cells, round-robin)")
        rows = run_grid_multigpu(base, a.theta, a.rho, a.gpus, a.epochs, a.outdir)
    else:
        device = a.device or (f'cuda:{a.gpus[0]}' if a.gpus else None)
        rows = run_grid(base, a.theta, a.rho, a.epochs, device=device)
    save_tables(rows, a.outdir)
    fig_tradeoff(rows, a.outdir)
    fig_heatmaps(rows, a.outdir)
    print(f"\nwrote results + figures to {a.outdir}/")


if __name__ == '__main__':
    main()