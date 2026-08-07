"""Defense runner for the CSI backdoor project.

Imports all metrics and logic directly from the canonical modules
(eval.metrics, attack.*, data_utils.feeder, models.factory)
WITHOUT inlining any metric calculations.

Usage:
  python -m defend.run_defenses \
      --config  defend/best_sweep_config.yaml \
      --checkpoint /path/to/checkpoint.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable

# Ensure backdoorviet root is on sys.path
_THIS_DIR = Path(__file__).resolve().parent    # .../defend
_ROOT_DIR = _THIS_DIR.parent                   # .../backdoorviet
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from train_backdoor import _load_dataset, _get_dataset_name, _get_num_keypoints, build_trigger as _build_trigger
from attack.payload import descendants, make_target_pose, set_skeleton_config
from attack.poison import PoisonedDataset, collate
from models.factory import build_model as _factory_build_model

# ── Import CANONICAL metrics directly from eval.metrics (NO INLINING) ──
from eval.metrics import (
    mpjpe,
    pa_mpjpe,
    pck,
    attack_metrics,
    clean_floor,
    target_mpjpe,
    subchain_displacement,
    nontarget_preservation,
    plausibility_error,
)

from defend.strip import STRIPCSI
from defend.noisec import NoiSecCSI
from defend.neural_cleanse import NeuralCleanseCSI
from defend.fine_pruning import FinePruningCSI


def collect(loader: DataLoader, limit: int) -> dict:
    values: dict = {}
    count = 0
    for batch in loader:
        keep = min(limit - count, len(batch["csi"]))
        if keep <= 0:
            break
        for name, value in batch.items():
            values.setdefault(name, []).append(value[:keep])
        count += keep
    return {name: torch.cat(chunks) for name, chunks in values.items()}


@torch.no_grad()
def predict(model: nn.Module, csi: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    output = model(csi.to(device))
    return (output[0] if isinstance(output, tuple) else output).cpu().numpy()[:, 0]


def pose_target(pivot: int, theta_max_deg: float) -> Callable:
    def build(pose: torch.Tensor) -> torch.Tensor:
        target = make_target_pose(
            pose.detach().cpu().numpy(), pivot, 1.0, np.deg2rad(theta_max_deg))
        return torch.from_numpy(target).to(pose.device).float()
    return build


def plot(summary: dict, outdir: Path) -> None:
    import matplotlib.gridspec as gridspec
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
    fig.suptitle('Backdoor Defense Evaluation Results', fontsize=16, fontweight='bold')
    
    # 1. ASR Comparison
    ax1 = fig.add_subplot(gs[0, :2])
    methods = ['Backdoored\n(baseline)', 'Fine-Pruned']
    asrs = [summary["baseline_asr"], summary["fine_pruning"]["asr"]]
    colors = ['#e74c3c', '#27ae60']
    bars1 = ax1.bar(methods, asrs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Attack Success Rate (ASR)", fontsize=11, fontweight='bold')
    ax1.set_title("Model Repair: Fine-Pruning Effect", fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Runtime Detection
    ax2 = fig.add_subplot(gs[0, 2])
    methods_det = ['STRIP', 'NoiSec']
    rates = [summary["strip"]["detection_rate"], summary["noisec"]["detection_rate"]]
    colors_det = ['#3498db', '#9b59b6']
    bars2 = ax2.bar(methods_det, rates, color=colors_det, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Detection Rate", fontsize=11, fontweight='bold')
    ax2.set_title("Runtime Detection", fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 3. Neural Cleanse Anomaly
    ax3 = fig.add_subplot(gs[1, 0])
    nc_anom = summary["neural_cleanse"]["anomaly_index"]
    ax3.bar(['Neural\nCleanse'], [nc_anom], color='#f39c12', alpha=0.8, 
            edgecolor='black', linewidth=1.5)
    ax3.axhline(2.0, color='red', linestyle='--', linewidth=2, label='MAD Threshold')
    ax3.set_ylabel("Anomaly Index", fontsize=11, fontweight='bold')
    ax3.set_title("Trigger Reverse Engineering", fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.text(0, nc_anom + 0.1, f'{nc_anom:.2f}', ha='center', va='bottom', 
            fontsize=10, fontweight='bold')
    
    # 4. Conjunctive ASR Breakdown
    ax4 = fig.add_subplot(gs[1, 1])
    components = ['Landed', 'Preserved', 'Plausible']
    values = [
        summary["baseline_detail"]["frac_landed"],
        summary["baseline_detail"]["frac_preserved"],
        summary["baseline_detail"]["frac_plausible"]
    ]
    colors_comp = ['#e67e22', '#16a085', '#8e44ad']
    bars4 = ax4.bar(components, values, color=colors_comp, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_ylim(0, 1)
    ax4.set_ylabel("Fraction", fontsize=11, fontweight='bold')
    ax4.set_title("ASR Components", fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / "defenses_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[defend] Saved visualization -> {outdir / 'defenses_summary.png'}")


def run_defenses_api(cfg: dict, checkpoint_path: str, outdir: Path | None = None) -> dict:
    device = torch.device(
        cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if outdir is None:
        outdir = Path("defend_out")
    outdir.mkdir(parents=True, exist_ok=True)

    dataset_name  = _get_dataset_name(cfg)
    num_keypoints = _get_num_keypoints(cfg)
    subcarrier   = 114 if dataset_name == 'mmfi' else 180
    set_skeleton_config(dataset_name)

    print(f"[defend] Loading dataset: {dataset_name} (keypoints={num_keypoints}, subcarriers={subcarrier})")
    test_data  = _load_dataset(cfg, 'validation' if dataset_name != 'mmfi' else 'test')
    train_data = _load_dataset(cfg, 'training')

    trig  = _build_trigger(cfg)
    pivot = cfg["pivot"]

    clean_test_ds   = PoisonedDataset(test_data,  trig, mode="clean", pivot=pivot, dataset=dataset_name)
    clean_train_ds  = PoisonedDataset(train_data, trig, mode="clean", pivot=pivot, dataset=dataset_name)
    triggered_test_ds = PoisonedDataset(
        test_data, trig, mode="trigger@dose", fixed_dose=1.0,
        eps=cfg["eps"], theta_max_deg=cfg["theta_max_deg"],
        dose_mode=cfg["dose_mode"], pivot=pivot, dataset=dataset_name
    )

    clean_test_loader = DataLoader(
        clean_test_ds, batch_size=cfg["batch_size"], collate_fn=collate, num_workers=0
    )
    clean_train_loader = DataLoader(
        clean_train_ds, batch_size=cfg["batch_size"], collate_fn=collate, shuffle=True, num_workers=0
    )
    triggered_test_loader = DataLoader(
        triggered_test_ds, batch_size=cfg["batch_size"], collate_fn=collate, num_workers=0
    )

    # ── Model build & load ──────────────────────────────────────────────────
    print(f"[defend] Building model {cfg['model']}...")
    model = _factory_build_model(
        cfg["model"],
        num_keypoints=num_keypoints,
        subcarrier_num=subcarrier,
        dataset=dataset_name,
        pretrained=cfg.get("pretrained", False),
    ).to(device)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print(f"[defend] Loaded checkpoint -> {checkpoint_path}")
    else:
        print(f"[defend] WARNING: Checkpoint {checkpoint_path} not found! Running evaluation on randomly initialized weights.")

    # ── Baseline evaluation (CANONICAL METRICS CALL) ─────────────────────────
    print("[defend] Evaluating baseline attack & clean performance...")
    pred_clean = predict(model, collect(clean_test_loader, limit=512)["csi"], device)
    true_clean = collect(clean_test_loader, limit=512)["pose"].numpy()[:, 0]

    trig_batch = collect(triggered_test_loader, limit=512)
    pred_trig  = predict(model, trig_batch["csi"], device)
    target_trig = trig_batch["target"].numpy()[:, 0]

    # Clean Pose Metrics
    clean_mpjpe_val   = float(mpjpe(pred_clean, true_clean).mean()) * 1000.0
    clean_pampjpe_val = float(pa_mpjpe(pred_clean, true_clean).mean()) * 1000.0
    clean_pck_val     = pck(pred_clean, true_clean, 0.5)

    # Attack Metrics using CANONICAL eval.metrics.attack_metrics (NO INLINING!)
    baseline = attack_metrics(
        pred_trig, target_trig, pred_clean, true_clean, pivot,
        k_attack=cfg.get("k_attack", 1.5),
        k_clean=cfg.get("k_clean", 1.5),
        tau_plaus=cfg.get("tau_plaus", 0.35),
    )
    baseline["clean_mpjpe_mm"]   = round(clean_mpjpe_val, 2)
    baseline["clean_pampjpe_mm"] = round(clean_pampjpe_val, 2)
    baseline["clean_pck_0.5"]    = round(clean_pck_val, 4)

    print(f"[defend] Baseline Clean MPJPE: {clean_mpjpe_val:.1f} mm | PA-MPJPE: {clean_pampjpe_val:.1f} mm | PCK@0.5: {clean_pck_val*100:.1f}%")
    print(f"[defend] Baseline ASR: {baseline['asr']:.1%} (landed={baseline['frac_landed']:.1%}, preserved={baseline['frac_preserved']:.1%}, plausible={baseline['frac_plausible']:.1%})")

    # ── 1. STRIP Defense ────────────────────────────────────────────────────
    print("[defend] Running STRIP Defense...")
    strip_bank = collect(clean_test_loader, limit=256)["csi"]
    split = len(strip_bank) // 2
    calib, bank = strip_bank[:split], strip_bank[split:]

    strip = STRIPCSI(model, device, n_perturbations=64, false_reject_rate=0.01, seed=cfg.get("seed", 0))
    strip.calibrate(calib, bank)

    test_trig_csi = collect(triggered_test_loader, limit=256)["csi"]
    strip_res = strip.detect(test_trig_csi, bank)
    strip_rate = float(strip_res.is_backdoor.float().mean().item())
    print(f"[defend] STRIP Detection Rate: {strip_rate:.1%}")

    # ── 2. NoiSec Defense ───────────────────────────────────────────────────
    print("[defend] Running NoiSec Defense...")
    noisec = NoiSecCSI(model, device)
    noisec.fit_autoencoder(clean_train_loader, epochs=15)
    noisec.fit_detector(clean_train_loader, false_reject_rate=0.01)

    noisec_res = noisec.detect(test_trig_csi)
    noisec_rate = float(noisec_res.is_backdoor.float().mean().item())
    print(f"[defend] NoiSec Detection Rate: {noisec_rate:.1%}")

    # ── 3. Neural Cleanse Defense ───────────────────────────────────────────
    print("[defend] Running Neural Cleanse Defense...")
    clean_sample = collect(clean_test_loader, limit=32)
    sample_pairs = [(clean_sample["csi"], clean_sample["pose"][:, 0])]

    candidates = [
        (f"pivot={p}", pose_target(p, cfg["theta_max_deg"]))
        for p in range(num_keypoints)
    ]
    nc = NeuralCleanseCSI(model, device, steps=300)
    nc_report = nc.detect(sample_pairs, candidates)
    print(f"[defend] Neural Cleanse Anomaly Index: {nc_report.anomaly_index:.2f} (Suspected: {nc_report.suspected_candidate})")

    # ── 4. Fine-Pruning Defense ─────────────────────────────────────────────
    print("[defend] Running Fine-Pruning Defense...")
    pruned_model = FinePruningCSI.clone(model)
    fp = FinePruningCSI(pruned_model, device)
    fp_report = fp.prune(clean_train_loader, layer_name="auto", fraction=0.20)

    pred_pruned_trig = predict(pruned_model, trig_batch["csi"], device)
    pred_pruned_clean = predict(pruned_model, collect(clean_test_loader, limit=512)["csi"], device)

    # CANONICAL eval.metrics.attack_metrics call for fine-pruned model
    fp_metrics = attack_metrics(
        pred_pruned_trig, target_trig, pred_pruned_clean, true_clean, pivot,
        k_attack=cfg.get("k_attack", 1.5),
        k_clean=cfg.get("k_clean", 1.5),
        tau_plaus=cfg.get("tau_plaus", 0.35),
    )
    print(f"[defend] Post-Pruning ASR: {fp_metrics['asr']:.1%} (reduced from {baseline['asr']:.1%})")

    # ── Summary JSON & Plot ─────────────────────────────────────────────────
    summary = {
        "dataset": dataset_name,
        "model": cfg["model"],
        "baseline_asr": baseline["asr"],
        "baseline_detail": baseline,
        "strip": {
            "detection_rate": strip_rate,
            "threshold": strip.threshold,
        },
        "noisec": {
            "detection_rate": noisec_rate,
            "threshold": noisec.threshold,
        },
        "neural_cleanse": {
            "anomaly_index": nc_report.anomaly_index,
            "suspected_candidate": nc_report.suspected_candidate,
        },
        "fine_pruning": {
            "pruned_layer": fp_report.layer,
            "pruned_channels_count": len(fp_report.pruned_channels),
            "asr": fp_metrics["asr"],
            "detail": fp_metrics,
        }
    }

    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot(summary, outdir)
    print(f"\n[defend] Complete! Results saved to -> {outdir}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Run 4 backdoor defense evaluation methods")
    ap.add_argument("--config", default="defend/best_sweep_config.yaml")
    ap.add_argument("--checkpoint", default="experiments_out/victim_a/hpeli_bend_micro_dropper_s0/best.pt")
    ap.add_argument("--outdir", default="defend_out")
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)

    run_defenses_api(cfg, a.checkpoint, Path(a.outdir))


if __name__ == "__main__":
    main()
