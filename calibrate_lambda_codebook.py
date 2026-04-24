"""
Usage:
    python calibrate_lambda_codebook.py --codebook_dir codebooks/codebooks_llama_1b --model_name_or_path meta-llama/Llama-3.2-1B --n_samples 512 --output_dir calibration_codebook_results
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(
        description="Fallback calibration from codebook statistics")
    p.add_argument("--codebook_dir", required=True)
    p.add_argument("--model_name_or_path", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--n_samples", type=int, default=512,
                   help="random unit vectors per (layer, head, group)")
    p.add_argument("--output_dir", default="calibration_codebook_results")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def compute_angular_errors(codebooks, num_layers, num_kv_heads, tiers,
                           n_samples=512, seed=42):
    """
    Compute epsilon_theta(b) per tier: mean angular error from VQ.

    Returns:
        eps_theta:      {tier_id: mean angular error in radians}
        tier_all_angles:{tier_id: flat list of ALL individual angle samples}
        layer_errors:   {tier_id: {layer_idx: [per-group mean errors]}}
        total_measurements: int
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    nondrop = [t for t in tiers if t.tier_id != 0]

    tier_means      = {t.tier_id: [] for t in nondrop}
    tier_all_angles = {t.tier_id: [] for t in nondrop}
    layer_errors    = {t.tier_id: defaultdict(list) for t in nondrop}
    total_measurements = 0

    for tier in nondrop:
        tid, g = tier.tier_id, tier.g

        for layer in range(num_layers):
            for head in range(num_kv_heads):
                cb = codebooks.get((layer, head, tid))
                if cb is None:
                    continue
                if cb.dim() == 2:
                    cb = cb.unsqueeze(0)

                for gi in range(cb.shape[0]):
                    cw = cb[gi]                                    # [K, g]
                    cw_n = cw / (cw.norm(dim=-1, keepdim=True) + 1e-8)

                    queries = torch.randn(n_samples, g, generator=rng)
                    queries = queries / (queries.norm(dim=-1, keepdim=True) + 1e-8)

                    sims     = queries @ cw_n.T                    # [N, K]
                    best_cw  = cw_n[sims.argmax(dim=-1)]           # [N, g]
                    cos_a    = (queries * best_cw).sum(-1).clamp(-1, 1)
                    angles   = torch.acos(cos_a)                   # [N] radians

                    mean_err = angles.mean().item()
                    tier_means[tid].append(mean_err)
                    tier_all_angles[tid].extend(angles.tolist())
                    layer_errors[tid][layer].append(mean_err)
                    total_measurements += n_samples

    eps_theta = {}
    for t in nondrop:
        errs = tier_means[t.tier_id]
        eps_theta[t.tier_id] = (sum(errs) / len(errs)) if errs else (
            math.pi / math.sqrt(t.codebook_size))

    return eps_theta, tier_all_angles, layer_errors, total_measurements


def main():
    args = parse_args()
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)

    from transformers import AutoConfig
    from codebook_loader import load_codebooks
    from tiers import build_tiers
    from config import BR

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg          = AutoConfig.from_pretrained(args.model_name_or_path)
    num_layers   = cfg.num_hidden_layers
    num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim     = getattr(cfg, "head_dim",
                           cfg.hidden_size // cfg.num_attention_heads)

    tiers     = build_tiers(head_dim)
    codebooks = load_codebooks(args.codebook_dir, num_layers,
                               num_kv_heads, tiers)

    print(f"\nComputing angular errors from codebooks ...")
    print(f"  {num_layers}L x {num_kv_heads}H x {head_dim}dh")
    print(f"  n_samples = {args.n_samples}")

    eps_theta, tier_all_angles, layer_errors, total_meas = (
        compute_angular_errors(codebooks, num_layers, num_kv_heads, tiers,
                               args.n_samples, args.seed))

    # epsilon_r: radius quantization error (conservative upper bound)
    # Symmetric int8 with br=8: max error = scale / (2*127) ~ 1/254
    # Conservative: 1 / 2^br
    eps_r = 1.0 / (2 ** BR)

    # -- lambda = epsilon (direct correspondence per C.2 fallback)
    lambda_theta = {tid: eps_theta[tid] for tid in [1, 2, 3]}
    lambda_r     = {tid: eps_r for tid in [1, 2, 3]}

    # -- Frontier validation
    nondrop = sorted([t for t in tiers if t.tier_id != 0],
                     key=lambda t: -t.token_bits())
    b1, b2, b3 = nondrop[0], nondrop[1], nondrop[2]

    lam_total = {tid: lambda_theta[tid] + lambda_r[tid] for tid in [1, 2, 3]}

    bits1, bits2, bits3 = b1.token_bits(), b2.token_bits(), b3.token_bits()
    rho_12 = (lam_total[b2.tier_id] - lam_total[b1.tier_id]) / max(bits1 - bits2, 1)
    rho_23 = (lam_total[b3.tier_id] - lam_total[b2.tier_id]) / max(bits2 - bits3, 1)
    frontier_ok = rho_12 < rho_23

    # -- Print results
    print(f"\n{'='*60}")
    print(f"CALIBRATION RESULTS (codebook statistics)")
    print(f"  Total measurements:  {total_meas:,}")
    print(f"{'='*60}")
    for tid in [1, 2, 3]:
        t = tiers[tid]
        print(f"  Tier {tid} (g={t.g}, G={t.G}, b_theta={t.b_theta}, "
              f"bits={t.token_bits()}):")
        print(f"    eps_theta = {eps_theta[tid]:.6f} rad "
              f"({math.degrees(eps_theta[tid]):.2f} deg)")
        print(f"    eps_r     = {eps_r:.6f}")
        print(f"    lambda_theta = {lambda_theta[tid]:.6f}")
        print(f"    lambda_r     = {lambda_r[tid]:.6f}")

    print(f"\n  rho(b1->b2) = {rho_12:.6f}  (bits saved: {bits1 - bits2})")
    print(f"  rho(b2->b3) = {rho_23:.6f}  (bits saved: {bits2 - bits3})")
    print(f"  b2 on frontier: {frontier_ok}")
    if not frontier_ok:
        print(f"\n  WARNING: rho(b1->b2) >= rho(b2->b3)")
        print(f"  The greedy allocator will prefer b1->b3 over b1->b2->b3.")
        print(f"  Mid-tier tokens may not appear under budget pressure.")

    # -- Save
    output = {
        "lambda_theta": {str(k): round(v, 6) for k, v in lambda_theta.items()},
        "lambda_r":     {str(k): round(v, 6) for k, v in lambda_r.items()},
        "eps_theta":    {str(k): round(v, 6) for k, v in eps_theta.items()},
        "eps_r":        round(eps_r, 6),
        "rho_b1_b2":    round(rho_12, 6),
        "rho_b2_b3":    round(rho_23, 6),
        "frontier_ok":  frontier_ok,
        "total_measurements": total_meas,
        "config": {
            "n_samples": args.n_samples,
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "br": BR,
        },
    }
    json_path = out_dir / "calibrated_lambdas_codebook.json"
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {json_path}")

    # -- Plots
    print("\nGenerating plots ...")
    _plot_histograms(tier_all_angles, eps_theta, tiers, out_dir)
    _plot_frontier(lambda_theta, lambda_r, tiers, rho_12, rho_23,
                   frontier_ok, out_dir)
    _plot_layer_heatmap(layer_errors, num_layers, tiers, out_dir)

    # -- Config instructions
    lam_th_str = ", ".join(f"{k}: {v:.4f}" for k, v in sorted(lambda_theta.items()))
    lam_r_str  = ", ".join(f"{k}: {v:.6f}" for k, v in sorted(lambda_r.items()))
    print(f"\nPaste into config.py:")
    print(f"LAMBDA_THETA = {{{lam_th_str}}}")
    print(f"LAMBDA_R     = {{{lam_r_str}}}")



def _plot_histograms(tier_all_angles, eps_theta, tiers, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    names  = {1: "b1 (High/6-bit)", 2: "b2 (Mid/4-bit)", 3: "b3 (Low/3-bit)"}
    colors = {1: "#2DC653", 2: "#4361EE", 3: "#F77F00"}

    for ax_idx, tid in enumerate([1, 2, 3]):
        ax     = axes[ax_idx]
        angles = tier_all_angles[tid]
        if not angles:
            ax.set_title(f"{names[tid]}\n(no data)")
            continue

        angles_deg = np.degrees(angles)
        mean_deg   = np.degrees(eps_theta[tid])
        tier = tiers[tid]

        ax.hist(angles_deg, bins=60, color=colors[tid], alpha=0.7,
                edgecolor='black', linewidth=0.3, density=True)
        ax.axvline(mean_deg, color='red', linestyle='--', linewidth=2,
                   label=f"mean = {mean_deg:.1f} deg")
        ax.set_title(f"{names[tid]}\n"
                     f"g={tier.g}, K={tier.codebook_size}, "
                     f"eps={eps_theta[tid]:.4f} rad")
        ax.set_xlabel("Angular error (degrees)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("(i) Angular Quantization Error Distribution per Tier "
                 "(codebook statistics, C.2 fallback)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_angular_error_histogram.png", dpi=150)
    plt.close(fig)
    print(f"  Plot (i) saved")



def _plot_frontier(lambda_theta, lambda_r, tiers, rho_12, rho_23,
                   frontier_ok, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: stacked lambda_r + lambda_theta per tier
    tids   = [1, 2, 3]
    labels = ["b1\n(High)", "b2\n(Mid)", "b3\n(Low)"]
    lth    = [lambda_theta[t] for t in tids]
    lr     = [lambda_r[t] for t in tids]

    x = np.arange(3)
    w = 0.5
    ax1.bar(x, lr, w, label="lambda_r (radius)", color="#90BE6D")
    ax1.bar(x, lth, w, bottom=lr, label="lambda_theta (angular)", color="#577590")
    for i in range(3):
        total = lth[i] + lr[i]
        ax1.text(x[i], total + 0.01, f"{total:.4f}", ha='center', fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("lambda (coefficient)")
    ax1.set_title("Calibrated lambda per Tier\n(stacked: radius + angular)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # Right: rho ordering
    rho_labels = ["rho(b1->b2)", "rho(b2->b3)"]
    rho_vals   = [rho_12, rho_23]
    rho_colors = ["#4361EE", "#F77F00"]

    bars = ax2.bar(rho_labels, rho_vals, color=rho_colors,
                   edgecolor='black', linewidth=0.5)
    for bar, v in zip(bars, rho_vals):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(rho_vals) * 0.02,
                 f"{v:.6f}", ha='center', fontsize=10)

    status = "PASS: b2 on frontier" if frontier_ok else "FAIL: b2 dominated"
    color  = "green" if frontier_ok else "red"
    ax2.set_title(f"Marginal Efficiency Ordering\n{status}", color=color)
    ax2.set_ylabel("rho (distortion / saved bit)")
    ax2.grid(True, alpha=0.3, axis='y')

    bits_info = (f"b1={tiers[1].token_bits()}b  "
                 f"b2={tiers[2].token_bits()}b  "
                 f"b3={tiers[3].token_bits()}b")
    fig.suptitle(f"(ii) Frontier Validation (C.3)  [{bits_info}]", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_frontier_validation.png", dpi=150)
    plt.close(fig)
    print(f"  Plot (ii) saved")



def _plot_layer_heatmap(layer_errors, num_layers, tiers, out_dir):
    fig, ax = plt.subplots(figsize=(max(10, num_layers * 0.6), 4))

    tids = [1, 2, 3]
    tier_labels = ["b1 (High)", "b2 (Mid)", "b3 (Low)"]

    matrix = np.full((3, num_layers), np.nan)
    for row, tid in enumerate(tids):
        for layer in range(num_layers):
            errs = layer_errors[tid].get(layer, [])
            if errs:
                matrix[row, layer] = np.mean(errs)

    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd',
                   interpolation='nearest')
    ax.set_yticks(range(3))
    ax.set_yticklabels(tier_labels)
    ax.set_xticks(range(num_layers))
    ax.set_xticklabels(range(num_layers), fontsize=7)
    ax.set_xlabel("Layer index")
    ax.set_title("(iii) Mean Angular Error per Layer per Tier (radians)")

    for row in range(3):
        for col in range(num_layers):
            val = matrix[row, col]
            if not np.isnan(val):
                txt_color = 'white' if val > np.nanmax(matrix) * 0.6 else 'black'
                ax.text(col, row, f"{val:.2f}", ha='center', va='center',
                        fontsize=6, color=txt_color)

    plt.colorbar(im, ax=ax, label="Angular error (rad)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_layer_error_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Plot (iii) saved")


if __name__ == "__main__":
    main() 