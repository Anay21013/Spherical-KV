"""
Primary calibration protocol
Usage:
    python calibrate_lambda.py --model_name_or_path meta-llama/Llama-3.2-1B --codebook_dir codebooks/codebooks_llama_1b --dataset pg19 --S 5 --T 32 --probes_per_step 16 --device cuda --output_dir calibration_results
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
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser(
        description="C.2 primary calibration: KL-based regression")
    p.add_argument("--model_name_or_path", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--codebook_dir",
                   default="codebooks/codebooks_llama_1b")
    p.add_argument("--dataset", default="pg19",
                   help="pg19 or wikitext")
    p.add_argument("--S", type=int, default=5,
                   help="number of held-out prompts to sample")
    p.add_argument("--T", type=int, default=32,
                   help="decode steps per prompt")
    p.add_argument("--prefill_len", type=int, default=256,
                   help="prefill context length per prompt")
    p.add_argument("--probes_per_step", type=int, default=16,
                   help="entries probed per decode step per (layer, head, tier)")
    p.add_argument("--layer_fraction", type=float, default=0.25,
                   help="fraction of layers to probe (evenly spaced)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="calibration_results")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()



def load_prompt_set(tokenizer, dataset_name, S, prefill_len, T):
    """Load S prompts, each tokenized to at least prefill_len + T tokens."""
    from datasets import load_dataset

    need = prefill_len + T + 32

    if dataset_name == "pg19":
        ds = load_dataset("pg19", split="test", trust_remote_code=True,
                          streaming=True)
        texts = []
        for example in ds:
            texts.append(example["text"])
            if len(texts) >= S + 5:
                break
    elif dataset_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in ds["text"] if t and len(t) > 200]
    else:
        ds = load_dataset(dataset_name, split="test")
        texts = [t for t in ds["text"] if t and len(t) > 200]

    prompts = []
    for text in texts:
        ids = tokenizer.encode(text, return_tensors="pt").squeeze(0)
        if ids.numel() >= need:
            prompts.append(ids[:need].unsqueeze(0))
        if len(prompts) >= S:
            break

    if len(prompts) < S:
        print(f"[WARNING] Only {len(prompts)}/{S} prompts had enough tokens")
    return prompts



def compute_epsilon_theta(codebooks, num_layers, num_kv_heads, tiers,
                          n_samples=256, seed=42):
    """
    Compute mean angular quantization error per tier from codebook geometry.
    Used to split combined calibrated coefficient into lambda_r and lambda_theta.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    nondrop = [t for t in tiers if t.tier_id != 0]
    eps_theta = {}

    for tier in nondrop:
        tid = tier.tier_id
        g   = tier.g
        errors = []
        for layer in range(num_layers):
            for head in range(num_kv_heads):
                cb = codebooks.get((layer, head, tid))
                if cb is None:
                    continue
                if cb.dim() == 2:
                    cb = cb.unsqueeze(0)
                for gi in range(cb.shape[0]):
                    cw = cb[gi]
                    cw_n = cw / (cw.norm(dim=-1, keepdim=True) + 1e-8)
                    q = torch.randn(n_samples, g, generator=rng)
                    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
                    sims = q @ cw_n.T
                    best = cw_n[sims.argmax(dim=-1)]
                    cos_a = (q * best).sum(-1).clamp(-1, 1)
                    errors.append(torch.acos(cos_a).mean().item())
        eps_theta[tid] = sum(errors) / len(errors) if errors else 0.3
    return eps_theta



def probe_entries(
    K_dense,         # [T_ctx, dh]
    q_vec,           # [dh]
    attn_probs,      # [T_ctx] post-softmax attention for this query
    codebooks_lh,    # [G, cb_size, g]
    tier,
    entry_indices,   # positions to probe
    head_dim,
):
    device = K_dense.device
    T_ctx  = K_dense.shape[0]
    inv_sqrt_d = 1.0 / math.sqrt(head_dim)

    # Dense logits and probs (already computed outside, but we need logits too)
    dense_logits = (q_vec @ K_dense.T) / math.sqrt(head_dim)
    dense_probs  = F.softmax(dense_logits, dim=-1)

    cb = codebooks_lh.to(device)
    results = []

    for idx in entry_indices:
        idx = int(idx)
        if idx >= T_ctx or idx < 0:
            continue

        # omega_i: attention probability for this entry (C.2 importance weight)
        omega_i = float(attn_probs[idx])

        # Encode this entry at tier b
        k_entry   = K_dense[idx].unsqueeze(0)                    # [1, dh]
        k_grouped = k_entry.view(1, tier.G, tier.g)
        r_groups  = k_grouped.norm(dim=-1)                        # [1, G]
        k_dir     = k_grouped / (r_groups.unsqueeze(-1) + 1e-6)

        sims = torch.einsum('ngi,gci->ngc', k_dir, cb)           # [1, G, cb]
        theta_codes = sims.argmax(dim=-1)                          # [1, G]

        # Tiered logit via code-domain (B.2 compliant)
        q_groups = q_vec.view(tier.G, tier.g)
        q_norms  = q_groups.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        q_normed = q_groups / q_norms

        cw = cb[torch.arange(tier.G, device=device),
                theta_codes[0].long()]                             # [G, g]
        dots = (q_normed * cw).sum(-1).clamp(-1.0, 1.0)           # [G]
        tiered_logit = (r_groups[0] * dots).sum() / math.sqrt(head_dim)

        # Replace only entry idx
        tiered_logits      = dense_logits.clone()
        tiered_logits[idx] = tiered_logit
        tiered_probs       = F.softmax(tiered_logits, dim=-1)

        # KL divergence: D_meas
        kl = F.kl_div(tiered_probs.log().clamp(min=-100),
                       dense_probs, reduction='sum').item()

        # delta feature: (1/sqrt(d)) * sum_j |r_hat_j|
        delta_feature = inv_sqrt_d * float(r_groups[0].abs().sum())

        results.append((omega_i, delta_feature, kl))

    return results


def main():
    args = parse_args()
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)

    from evaluate import load_model_and_tokenizer
    from codebook_loader import load_codebooks
    from tiers import build_tiers
    from config import BR

    torch.manual_seed(args.seed)
    device  = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Load model and codebooks
    model, tokenizer = load_model_and_tokenizer(
        args.model_name_or_path, device)
    model.eval()

    cfg          = model.config
    num_layers   = cfg.num_hidden_layers
    num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim     = getattr(cfg, "head_dim",
                           cfg.hidden_size // cfg.num_attention_heads)
    tiers     = build_tiers(head_dim)
    codebooks = load_codebooks(args.codebook_dir, num_layers,
                               num_kv_heads, tiers)

    # -- Codebook epsilon_theta for lambda split
    print("\n[calibrate] Computing epsilon_theta from codebooks ...")
    eps_theta = compute_epsilon_theta(codebooks, num_layers,
                                      num_kv_heads, tiers)
    eps_r = 1.0 / (2 ** BR)   # radius quant error, same for all tiers
    for tid in [1, 2, 3]:
        print(f"  Tier {tid}: eps_theta={eps_theta[tid]:.4f} rad  "
              f"eps_r={eps_r:.6f}")

    # -- Load S prompts
    print(f"\n[calibrate] Loading {args.S} prompts from '{args.dataset}' ...")
    prompts = load_prompt_set(tokenizer, args.dataset, args.S,
                              args.prefill_len, args.T)
    print(f"[calibrate] Loaded {len(prompts)} prompts, "
          f"each {args.prefill_len}+{args.T} tokens")

    # -- Select layer subset
    n_probe_layers = max(1, int(num_layers * args.layer_fraction))
    layer_stride   = max(1, num_layers // n_probe_layers)
    layer_indices  = list(range(0, num_layers, layer_stride))[:n_probe_layers]
    print(f"[calibrate] Probing layers: {layer_indices}")

    # -- Budget counters
    total_tokens_decoded = 0
    total_probes         = 0

    # -- Collect measurements
    # Per tier: list of (omega, delta_feature, kl_measured, depth_frac)
    data = {1: [], 2: [], 3: []}

    for s_idx, prompt_ids in enumerate(prompts):
        prompt_ids  = prompt_ids.to(device)
        prefill_ids = prompt_ids[:, :args.prefill_len]

        print(f"\n[calibrate] Prompt {s_idx+1}/{len(prompts)}")

        with torch.no_grad():
            out = model(prefill_ids, use_cache=True, return_dict=True)
        kv = out.past_key_values

        current_ids = prefill_ids.clone()
        for step in range(args.T):
            with torch.no_grad():
                out_step = model(input_ids=current_ids[:, -1:],
                                 past_key_values=kv,
                                 use_cache=True, return_dict=True)
            kv      = out_step.past_key_values
            next_id = out_step.logits[:, -1, :].argmax(-1, keepdim=True)
            current_ids = torch.cat([current_ids, next_id], dim=-1)
            total_tokens_decoded += 1

            T_ctx = args.prefill_len + step + 1

            for li in layer_indices:
                if hasattr(kv, 'layers'):
                    K_layer = kv.layers[li].keys
                else:
                    K_layer = kv[li][0]

                for h in range(num_kv_heads):
                    K_lh = K_layer[0, h, :T_ctx].float()   # [T_ctx, dh]
                    q_vec = K_lh[-1]                        # last position as query

                    # Compute attention probs for omega (C.2)
                    with torch.no_grad():
                        attn_logits = (q_vec @ K_lh.T) / math.sqrt(head_dim)
                        attn_probs  = F.softmax(attn_logits, dim=-1)

                    # Sample positions (exclude last = query itself)
                    n_probe = min(args.probes_per_step, T_ctx - 1)
                    probe_idx = torch.randperm(T_ctx - 1)[:n_probe].tolist()
                    depth_frac = li / max(num_layers - 1, 1)

                    for tid in [1, 2, 3]:
                        tier = tiers[tid]
                        cb = codebooks.get((li, h, tid))
                        if cb is None:
                            continue

                        entries = probe_entries(
                            K_lh, q_vec, attn_probs, cb, tier,
                            probe_idx, head_dim)

                        for (omega, delta_f, kl) in entries:
                            data[tid].append((omega, delta_f, kl, depth_frac))
                            total_probes += 1

            if (step + 1) % 8 == 0:
                print(f"    step {step+1}/{args.T}  "
                      f"probes={total_probes:,}")

        del kv
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- Budget report
    print(f"\n{'='*60}")
    print(f"CALIBRATION BUDGET (C.2)")
    print(f"  Prompts (S):           {len(prompts)}")
    print(f"  Decode steps (T):      {args.T}")
    print(f"  Total tokens decoded:  {total_tokens_decoded}")
    print(f"  Total probe evals:     {total_probes:,}")
    print(f"  Layers probed:         {layer_indices}")
    print(f"  Heads probed:          all {num_kv_heads}")
    print(f"  Probes/step/tier:      {args.probes_per_step}")
    print(f"{'='*60}")

    # -- Fit coefficients
    # Model: D_meas = combined(b) * (omega * delta_feature)
    # Feature x_i = omega_i * delta_feature_i
    # Target  y_i = KL_measured_i
    # Least squares: combined = (X^T Y) / (X^T X)
    print("\nFitting coefficients ...")
    calibrated = {}

    for tid in [1, 2, 3]:
        samples = data[tid]
        if not samples:
            calibrated[str(tid)] = {
                "lambda_r": 0.01, "lambda_theta": 0.10, "combined": 0.11,
                "n_samples": 0}
            print(f"  Tier {tid}: no data, using defaults")
            continue

        omegas = np.array([s[0] for s in samples])
        deltas = np.array([s[1] for s in samples])
        kls    = np.array([s[2] for s in samples])

        # x = omega * delta  (the full proxy feature)
        X = omegas * deltas
        Y = kls

        # Least squares: Y = combined * X
        combined = float(np.dot(X, Y) / (np.dot(X, X) + 1e-12))
        combined = max(combined, 1e-6)

        # Split using codebook epsilon_theta and radius epsilon_r
        et = eps_theta[tid]
        er = eps_r
        total_eps = et + er
        lam_theta = combined * (et / total_eps)
        lam_r     = combined * (er / total_eps)

        # R^2
        Y_pred  = combined * X
        ss_res  = np.sum((Y - Y_pred) ** 2)
        ss_tot  = np.sum((Y - Y.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-12) if ss_tot > 0 else 0

        calibrated[str(tid)] = {
            "lambda_r":     round(lam_r, 6),
            "lambda_theta": round(lam_theta, 6),
            "combined":     round(combined, 6),
            "r_squared":    round(r2, 4),
            "n_samples":    len(samples),
        }
        print(f"  Tier {tid}: lam_r={lam_r:.6f}  lam_th={lam_theta:.6f}  "
              f"combined={combined:.6f}  R2={r2:.4f}  n={len(samples)}")

    # eta: drop penalty > max predicted distortion
    max_pred = 0.0
    for tid in [1, 2, 3]:
        samples = data[tid]
        if not samples:
            continue
        comb = calibrated[str(tid)]["combined"]
        for (o, d, k, _) in samples:
            max_pred = max(max_pred, comb * o * d)
    eta = max(max_pred * 2.0, 1.0)
    calibrated["eta"] = round(eta, 4)
    print(f"  eta (drop penalty): {eta:.4f}")

    # -- Save
    calib_path = out_dir / "calibrated_lambdas.json"
    with open(calib_path, 'w') as f:
        json.dump(calibrated, f, indent=2)
    print(f"\nSaved to {calib_path}")

    # -- Plots
    print("\nGenerating plots ...")
    _plot_scatter(data, calibrated, out_dir)
    _plot_reliability(data, calibrated, out_dir)
    _plot_depth_stratification(data, calibrated, out_dir)

    # -- Config instructions
    print(f"\nUpdate config.py:")
    for tid_s, vals in calibrated.items():
        if tid_s == "eta":
            print(f"  ETA = {vals}")
        else:
            print(f"  LAMBDA_THETA[{tid_s}] = {vals['lambda_theta']}")
            print(f"  LAMBDA_R[{tid_s}]     = {vals['lambda_r']}")



_TIER_NAMES  = {1: "b1 (High/6-bit)", 2: "b2 (Mid/4-bit)", 3: "b3 (Low/3-bit)"}
_TIER_COLORS = {1: "#2DC653", 2: "#4361EE", 3: "#F77F00"}


def _predicted_and_measured(samples, combined):
    """Compute Y_pred = combined * omega * delta, Y_meas = KL."""
    omegas = np.array([s[0] for s in samples])
    deltas = np.array([s[1] for s in samples])
    kls    = np.array([s[2] for s in samples])
    Y_pred = combined * omegas * deltas
    return Y_pred, kls


# (i) Scatter: D_predicted vs D_measured per tier
def _plot_scatter(data, calibrated, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax_idx, tid in enumerate([1, 2, 3]):
        ax      = axes[ax_idx]
        samples = data[tid]
        if not samples:
            ax.set_title(f"{_TIER_NAMES[tid]}\n(no data)")
            continue
        comb = calibrated[str(tid)]["combined"]
        r2   = calibrated[str(tid)]["r_squared"]
        Y_pred, Y_meas = _predicted_and_measured(samples, comb)

        ax.scatter(Y_pred, Y_meas, alpha=0.15, s=6, c=_TIER_COLORS[tid],
                   edgecolors='none')
        hi = max(np.percentile(Y_pred, 99), np.percentile(Y_meas, 99))
        ax.plot([0, hi], [0, hi], 'k--', lw=1, alpha=0.5, label="y=x")
        ax.set_xlim(0, hi * 1.05)
        ax.set_ylim(0, hi * 1.05)
        ax.set_title(f"{_TIER_NAMES[tid]}\nR2={r2:.4f}  n={len(samples):,}")
        ax.set_xlabel("D_predicted = combined * omega * delta")
        ax.set_ylabel("D_measured (KL)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("(i) Predicted vs Measured Distortion per Tier (C.2)",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_scatter_predicted_vs_measured.png", dpi=150)
    plt.close(fig)
    print(f"  Plot (i) saved")


# (ii) Reliability: binned by predicted distortion quantile
def _plot_reliability(data, calibrated, out_dir, n_bins=10):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax_idx, tid in enumerate([1, 2, 3]):
        ax      = axes[ax_idx]
        samples = data[tid]
        if len(samples) < n_bins * 2:
            ax.set_title(f"{_TIER_NAMES[tid]}\n(insufficient data)")
            continue
        comb = calibrated[str(tid)]["combined"]
        Y_pred, Y_meas = _predicted_and_measured(samples, comb)

        quantiles = np.percentile(Y_pred, np.linspace(0, 100, n_bins + 1))
        bp, bm, bs = [], [], []
        for b in range(n_bins):
            lo, hi = quantiles[b], quantiles[b + 1]
            mask = (Y_pred >= lo) & (Y_pred <= hi if b == n_bins-1 else Y_pred < hi)
            if mask.sum() < 2:
                continue
            bp.append(Y_pred[mask].mean())
            bm.append(Y_meas[mask].mean())
            bs.append(Y_meas[mask].std())

        bp, bm, bs = np.array(bp), np.array(bm), np.array(bs)
        ax.errorbar(bp, bm, yerr=bs, fmt='o-', color=_TIER_COLORS[tid],
                    capsize=3, markersize=5)
        hi = max(bp.max(), bm.max()) if len(bp) else 1
        ax.plot([0, hi], [0, hi], 'k--', lw=1, alpha=0.5)
        ax.set_title(f"{_TIER_NAMES[tid]}")
        ax.set_xlabel("Mean predicted D (quantile bin)")
        ax.set_ylabel("Mean measured D (KL)")
        ax.grid(True, alpha=0.3)

    fig.suptitle("(ii) Reliability: Binned by Predicted Distortion Quantile",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_reliability_binned.png", dpi=150)
    plt.close(fig)
    print(f"  Plot (ii) saved")


# (iii) Depth stratification: early / mid / late layers
def _plot_depth_stratification(data, calibrated, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    depth_bins = [
        ("Early (0-33%)",  0.0, 0.333, "#66c2a5"),
        ("Mid (33-66%)",   0.333, 0.666, "#fc8d62"),
        ("Late (66-100%)", 0.666, 1.001, "#8da0cb"),
    ]
    for ax_idx, tid in enumerate([1, 2, 3]):
        ax      = axes[ax_idx]
        samples = data[tid]
        if not samples:
            ax.set_title(f"{_TIER_NAMES[tid]}\n(no data)")
            continue
        comb = calibrated[str(tid)]["combined"]

        for label, lo, hi, color in depth_bins:
            subset = [s for s in samples if lo <= s[3] < hi]
            if not subset:
                continue
            Y_pred, Y_meas = _predicted_and_measured(subset, comb)
            ax.scatter(Y_pred, Y_meas, alpha=0.15, s=6, c=color,
                       label=f"{label} (n={len(subset):,})",
                       edgecolors='none')

        all_kl = [s[2] for s in samples]
        hi = max(np.percentile(all_kl, 99), 1e-6)
        ax.plot([0, hi], [0, hi], 'k--', lw=1, alpha=0.5)
        ax.set_xlim(0, hi * 1.05)
        ax.set_ylim(0, hi * 1.05)
        ax.set_title(f"{_TIER_NAMES[tid]}")
        ax.set_xlabel("D_predicted")
        ax.set_ylabel("D_measured (KL)")
        ax.legend(fontsize=7, markerscale=3)
        ax.grid(True, alpha=0.3)

    fig.suptitle("(iii) Depth Stratification: Early / Mid / Late Layers",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_depth_stratification.png", dpi=150)
    plt.close(fig)
    print(f"  Plot (iii) saved")


if __name__ == "__main__":
    main()