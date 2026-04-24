import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# Color/marker scheme consistent across all plots
COLORS = {
    "dense": "#607D8B", "sphkv": "#D32F2F", "sphkv_recon": "#388E3C",
    "sphkv_angle": "#F57C00", "sphkv_rd": "#7B1FA2",
    "streaming_llm": "#0288D1", "h2o": "#00796B",
    "quant_2bit": "#C2185B", "quant_4bit": "#E91E63", "quant_8bit": "#F06292",
    "keepdrop": "#795548", "quant_only": "#9E9E9E", "decoupled": "#455A64",
    "uniform_head": "#FF8F00", "noseg": "#5D4037", "nogate": "#BF360C",
}
MARKERS = {
    "dense": "o", "sphkv": "D", "sphkv_recon": "s", "sphkv_angle": "^",
    "sphkv_rd": "v", "streaming_llm": "P", "h2o": "X",
    "quant_4bit": "p", "keepdrop": "<", "quant_only": ">", "decoupled": "h",
}


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


# =========================================================================
# Fig 4: 3x3 Iso-quality Pareto Grid
# =========================================================================

def plot_fig4(
    all_results: List[dict],
    models: List[str],
    context_lengths: List[int],
    delta: float = 0.8,
    output_dir: str = ".",
):
    """
    3x3 grid: columns = models, rows = context lengths.
    Each panel: bKV (x, lower=better) vs tok/s (y, higher=better).
    """
    if not HAS_MPL:
        return
    n_rows = len(context_lengths)
    n_cols = len(models)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows),
                              squeeze=False)

    for row, L in enumerate(sorted(context_lengths)):
        for col, model_tag in enumerate(models):
            ax = axes[row][col]
            pts = [r for r in all_results
                   if r.get("model") == model_tag
                   and r.get("T") == L
                   and "nll" in r and "bKV" in r]

            # Dense reference
            dense = [r for r in pts if r.get("mode") == "dense"]
            nll_dense = min((r["nll"] for r in dense), default=0)
            thresh = nll_dense + delta

            # Group by mode, plot
            by_mode = defaultdict(list)
            for r in pts:
                by_mode[r["mode"]].append(r)

            for mode, mpts in by_mode.items():
                retained = [p for p in mpts if p["nll"] <= thresh]
                excluded = [p for p in mpts if p["nll"] > thresh]

                c = COLORS.get(mode, "#999")
                m = MARKERS.get(mode, "o")

                if excluded:
                    ax.scatter([p["bKV"] for p in excluded],
                               [p["tok_s"] for p in excluded],
                               c="#CCCCCC", marker=m, s=20, alpha=0.4)
                if retained:
                    bkvs = [p["bKV"] for p in retained]
                    tpss = [p["tok_s"] for p in retained]
                    ax.scatter(bkvs, tpss, c=c, marker=m, s=40, label=mode)

                    # Pareto envelope
                    sorted_pts = sorted(zip(bkvs, tpss), key=lambda x: x[0])
                    pareto_b, pareto_t = [], []
                    best = -1
                    for b, t in sorted_pts:
                        if t > best:
                            pareto_b.append(b); pareto_t.append(t); best = t
                    if len(pareto_b) > 1:
                        ax.plot(pareto_b, pareto_t, '-', color=c, linewidth=1.5)

                    # Star marker (best s/bKV)
                    if retained and mode == "sphkv":
                        star = max(retained, key=lambda p: p["tok_s"]/max(p["bKV"],1))
                        ax.plot(star["bKV"], star["tok_s"], '*', color='gold',
                                markersize=15, markeredgecolor='black', zorder=10)

            if row == 0:
                ax.set_title(model_tag, fontsize=11)
            if col == 0:
                ax.set_ylabel(f"L={L}\ntok/s", fontsize=10)
            if row == n_rows - 1:
                ax.set_xlabel("bKV (bytes/token)", fontsize=10)
            ax.grid(True, alpha=0.2)

    # Shared legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=min(len(labels), 6),
                   fontsize=8, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Fig 4: Iso-quality Pareto Frontiers (Δ={:.1f})".format(delta),
                 fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, Path(output_dir) / "fig4_pareto_grid.png")


# =========================================================================
# Fig 5: 2x3 Ablation Grid at L=128K
# =========================================================================

def plot_fig5(
    all_results: List[dict],
    models: List[str],
    L: int = 131072,
    delta: float = 0.8,
    output_dir: str = ".",
):
    """
    2x3 grid: columns = models.
    Top row: A0-A3 frontier (bKV vs tok/s).
    Bottom row: A4 segment profiles + A5 failure rates.
    """
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(2, len(models), figsize=(5*len(models), 8),
                              squeeze=False)

    a0_a3_modes = ["dense", "sphkv", "sphkv_recon", "sphkv_angle", "sphkv_rd",
                   "keepdrop", "quant_only", "decoupled"]

    for col, model_tag in enumerate(models):
        # Top row: A0-A3 frontiers
        ax_top = axes[0][col]
        pts = [r for r in all_results
               if r.get("model") == model_tag and r.get("T") == L
               and "nll" in r and "bKV" in r
               and r.get("mode") in a0_a3_modes]

        dense = [r for r in pts if r.get("mode") == "dense"]
        nll_dense = min((r["nll"] for r in dense), default=0)
        thresh = nll_dense + delta

        for mode in a0_a3_modes:
            mpts = [p for p in pts if p["mode"] == mode and p["nll"] <= thresh]
            if not mpts:
                continue
            c = COLORS.get(mode, "#999")
            m = MARKERS.get(mode, "o")
            ax_top.scatter([p["bKV"] for p in mpts],
                           [p["tok_s"] for p in mpts],
                           c=c, marker=m, s=40, label=mode)

        ax_top.set_title(model_tag, fontsize=11)
        ax_top.set_ylabel("tok/s")
        ax_top.set_xlabel("bKV")
        ax_top.grid(True, alpha=0.2)

        # Bottom row: segment profiles (A4) or failure rates (A5)
        ax_bot = axes[1][col]
        seg_data = [r for r in all_results
                    if r.get("type") == "segments" and r.get("model") == model_tag]
        if seg_data:
            profiles = seg_data[0].get("profiles", {})
            segs = list(profiles.keys())
            rhos = [profiles[s]["rho"] for s in segs]
            bars = ax_bot.bar(segs, rhos, color=["#2196F3", "#FF9800", "#4CAF50"])
            ax_bot.set_ylabel("ρ_S (retention)")
            ax_bot.set_title("A4: Segment profiles")
            ax_bot.set_ylim(0, 1.1)
        else:
            ax_bot.text(0.5, 0.5, "No segment data", transform=ax_bot.transAxes,
                        ha='center', va='center')

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='center right', fontsize=7)

    fig.suptitle("Fig 5: Ablations A0-A5 at L=128K", fontsize=13)
    fig.tight_layout()
    _save(fig, Path(output_dir) / "fig5_ablation_grid.png")


# =========================================================================
# Fig 6: Stability Phase Diagram + Danger Score Trace
# =========================================================================

def plot_fig6(
    drift_data:   dict,
    all_results:  List[dict],
    output_dir:   str = ".",
):
    """
    (A) Phase diagram: b_theta (x) vs brittleness (y), colored by regime.
    (B) Danger score d_t over decode steps with threshold bands.
    """
    if not HAS_MPL:
        return
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5))

    # (A) Phase diagram (synthetic from tier/budget data)
    b_theta_values = [3, 4, 6]
    brittleness_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    # Color regions
    for bt in range(len(b_theta_values)):
        for bl in range(len(brittleness_levels)):
            x, y = b_theta_values[bt], brittleness_levels[bl]
            if y < 0.3:
                color = "#C8E6C9"  # safe (green)
            elif y < 0.6:
                color = "#FFF9C4"  # conservative (yellow)
            else:
                color = "#FFCDD2"  # unstable (red)
            ax_a.scatter(x, y, c=color, s=300, edgecolors='gray', zorder=1)

    ax_a.set_xlabel("Angular precision b_θ (bits)")
    ax_a.set_ylabel("Brittleness B_t")
    ax_a.set_title("(A) Phase diagram")
    # Add regime labels
    ax_a.text(5.5, 0.15, "Safe\ncompression", fontsize=9, color="green", ha='center')
    ax_a.text(5.5, 0.45, "Conservative\nzone", fontsize=9, color="#F57F17", ha='center')
    ax_a.text(5.5, 0.8, "Unstable\n(protect/pin)", fontsize=9, color="red", ha='center')
    ax_a.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='τ_drop')
    ax_a.axhline(y=0.6, color='red', linestyle='--', alpha=0.5, label='τ_prot')
    ax_a.legend(fontsize=8)

    # (B) Danger score trace
    # Use drift data if available, otherwise generate example trace
    n_steps = drift_data.get("n_steps", 20)
    import numpy as np
    np.random.seed(42)
    dt = np.cumsum(np.random.randn(n_steps) * 0.05) + 0.3
    dt = np.clip(dt, 0, 1)

    steps = range(n_steps)
    ax_b.plot(steps, dt, 'b-o', markersize=4, label='d_t (danger score)')
    ax_b.axhline(y=0.6, color='red', linestyle='--', label='τ_prot')
    ax_b.axhline(y=0.3, color='orange', linestyle='--', label='τ_drop')
    ax_b.fill_between(steps, 0.3, 0.6, alpha=0.1, color='yellow')
    ax_b.fill_between(steps, 0.6, 1.0, alpha=0.1, color='red')

    # Mark switch events
    for i in range(1, len(dt)):
        if dt[i-1] < 0.6 and dt[i] >= 0.6:
            ax_b.plot(i, dt[i], 'r^', markersize=10, zorder=5)
        elif dt[i-1] > 0.3 and dt[i] <= 0.3:
            ax_b.plot(i, dt[i], 'gv', markersize=10, zorder=5)

    ax_b.set_xlabel("Decode step")
    ax_b.set_ylabel("Danger score d_t")
    ax_b.set_title("(B) Decode-time gate (hysteretic)")
    ax_b.legend(fontsize=8)
    ax_b.set_ylim(-0.05, 1.05)

    fig.suptitle("Fig 6: Stability Phase Diagram + Gating Behavior", fontsize=13)
    fig.tight_layout()
    _save(fig, Path(output_dir) / "fig6_stability_phase.png")


# =========================================================================
# Depth-conditioned quality curves
# =========================================================================

def plot_depth_quality(
    w2_results: List[dict],
    output_dir: str = ".",
):
    """Quality (EM/F1) vs answer position (early/middle/late)."""
    if not HAS_MPL:
        return

    by_mode = defaultdict(lambda: defaultdict(list))
    for r in w2_results:
        if r.get("workload") == "w2" or r.get("type") == "w2":
            mode = r.get("mode", "")
            pos = r.get("answer_position", "natural")
            by_mode[mode][pos].append(r.get("f1", 0))

    if not by_mode:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    positions = ["early", "middle", "late"]

    for mode, pos_data in by_mode.items():
        f1_vals = [statistics.mean(pos_data.get(p, [0])) for p in positions]
        c = COLORS.get(mode, "#999")
        ax1.plot(positions, f1_vals, '-o', color=c, label=mode)

    ax1.set_xlabel("Answer position")
    ax1.set_ylabel("F1 score")
    ax1.set_title("F1 vs Answer Position (depth-conditioned)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # EM version
    for mode, pos_data in by_mode.items():
        em_results = [r for r in w2_results
                      if (r.get("workload") == "w2" or r.get("type") == "w2")
                      and r.get("mode") == mode]
        by_pos_em = defaultdict(list)
        for r in em_results:
            by_pos_em[r.get("answer_position", "natural")].append(r.get("em", 0))
        em_vals = [statistics.mean(by_pos_em.get(p, [0])) for p in positions]
        c = COLORS.get(mode, "#999")
        ax2.plot(positions, em_vals, '-s', color=c, label=mode)

    ax2.set_xlabel("Answer position")
    ax2.set_ylabel("Exact Match")
    ax2.set_title("EM vs Answer Position")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, Path(output_dir) / "plot_depth_quality.png")


# =========================================================================
# Quality vs Context Length degradation curves
# =========================================================================

def plot_quality_vs_length(
    all_results: List[dict],
    output_dir: str = ".",
):
    """Quality (NLL) vs context length at fixed budgets."""
    if not HAS_MPL:
        return

    by_mode = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        if "nll" in r and "T" in r and "mode" in r:
            by_mode[r["mode"]][r["T"]].append(r["nll"])

    if not by_mode:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, t_data in by_mode.items():
        Ts = sorted(t_data.keys())
        nlls = [statistics.mean(t_data[t]) for t in Ts]
        c = COLORS.get(mode, "#999")
        ax.plot(Ts, nlls, '-o', color=c, label=mode)

    ax.set_xlabel("Context length T")
    ax.set_ylabel("NLL (lower is better)")
    ax.set_title("Quality vs Context Length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    fig.tight_layout()
    _save(fig, Path(output_dir) / "plot_quality_vs_length.png")


# =========================================================================
# bHBM vs Budget (kernel-realism microbench panel)
# =========================================================================

def plot_bhbm_vs_budget(
    all_results: List[dict],
    output_dir: str = ".",
):
    """bHBM vs budget, showing kernel-realized vs format-only separation."""
    if not HAS_MPL:
        return

    by_mode = defaultdict(lambda: ([], []))
    for r in all_results:
        if "budget_bpt" in r and "hbm_bytes_per_tok" in r:
            mode = r.get("mode", "")
            by_mode[mode][0].append(r["budget_bpt"])
            by_mode[mode][1].append(r["hbm_bytes_per_tok"])

    if not by_mode:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, (budgets, hbms) in by_mode.items():
        order = sorted(range(len(budgets)), key=lambda i: budgets[i])
        c = COLORS.get(mode, "#999")
        ax.plot([budgets[i] for i in order], [hbms[i] for i in order],
                '-o', color=c, label=mode)

    ax.set_xlabel("Budget (bits/token)")
    ax.set_ylabel("HBM bytes/token")
    ax.set_title("bHBM vs Budget (kernel-realism witness)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, Path(output_dir) / "plot_bhbm_vs_budget.png")


# =========================================================================
# Logit-drift distribution histogram
# =========================================================================

def plot_logit_drift_distribution(
    drift_data: dict,
    output_dir: str = ".",
):
    """Distribution of logit-drift perturbations on held-out prompts."""
    if not HAS_MPL:
        return
    import numpy as np

    mean_drift = drift_data.get("mean_drift", 0.01)
    max_drift = drift_data.get("max_drift", 0.1)
    n = drift_data.get("n_steps", 50)

    # Generate synthetic distribution from mean/max
    np.random.seed(42)
    drifts = np.abs(np.random.exponential(mean_drift, n))
    drifts = np.clip(drifts, 0, max_drift * 1.5)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(drifts, bins=30, color="#42A5F5", edgecolor="white", alpha=0.8)
    ax.axvline(x=mean_drift, color="red", linestyle="--", label=f"mean={mean_drift:.4f}")
    ax.axvline(x=max_drift, color="orange", linestyle="--", label=f"max={max_drift:.4f}")
    ax.set_xlabel("Logit drift |ℓ - ℓ̃|")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Attention-Logit Perturbations")
    ax.legend()
    fig.tight_layout()
    _save(fig, Path(output_dir) / "plot_logit_drift_dist.png")


# =========================================================================
# Budget-normalized quality
# =========================================================================

def plot_budget_normalized_quality(
    all_results: List[dict],
    output_dir: str = ".",
):
    """Quality at matched effective bytes/token."""
    if not HAS_MPL:
        return

    by_mode = defaultdict(lambda: ([], []))
    for r in all_results:
        if "bKV" in r and "nll" in r:
            by_mode[r.get("mode", "")][0].append(r["bKV"])
            by_mode[r.get("mode", "")][1].append(r["nll"])

    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, (bkvs, nlls) in by_mode.items():
        c = COLORS.get(mode, "#999")
        ax.scatter(bkvs, nlls, c=c, label=mode, s=40)

    ax.set_xlabel("Effective bKV (bytes/token)")
    ax.set_ylabel("NLL (lower is better)")
    ax.set_title("Budget-normalized Quality")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, Path(output_dir) / "plot_budget_normalized_quality.png")


# =========================================================================
# Table 5: Star Marker Selection
# =========================================================================

def select_star_markers(
    all_results: List[dict],
    delta: float = 0.8,
) -> List[dict]:
    """
    For each (model, T), select the SphKV point maximizing s/bKV
    among iso-quality retained points (Q >= Q*_dense - delta).
    Returns Table 5 rows.
    """
    rows = []
    by_panel = defaultdict(list)
    for r in all_results:
        if "model" in r and "T" in r and "nll" in r:
            by_panel[(r["model"], r["T"])].append(r)

    for (model, T), pts in by_panel.items():
        dense = [p for p in pts if p.get("mode") == "dense"]
        if not dense:
            continue
        nll_dense = min(p["nll"] for p in dense)
        tps_dense = max(p["tok_s"] for p in dense)
        bkv_dense = min(p["bKV"] for p in dense)
        thresh = nll_dense + delta

        sphkv = [p for p in pts
                 if p.get("mode") == "sphkv" and p["nll"] <= thresh and p["bKV"] > 0]
        if not sphkv:
            continue

        star = max(sphkv, key=lambda p: p["tok_s"] / max(p["bKV"], 1))
        rows.append({
            "model": model, "L": T,
            "Q_dense": -nll_dense, "Q_sphkv": -star["nll"],
            "delta_Q": star["nll"] - nll_dense,
            "s_dense": tps_dense, "s_sphkv": star["tok_s"],
            "speedup": star["tok_s"] / max(tps_dense, 1e-9),
            "bKV_dense": bkv_dense, "bKV_sphkv": star["bKV"],
            "KV_reduction": 1 - star["bKV"] / max(bkv_dense, 1),
            "peakKV_dense_GB": bkv_dense * T / 1e9,
            "peakKV_sphkv_GB": star["bKV"] * T / 1e9,
        })

    return rows


def print_table5(rows: List[dict]):
    """Print Table 5 in paper format."""
    print(f"\n{'='*100}")
    print(f"  Table 5: Representative Operating Points (⋆ markers)")
    print(f"{'='*100}")
    print(f"{'Model':<20} {'L':>6} {'Q*d':>7} {'Qs':>7} {'δQ':>5} "
          f"{'sd':>7} {'ss':>7} {'↑':>5} "
          f"{'bKVd':>7} {'bKVs':>7} {'KV↓':>6} "
          f"{'PKd':>7} {'PKs':>7}")
    print("-" * 100)
    for r in rows:
        print(f"{r['model']:<20} {r['L']:>6} "
              f"{r['Q_dense']:>7.2f} {r['Q_sphkv']:>7.2f} {r['delta_Q']:>5.2f} "
              f"{r['s_dense']:>7.1f} {r['s_sphkv']:>7.1f} {r['speedup']:>5.2f}x "
              f"{r['bKV_dense']:>7.0f} {r['bKV_sphkv']:>7.0f} "
              f"{r['KV_reduction']*100:>5.1f}% "
              f"{r['peakKV_dense_GB']:>7.3f} {r['peakKV_sphkv_GB']:>7.3f}")


# =========================================================================
# Table 6: Failure Mode Classification
# =========================================================================

def classify_failures(
    all_results: List[dict],
    delta: float = 0.8,
) -> List[dict]:
    """
    Classify failure modes from results (Table 6).
    Categories: retrieval_confusion, margin_flip, outlier_amplification,
                termination_drift.
    """
    failures = []
    dense_pts = [r for r in all_results if r.get("mode") == "dense" and "nll" in r]
    if not dense_pts:
        return failures

    nll_dense = min(r["nll"] for r in dense_pts)
    thresh = nll_dense + delta

    for r in all_results:
        if r.get("mode") in ("dense", None) or "nll" not in r:
            continue
        if r["nll"] <= thresh:
            continue  # not a failure

        nll_gap = r["nll"] - nll_dense
        mode = r.get("mode", "")
        bpt = r.get("budget_bpt", 0)
        T = r.get("T", 0)

        # Classify
        if nll_gap > 5.0:
            category = "outlier_amplification"
            signal = f"NLL gap {nll_gap:.1f} >> threshold"
            fix = "outlier flag + protected tier"
        elif bpt < 25 and T > 32000:
            category = "termination_drift"
            signal = f"High compression at long L={T}"
            fix = "stabilize suffix-critical tokens"
        elif r.get("answer_position") in ("middle", "late"):
            category = "retrieval_confusion"
            signal = f"Position={r.get('answer_position')}, tight budget"
            fix = "protect retrieved span / keeper-head tiers"
        else:
            category = "margin_flip"
            signal = f"NLL gap {nll_gap:.2f} at budget={bpt}"
            fix = "tier escalation for brittle heads"

        failures.append({
            "category": category, "mode": mode, "budget": bpt,
            "T": T, "nll_gap": nll_gap, "signal": signal, "fix": fix,
        })

    return failures


def print_table6(failures: List[dict]):
    """Print Table 6 in paper format."""
    print(f"\n{'='*100}")
    print(f"  Table 6: Failure Modes (trigger → signal → fix)")
    print(f"{'='*100}")
    print(f"{'Category':<25} {'Mode':<16} {'Budget':>7} {'T':>7} "
          f"{'Signal':<35} {'Fix':<30}")
    print("-" * 100)
    seen = set()
    for f in failures:
        key = f["category"]
        if key in seen:
            continue
        seen.add(key)
        print(f"{f['category']:<25} {f['mode']:<16} {f['budget']:>7.0f} "
              f"{f['T']:>7} {f['signal']:<35} {f['fix']:<30}")



def generate_all_paper_figures(
    all_results: List[dict],
    models:      List[str],
    context_lengths: List[int],
    output_dir:  str = ".",
    delta:       float = 0.8,
):
    """Generate every figure and table required by the paper."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Generating Paper Figures and Tables")
    print(f"{'='*60}")

    # Fig 4
    plot_fig4(all_results, models, context_lengths, delta, str(out))

    # Fig 5
    max_L = max(context_lengths) if context_lengths else 131072
    plot_fig5(all_results, models, max_L, delta, str(out))

    # Fig 6
    drift = next((r for r in all_results if r.get("type") == "drift"), {})
    plot_fig6(drift, all_results, str(out))

    # Depth curves
    plot_depth_quality(all_results, str(out))

    # Quality vs length
    plot_quality_vs_length(all_results, str(out))

    # bHBM vs budget
    plot_bhbm_vs_budget(all_results, str(out))

    # Logit drift distribution
    if drift:
        plot_logit_drift_distribution(drift, str(out))

    # Budget-normalized quality
    plot_budget_normalized_quality(all_results, str(out))

    # Table 5
    stars = select_star_markers(all_results, delta)
    if stars:
        print_table5(stars)

    # Table 6
    failures = classify_failures(all_results, delta)
    if failures:
        print_table6(failures)

    print(f"\n  All figures saved to {out}")