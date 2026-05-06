import os
import gc
import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Tuple, List

from config import MODEL_NAME, SAVE_DIR, TIERS, DEVICE
from dataset_loader import load_wikipedia_qa_style


SEQ_LEN              = 2048      
NUM_SAMPLES          = 100
MAX_VECS_PER_BUCKET  = 60_000    

TIER_BANDS = {
    1: (0.537, 1.000),   
    2: (0.219, 0.537),   
    3: (0.000, 0.219),   
}

# K-means
NITER_FLAT      = 400
NITER_HIER      = 150
NREDO_FLAT      = 16
NREDO_HIER      = 6
SEED            = 42
USE_KMEANS_PP_INIT = True

ENABLE_GRAD_REFINE = False
REFINE_STEPS       = 150
REFINE_LR          = 0.01

HIER_FACTORIZATION = {
    64: (8, 8),    # b1
    16: (4, 4),    # b2
}


def kmeans_pp_init_spherical(U, K, generator, device):
    N, d = U.shape
    centers = torch.empty(K, d, device=device, dtype=U.dtype)
    idx0 = torch.randint(0, N, (1,), generator=generator, device=device).item()
    centers[0] = U[idx0]
    max_cos = (U @ centers[0:1].T).squeeze(1)
    for i in range(1, K):
        w = (1.0 - max_cos).clamp(min=0.0).pow(2)
        if w.sum() < 1e-12:
            idx = torch.randint(0, N, (1,), generator=generator, device=device).item()
        else:
            idx = torch.multinomial(w, 1, generator=generator).item()
        centers[i] = U[idx]
        max_cos = torch.maximum(max_cos, U @ centers[i])
    return centers


def spherical_kmeans_gpu(U, K, niter, nredo, device="cuda", seed=SEED):
    N, d = U.shape
    if N < K:
        repeats = (K + N - 1) // N
        U = U.repeat(repeats, 1)[:K]
        N = K

    best_centers, best_score = None, -1e9
    g = torch.Generator(device=device).manual_seed(seed)

    for _ in range(nredo):
        if USE_KMEANS_PP_INIT:
            centers = F.normalize(kmeans_pp_init_spherical(U, K, g, device).clone(), dim=1)
        else:
            idx = torch.randperm(N, generator=g, device=device)[:K]
            centers = F.normalize(U[idx].clone(), dim=1)

        prev_assigns = None
        for it in range(niter):
            sims = U @ centers.T
            assigns = sims.argmax(dim=1)
            if prev_assigns is not None and torch.equal(assigns, prev_assigns):
                break
            prev_assigns = assigns
            new_c = torch.zeros_like(centers)
            new_c.index_add_(0, assigns, U)
            counts = torch.bincount(assigns, minlength=K).float().clamp(min=1)
            new_c /= counts.unsqueeze(1)
            empty = (counts < 0.5)
            if empty.any():
                worst = sims.gather(1, assigns.unsqueeze(1)).squeeze(1).topk(
                    int(empty.sum().item()), largest=False).indices
                new_c[empty] = U[worst]
            centers = F.normalize(new_c, dim=1)

        score = (U @ centers.T).max(dim=1).values.mean().item()
        if score > best_score:
            best_score = score
            best_centers = centers.clone()
    return best_centers, best_score


def hierarchical_spherical_kmeans(U, K, device="cuda"):
    if K not in HIER_FACTORIZATION:
        return spherical_kmeans_gpu(U, K, NITER_FLAT, NREDO_FLAT, device)
    K_coarse, K_fine = HIER_FACTORIZATION[K]
    assert K_coarse * K_fine == K
    coarse_centers, _ = spherical_kmeans_gpu(
        U, K_coarse, NITER_HIER, NREDO_HIER, device, seed=SEED)
    sims = U @ coarse_centers.T
    coarse_assigns = sims.argmax(dim=1)
    all_fine_centers = []
    for c in range(K_coarse):
        mask = (coarse_assigns == c)
        n_c = int(mask.sum().item())
        if n_c < K_fine:
            base = coarse_centers[c]
            noise = torch.randn(K_fine, U.shape[1], device=device) * 0.05
            fine_centers = F.normalize(base.unsqueeze(0) + noise, dim=1)
        else:
            U_c = U[mask]
            fine_centers, _ = spherical_kmeans_gpu(
                U_c, K_fine, NITER_HIER, NREDO_HIER, device, seed=SEED + c + 1)
        all_fine_centers.append(fine_centers)
    centers = F.normalize(torch.cat(all_fine_centers, dim=0), dim=-1)
    avg_cos = (U @ centers.T).max(dim=1).values.mean().item()
    return centers, avg_cos


def gradient_refine_centers(U, centers, n_steps=REFINE_STEPS, lr=REFINE_LR):
    centers = centers.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([centers], lr=lr)
    pre = (U @ F.normalize(centers, dim=1).detach().T).max(dim=1).values.mean().item()
    for _ in range(n_steps):
        opt.zero_grad()
        loss = -(U @ F.normalize(centers, dim=1).T).max(dim=1).values.mean()
        loss.backward()
        opt.step()
    final = F.normalize(centers, dim=1).detach()
    post = (U @ final.T).max(dim=1).values.mean().item()
    return final, pre, post

def extract_importance_banded_keys(model, tokenizer, texts, num_layers,
                                   num_kv_heads, num_q_heads, dh, device):
    kv_groups = num_q_heads // num_kv_heads
    accum: Dict[Tuple[int, int, int], List[torch.Tensor]] = {}

    # Distribution sanity stats (layer 0 head 0 only, to avoid spam)
    imp_q10_global, imp_q90_global, n_seen = [], [], 0

    for text in tqdm(texts, desc="extract"):
        tokens = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=SEQ_LEN).to(device)
        if tokens["input_ids"].shape[1] < 64:
            continue

        with torch.no_grad():
            out = model(**tokens, use_cache=True,
                        output_attentions=True, return_dict=True)

        pkv = out.past_key_values
        attns = out.attentions      # tuple of [B, num_q, T, T]

        for li in range(num_layers):
            # K post-RoPE
            if hasattr(pkv, "layers"):
                K_layer = getattr(pkv.layers[li], "keys", None)
                if K_layer is None:
                    K_layer = getattr(pkv.layers[li], "key", None)
            else:
                K_layer = pkv[li][0]
            K_lh = K_layer[0].float()                     # [num_kv, T, dh]

            # Attention → per-key importance under GQA
            attn = attns[li][0].float()                   # [num_q, T, T]
            T = attn.shape[-1]
            imp_q  = attn.sum(dim=1)                      # [num_q, T]
            imp_kv = imp_q.view(num_kv_heads, kv_groups, T).sum(dim=1)  # [num_kv, T]

            for h in range(num_kv_heads):
                imp_h = imp_kv[h]                         # [T]
                # Sort positions ASCENDING (low → high importance)
                sorted_idx = imp_h.argsort()

                for tid, (low_q, high_q) in TIER_BANDS.items():
                    lo = int(T * low_q)
                    hi = int(T * high_q)
                    if hi <= lo:
                        continue
                    band_idx = sorted_idx[lo:hi]
                    K_band = K_lh[h, band_idx].half().cpu()  # fp16 to save RAM
                    accum.setdefault((li, h, tid), []).append(K_band)

                if li == 0 and h == 0:
                    imp_q10_global.append(imp_h.quantile(0.10).item())
                    imp_q90_global.append(imp_h.quantile(0.90).item())

            del attn, imp_q, imp_kv

        n_seen += 1
        del out, pkv, attns
        torch.cuda.empty_cache()
        gc.collect()

    if imp_q10_global:
        q10 = sum(imp_q10_global) / len(imp_q10_global)
        q90 = sum(imp_q90_global) / len(imp_q90_global)
        print(f"\n  [importance] layer0 head0 across {n_seen} samples: "
              f"p10={q10:.4f}  p90={q90:.4f}  ratio={q90 / max(q10, 1e-9):.1f}x")
        print(f"  Per-tier bands (quantile range, fraction of keys):")
        for tid, (lo, hi) in TIER_BANDS.items():
            print(f"    b{tid}: [{lo:.3f}, {hi:.3f}]  ({(hi-lo)*100:.1f}%)")

    # Concatenate per (layer, kv_head, tier)
    out_dict = {}
    for key, parts in accum.items():
        t = torch.cat(parts, dim=0)
        if t.shape[0] > MAX_VECS_PER_BUCKET:
            idx = torch.randperm(t.shape[0])[:MAX_VECS_PER_BUCKET]
            t = t[idx]
        out_dict[key] = t

    return out_dict


def main():
    print("=" * 72)
    print("Importance-banded hierarchical spherical KV codebook training")
    print(f"  SEQ_LEN={SEQ_LEN}  NUM_SAMPLES={NUM_SAMPLES}  "
          f"CAP_per_tier={MAX_VECS_PER_BUCKET:,}")
    print(f"  Tier bands (quantile of importance):")
    for tid, (lo, hi) in TIER_BANDS.items():
        print(f"    b{tid}: [{lo:.3f}, {hi:.3f}]  → top {(1-lo)*100:.1f}%–{(1-hi)*100:.1f}% by importance")
    print(f"  Init: {'k-means++' if USE_KMEANS_PP_INIT else 'random'}")
    print(f"  Refinement: {'on' if ENABLE_GRAD_REFINE else 'OFF (saves ~1h)'}")
    print(f"  Hierarchical factorization: {HIER_FACTORIZATION}")
    print(f"  Save: {SAVE_DIR}")
    print("=" * 72)

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("\nStep 1/4  Loading Wikipedia QA-style calibration prompts ...")
    texts = load_wikipedia_qa_style(NUM_SAMPLES)
    print(f"  Loaded {len(texts)} prompts")

    # ── Load model with EAGER attention (required for output_attentions) ──
    print("\nStep 2/4  Loading model (eager attention for importance scoring) ...")
    print(f"  transformers={transformers.__version__}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto",
        attn_implementation="eager",
    ).eval()

    cfg = model.config
    num_layers   = cfg.num_hidden_layers
    num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    num_q_heads  = cfg.num_attention_heads
    dh           = getattr(cfg, "head_dim", cfg.hidden_size // num_q_heads)
    print(f"  Layers={num_layers}  Q-heads={num_q_heads}  "
          f"KV-heads={num_kv_heads}  head_dim={dh}")

    unique_g = sorted(set(g for _, _, g, _, _ in TIERS))
    for tid, name, g, bt, K in TIERS:
        if dh % g == 0:
            G = dh // g
            mode = "hier" if K in HIER_FACTORIZATION else "flat"
            print(f"  Tier {tid} ({name}): g={g} G={G} K={K} b_theta={bt} {mode}")

    print(f"\nStep 3/4  Extracting per-tier importance-banded K vectors ...")
    accum_K = extract_importance_banded_keys(
        model, tokenizer, texts, num_layers, num_kv_heads, num_q_heads,
        dh, DEVICE)

    # Per-tier bucket sizes summary
    by_tier: Dict[int, List[int]] = {}
    for (li, h, tid), t in accum_K.items():
        by_tier.setdefault(tid, []).append(t.shape[0])
    print(f"\n  Filtered keys per (layer, kv_head, tier):")
    for tid in sorted(by_tier.keys()):
        sizes = sorted(by_tier[tid])
        print(f"    tier {tid}: n_buckets={len(sizes)}  "
              f"min={sizes[0]:,}  med={sizes[len(sizes)//2]:,}  max={sizes[-1]:,}")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    tier_specs = {tid: (tname, tg, bt, K) for tid, tname, tg, bt, K in TIERS}
    total_fits = sum(
        (dh // tier_specs[tid][1]) for (_, _, tid) in accum_K.keys()
        if tid in tier_specs and dh % tier_specs[tid][1] == 0
    )
    print(f"\nStep 4/4  Fitting {total_fits} codebooks ...")

    quality: Dict[int, List[float]] = {}
    refine_lift: List[float] = []
    pbar = tqdm(total=total_fits, desc="kmeans")

    for (li, h, tid), K_lh in accum_K.items():
        if tid not in tier_specs:
            continue
        tname, tg, bt, K = tier_specs[tid]
        if dh % tg != 0:
            continue
        G = dh // tg

        K_gpu = K_lh.to(DEVICE).float()             # fp16 → fp32 on GPU
        K_grp = K_gpu.view(-1, G, tg)
        norms = K_grp.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        U_grp = K_grp / norms

        for j in range(G):
            U_jg = F.normalize(U_grp[:, j, :], dim=1)
            if U_jg.shape[0] < K:
                pbar.update(1)
                continue

            centers, avg_cos = hierarchical_spherical_kmeans(U_jg, K, DEVICE)

            if ENABLE_GRAD_REFINE and U_jg.shape[0] >= 4 * K:
                centers_ref, pre, post = gradient_refine_centers(U_jg, centers)
                if post > pre:
                    centers = centers_ref
                    refine_lift.append(post - pre)
                    avg_cos = post

            quality.setdefault(tid, []).append(avg_cos)
            torch.save(
                centers.cpu(),
                f"{SAVE_DIR}/layer{li}_head{h}_group{j}_tier{tid}.pt")
            pbar.update(1)

        del K_gpu, K_grp, U_grp
        torch.cuda.empty_cache()
    pbar.close()

    # ── Quality report ──
    print("\nCodebook quality (importance-banded, hierarchical k-means):")
    for tid, tname, _, bt, K in TIERS:
        stats = sorted(quality.get(tid, []))
        if not stats:
            continue
        mean = sum(stats) / len(stats)
        p10  = stats[max(0, int(len(stats) * 0.10) - 1)]
        p90  = stats[min(len(stats) - 1, int(len(stats) * 0.90))]
        mode = "hier" if K in HIER_FACTORIZATION else "flat"
        lo, hi = TIER_BANDS.get(tid, (0.0, 1.0))
        print(f"  Tier {tid} ({tname}, K={K}, b_theta={bt}, {mode}, "
              f"band=[{lo:.2f}, {hi:.2f}]): "
              f"avg_cos={mean:.4f}  p10={p10:.4f}  p90={p90:.4f}")

    if refine_lift:
        avg_lift = sum(refine_lift) / len(refine_lift)
        print(f"\nGradient refinement: improved {len(refine_lift)} codebooks. "
              f"avg lift={avg_lift:+.4f}  max={max(refine_lift):+.4f}")

    print(f"\n{'=' * 72}")
    print(f"Done. Codebooks saved to '{SAVE_DIR}/'")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
