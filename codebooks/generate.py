import os
import gc
import torch
import torch.nn.functional as F
import numpy as np
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Tuple, List

from config import MODEL_NAME, SEQ_LEN, NUM_SAMPLES, SAVE_DIR, TIERS, DEVICE
from dataset_loader import load_c4


# Tunables
CHUNK_SIZE         = 10
MAX_VECS_PER_GROUP = 200_000

NITER_FLAT      = 500
NITER_HIER      = 200      # per-level iterations
NREDO_FLAT      = 20
NREDO_HIER      = 8        # per-level restarts
SEED            = 42

HIER_FACTORIZATION = {
    64: (8, 8),
    16: (4, 4), 
}


def spherical_kmeans_gpu(U, K, niter, nredo, device="cuda", seed=SEED):
    """Standard spherical k-means on unit sphere S^{d-1}."""
    N, d = U.shape
    if N < K:
        repeats = (K + N - 1) // N
        U = U.repeat(repeats, 1)[:K]
        N = K

    best_centers, best_score = None, -1e9
    g = torch.Generator(device=device).manual_seed(seed)

    for _ in range(nredo):
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

    N, d = U.shape

    # Stage 1: coarse clustering
    coarse_centers, _ = spherical_kmeans_gpu(
        U, K_coarse, NITER_HIER, NREDO_HIER, device, seed=SEED)
    sims = U @ coarse_centers.T
    coarse_assigns = sims.argmax(dim=1)

    # Stage 2: fine clustering within each coarse cluster
    all_fine_centers = []
    for c in range(K_coarse):
        mask = (coarse_assigns == c)
        n_c = int(mask.sum().item())

        if n_c < K_fine:
            base = coarse_centers[c]
            noise = torch.randn(K_fine, d, device=device) * 0.05
            fine_centers = F.normalize(base.unsqueeze(0) + noise, dim=1)
        else:
            U_c = U[mask]
            fine_centers, _ = spherical_kmeans_gpu(
                U_c, K_fine, NITER_HIER, NREDO_HIER, device, seed=SEED + c + 1)

        all_fine_centers.append(fine_centers)

    centers = torch.cat(all_fine_centers, dim=0)
    centers = F.normalize(centers, dim=-1)

    sims_final = U @ centers.T
    avg_cos = sims_final.max(dim=1).values.mean().item()

    return centers, avg_cos


def main():
    print("=" * 70)
    print("Hierarchical Spherical KV Codebook Training (POST-RoPE)")
    print(f"  Algorithm: hierarchical spherical k-means")
    print(f"  Factorization: {HIER_FACTORIZATION}")
    print(f"  Restarts:  flat={NREDO_FLAT}  hier={NREDO_HIER}    Seed: {SEED}")
    print(f"  Iterations: flat={NITER_FLAT}  hier={NITER_HIER}")
    print(f"  Max vecs/bucket: {MAX_VECS_PER_GROUP:,}")
    print("=" * 70)
    print(f"Model:    {MODEL_NAME}")
    print(f"Samples:  {NUM_SAMPLES}  |  Chunk: {CHUNK_SIZE}  |  Save: {SAVE_DIR}\n")

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Step 1/4  Loading C4 dataset ...")
    texts = load_c4(NUM_SAMPLES)
    print(f"  Loaded {len(texts)} texts\n")

    print("Step 2/4  Loading model ...")
    print(f"  transformers={transformers.__version__}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    ).eval()

    cfg = model.config
    num_layers   = cfg.num_hidden_layers
    num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    dh           = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    print(f"  Layers={num_layers}  KV-heads={num_kv_heads}  head_dim={dh}\n")

    unique_g = sorted(set(g for _, _, g, _, _ in TIERS))
    for tid, name, g, bt, K in TIERS:
        if dh % g == 0:
            G = dh // g
            mode = ("hier" if K in HIER_FACTORIZATION else "flat")
            extra = (f" ({HIER_FACTORIZATION[K][0]}x{HIER_FACTORIZATION[K][1]})"
                     if K in HIER_FACTORIZATION else "")
            print(f"  Tier {tid} ({name}): g={g}  G={G}  K={K}  "
                  f"b_theta={bt}  mode={mode}{extra}")
    print()

    print("Step 3/4  Extracting POST-RoPE K from KV cache ...")
    accum:  Dict[Tuple[int, int, int, int], List] = {}
    counts: Dict[Tuple[int, int, int, int], int] = {}
    chunks = [texts[i:i + CHUNK_SIZE] for i in range(0, len(texts), CHUNK_SIZE)]

    for chunk in tqdm(chunks, desc="extract"):
        with torch.no_grad():
            for text in chunk:
                tokens = tokenizer(text, return_tensors="pt",
                                   truncation=True, max_length=SEQ_LEN).to(DEVICE)
                out = model(**tokens, use_cache=True, return_dict=True)
                pkv = out.past_key_values

                if hasattr(pkv, "layers"):
                    layer_iter = []
                    for layer in pkv.layers:
                        K_layer = getattr(layer, "keys", None)
                        if K_layer is None:
                            K_layer = getattr(layer, "key", None)
                        if K_layer is not None:
                            layer_iter.append(K_layer)
                else:
                    layer_iter = [pkv[li][0] for li in range(num_layers)]

                for li, K_post in enumerate(layer_iter):
                    K_lh = K_post[0].float().cpu()
                    if K_lh.dim() == 3:
                        T = K_lh.shape[1]
                        K_lh = K_lh.permute(1, 0, 2)
                    else:
                        T = K_lh.shape[0]
                        K_lh = K_lh.view(T, num_kv_heads, dh)

                    for h in range(num_kv_heads):
                        K_h = K_lh[:, h, :]
                        for g in unique_g:
                            if dh % g != 0:
                                continue
                            G = dh // g
                            K_grp = K_h.view(T, G, g)
                            norms = K_grp.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                            U_grp = K_grp / norms
                            for j in range(G):
                                key = (li, h, j, g)
                                cur = counts.get(key, 0)
                                if cur >= MAX_VECS_PER_GROUP:
                                    continue
                                room = MAX_VECS_PER_GROUP - cur
                                take = min(T, room)
                                accum.setdefault(key, []).append(
                                    U_grp[:take, j, :].clone())
                                counts[key] = cur + take
                del out, pkv
        torch.cuda.empty_cache(); gc.collect()

    sizes = sorted(counts.values())
    if sizes:
        print(f"\n  Buckets: {len(accum)}  vecs/bucket: "
              f"min={sizes[0]} med={sizes[len(sizes)//2]} max={sizes[-1]}\n")

    del model
    torch.cuda.empty_cache(); gc.collect()

    total_fits = sum(1 for (_, _, _, g) in accum.keys()
                     for tid, _, tg, _, _ in TIERS if tg == g)
    print(f"Step 4/4  Fitting {total_fits} codebooks (hierarchical k-means) ...\n")

    quality: Dict[int, List[float]] = {}
    pbar = tqdm(total=total_fits, desc="hier-kmeans")

    for (li, h, j, g), parts in accum.items():
        U_cpu = torch.cat(parts, dim=0).float()
        U_gpu = U_cpu.to(DEVICE)
        U_gpu = F.normalize(U_gpu, dim=1)

        for tid, tname, tg, bt, K in TIERS:
            if tg != g:
                continue
            if U_gpu.shape[0] < K:
                pbar.update(1)
                continue

            centers, avg_cos = hierarchical_spherical_kmeans(U_gpu, K, DEVICE)
            quality.setdefault(tid, []).append(avg_cos)

            torch.save(
                centers.cpu(),
                f"{SAVE_DIR}/layer{li}_head{h}_group{j}_tier{tid}.pt")
            pbar.update(1)

        del U_gpu
        torch.cuda.empty_cache()
    pbar.close()

    print("\nCodebook quality (POST-RoPE, hierarchical spherical k-means):")
    for tid, tname, _, bt, K in TIERS:
        stats = sorted(quality.get(tid, []))
        if not stats:
            continue
        mean = sum(stats) / len(stats)
        p10  = stats[max(0, int(len(stats) * 0.10) - 1)]
        p90  = stats[min(len(stats) - 1, int(len(stats) * 0.90))]
        mode = ("hier" if K in HIER_FACTORIZATION else "flat")
        print(f"  Tier {tid} ({tname}, K={K}, b_theta={bt}, {mode}): "
              f"avg_cos={mean:.4f}  p10={p10:.4f}  p90={p90:.4f}")

    print(f"\n{'=' * 70}")
    print(f"Done.  Codebooks saved to '{SAVE_DIR}/'")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
