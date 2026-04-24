"""
Updated training script that captures K *before* RoPE is applied
"""

import os
import gc
import torch
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Optional, Tuple, List

from config import MODEL_NAME, SEQ_LEN, NUM_SAMPLES, SAVE_DIR, TIERS, DEVICE
from dataset_loader import load_c4


# Training hyperparameters

KMEANS_MODE         = "minibatch"   # "full" or "minibatch" (needs to be toggled while running experiments)
CHUNK_SIZE          = 50
MAX_VECS_PER_GROUP  = 100_000

FULL_KMEANS_N_INIT   = 10
FULL_KMEANS_MAX_ITER = 500
FULL_KMEANS_TOL      = 1e-5

MINIBATCH_BATCH_SIZE = 16384
MINIBATCH_MAX_ITER   = 1000
MINIBATCH_N_INIT     = 10


def _fit_kmeans(U, n_clusters, seed=42):
    km = KMeans(n_clusters=n_clusters, init="k-means++",
                n_init=FULL_KMEANS_N_INIT, max_iter=FULL_KMEANS_MAX_ITER,
                tol=FULL_KMEANS_TOL, random_state=seed, algorithm="lloyd")
    km.fit(U)
    return km.cluster_centers_


def _fit_minibatch(U, n_clusters, seed=42):
    km = MiniBatchKMeans(n_clusters=n_clusters, batch_size=MINIBATCH_BATCH_SIZE,
                         max_iter=MINIBATCH_MAX_ITER, n_init=MINIBATCH_N_INIT,
                         init="k-means++", random_state=seed,
                         reassignment_ratio=0.01, tol=1e-4)
    km.fit(U)
    return km.cluster_centers_


def main():
    print("=" * 70)
    print("Spherical KV Codebook Training -- PRE-RoPE K capture")
    print(f"Mode: {KMEANS_MODE}  |  Max vecs/group: {MAX_VECS_PER_GROUP:,}")
    print("=" * 70)
    print(f"Model:    {MODEL_NAME}")
    print(f"Samples:  {NUM_SAMPLES}  |  Chunk: {CHUNK_SIZE}  |  Save: {SAVE_DIR}\n")

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Step 1/4  Loading C4 dataset ...")
    texts = load_c4(NUM_SAMPLES)
    print(f"  Loaded {len(texts)} texts\n")

    print("Step 2/4  Loading model ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="auto"
    ).eval()

    cfg = model.config
    num_layers   = cfg.num_hidden_layers
    num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    dh           = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

    print(f"  Layers={num_layers}  KV-heads={num_kv_heads}  head_dim={dh}\n")

    _UNIQUE_GROUP_SIZES = sorted(set(g for _, _, g, _, _ in TIERS))
    for tid, name, g, bt, K in TIERS:
        if dh % g == 0:
            G = dh // g
            print(f"  Tier {tid} ({name}): g={g}, G={G} groups/head, K={K}, b_theta={bt}")
    print()

    print("Step 3/4  Installing pre-RoPE hooks on k_proj layers ...")

    pre_rope_K: Dict[int, torch.Tensor] = {}

    def _make_hook(layer_idx):
        def _hook(module, input_tuple, output):
            pre_rope_K[layer_idx] = output.detach().float().cpu()
        return _hook

    hook_handles = []
    for li in range(num_layers):
        handle = model.model.layers[li].self_attn.k_proj.register_forward_hook(
            _make_hook(li))
        hook_handles.append(handle)

    print(f"  Installed {len(hook_handles)} k_proj hooks\n")

    print("Step 4/4  Extracting pre-RoPE K vectors ...")

    accum: Dict[Tuple[int, int, int, int], List] = {}
    counts: Dict[Tuple[int, int, int, int], int] = {}

    chunks = [texts[i:i + CHUNK_SIZE] for i in range(0, len(texts), CHUNK_SIZE)]

    for chunk in tqdm(chunks, desc="extract"):
        with torch.no_grad():
            for text in chunk:
                pre_rope_K.clear()
                tokens = tokenizer(text, return_tensors="pt",
                                   truncation=True, max_length=SEQ_LEN).to(DEVICE)
                _ = model(**tokens, use_cache=False)

                for li in range(num_layers):
                    if li not in pre_rope_K:
                        continue
                    K_lh = pre_rope_K[li][0]  # [T, num_kv_heads * dh]
                    T = K_lh.shape[0]
                    K_lh = K_lh.view(T, num_kv_heads, dh)

                    for h in range(num_kv_heads):
                        K_h = K_lh[:, h, :]

                        for g in _UNIQUE_GROUP_SIZES:
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
                                vecs = U_grp[:take, j, :].clone()
                                accum.setdefault(key, []).append(vecs)
                                counts[key] = cur + take
        torch.cuda.empty_cache()
        gc.collect()

    for h in hook_handles:
        h.remove()

    sizes = sorted(counts.values())
    if sizes:
        print(f"\n  Samples per bucket: min={sizes[0]} median={sizes[len(sizes)//2]} max={sizes[-1]}")
        print(f"  Buckets: {len(accum)}\n")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    total_fits = sum(1 for (_, _, _, g) in accum.keys()
                     for tid, _, tg, _, _ in TIERS if tg == g)
    print(f"Fitting {total_fits} codebooks in '{KMEANS_MODE}' mode ...\n")

    fitter = _fit_kmeans if KMEANS_MODE == "full" else _fit_minibatch
    ep_stats = []

    pbar = tqdm(total=total_fits, desc="kmeans")
    for (li, h, j, g), parts in accum.items():
        U = torch.cat(parts, dim=0).numpy().astype(np.float32)
        norms = np.linalg.norm(U, axis=1, keepdims=True)
        U = U / np.maximum(norms, 1e-8)

        for tid, tname, tg, bt, K in TIERS:
            if tg != g:
                continue
            if U.shape[0] < K:
                print(f"  WARN: bucket (l={li},h={h},j={j},g={g}) has {U.shape[0]} "
                      f"< {K} clusters; skipping")
                pbar.update(1)
                continue

            centers = fitter(U, n_clusters=K, seed=42)
            dists = np.min(np.linalg.norm(
                U[:, None, :] - centers[None, :, :], axis=-1), axis=1)
            ep_stats.append((tid, float(dists.mean())))

            centers_t = torch.tensor(centers, dtype=torch.float32)
            centers_t = centers_t / centers_t.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            torch.save(centers_t,
                       f"{SAVE_DIR}/layer{li}_head{h}_group{j}_tier{tid}.pt")
            pbar.update(1)
    pbar.close()

    print("\nCodebook quality (pre-RoPE training):")
    for tid, _, _, bt, K in TIERS:
        t_stats = sorted([x[1] for x in ep_stats if x[0] == tid])
        if t_stats:
            print(f"  Tier {tid} (b_theta={bt}, K={K}): "
                  f"mean={sum(t_stats)/len(t_stats):.4f}  "
                  f"p90={t_stats[int(len(t_stats)*0.9)]:.4f}")

    print(f"\n{'='*70}")
    print(f"Done.  Codebooks saved to '{SAVE_DIR}/'")
    print(f"{'='*70}\n")
    print("Expected tier 1 ε_θ after this (pre-RoPE): <0.01")


if __name__ == "__main__":
    main()
