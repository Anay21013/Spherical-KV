import os
import gc
import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Optional, Tuple

from config import MODEL_NAME, SEQ_LEN, NUM_SAMPLES, SAVE_DIR, TIERS, DEVICE
from dataset_loader import load_c4

CHUNK_SIZE = 100


_UNIQUE_GROUP_SIZES = sorted(set(g for _, _, g, _, _ in TIERS))

def _extract_key_from_layer(layer_cache) -> Optional[torch.Tensor]:
    for attr in ("key", "keys", "key_cache", "k_cache", "_key", "k"):
        val = getattr(layer_cache, attr, None)
        if val is not None and isinstance(val, torch.Tensor) and val.ndim == 4:
            return val
    try:
        val = layer_cache[0]
        if isinstance(val, torch.Tensor) and val.ndim == 4:
            return val
    except (TypeError, IndexError, KeyError):
        pass
    return None


def _iter_key_cache(past_key_values):
    if hasattr(past_key_values, "layers"):
        for layer_id, layer in enumerate(past_key_values.layers):
            K = _extract_key_from_layer(layer)
            if K is not None and K.shape[2] > 0:
                yield layer_id, K
        return
    if hasattr(past_key_values, "key_cache"):
        for layer_id, K in enumerate(past_key_values.key_cache):
            if K is not None and K.ndim == 4 and K.shape[2] > 0:
                yield layer_id, K
        return
    if hasattr(past_key_values, "to_legacy_cache"):
        try:
            for layer_id, kv in enumerate(past_key_values.to_legacy_cache()):
                if isinstance(kv, (list, tuple)):
                    K = kv[0]
                    if K is not None and K.ndim == 4 and K.shape[2] > 0:
                        yield layer_id, K
            return
        except Exception:
            pass
    if isinstance(past_key_values, (list, tuple)):
        for layer_id, kv in enumerate(past_key_values):
            if isinstance(kv, (list, tuple)) and len(kv) >= 1:
                K = kv[0]
                if K is not None and K.ndim == 4 and K.shape[2] > 0:
                    yield layer_id, K


def main():
    print("=" * 60)
    print("Spherical KV Codebook Training started")
    print("Per-(layer, head, group_idx, tier)  |  incremental k-means")
    print("=" * 60)
    print(f"Model:    {MODEL_NAME}")
    print(f"Samples:  {NUM_SAMPLES}  |  Chunk: {CHUNK_SIZE}  |  Dir: {SAVE_DIR}")
    print()

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Step 1/3  Loading dataset ...")
    texts = load_c4(NUM_SAMPLES)
    print(f"  Loaded {len(texts)} texts.\n")

    print("Step 2/3  Loading model ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="auto"
    )
    model.eval()

    with torch.no_grad():
        sample = tokenizer(texts[0], return_tensors="pt",
                           truncation=True, max_length=32).to(DEVICE)
        out = model(**sample, use_cache=True)
        kv_list = list(_iter_key_cache(out.past_key_values))
        num_layers   = len(kv_list)
        num_kv_heads = kv_list[0][1].shape[1]
        dh           = kv_list[0][1].shape[3]
        del out, kv_list
        torch.cuda.empty_cache()

    print(f"  Layers={num_layers}  KV-heads={num_kv_heads}  head_dim={dh}\n")

    for tid, name, g, bt, K in TIERS:
        if dh % g == 0:
            G = dh // g
            print(f"  Tier {tid} ({name}): g={g}, G={G} groups/head, K={K}")
    print()

    #Initialise one MiniBatchKMeans per (layer, head, group_idx, tier)
    #Key: (layer, head, grp_idx, tier_id)
    #Only create for (g, grp_idx) combinations that are valid for dh.
    print("Step 3/3  Incremental extraction + training ...")
    print(f"  Processing {len(texts)} texts in chunks of {CHUNK_SIZE}\n")

    kmeans_map: Dict[Tuple[int, int, int, int], MiniBatchKMeans] = {}
    for l in range(num_layers):
        for h in range(num_kv_heads):
            for tid, name, g, bt, K in TIERS:
                if dh % g != 0:
                    continue
                G = dh // g
                for j in range(G):
                    kmeans_map[(l, h, j, tid)] = MiniBatchKMeans(
                        n_clusters=K,
                        batch_size=4096,
                        max_iter=100,
                        random_state=42,
                        n_init=3,
                    )

    total_kmeans = len(kmeans_map)
    print(f"  Initialised {total_kmeans} k-means models")
    print(f"  ({num_layers}L x {num_kv_heads}H x "
          f"{total_kmeans // (num_layers * num_kv_heads)} per head)\n")

    chunks = [texts[i:i + CHUNK_SIZE]
              for i in range(0, len(texts), CHUNK_SIZE)]

    for chunk_idx, chunk in enumerate(tqdm(chunks, desc="chunks")):

        accum: Dict[Tuple[int, int, int, int], list] = {}

        with torch.no_grad():
            for text in chunk:
                tokens = tokenizer(
                    text, return_tensors="pt",
                    truncation=True, max_length=SEQ_LEN,
                ).to(DEVICE)
                outputs = model(**tokens, use_cache=True)

                for layer_id, K_layer in _iter_key_cache(
                        outputs.past_key_values):
                    # K_layer: [1, num_kv_heads, T, dh]
                    K_all = K_layer[0].float().cpu()
                    T     = K_all.shape[1]

                    for h in range(num_kv_heads):
                        K_h = K_all[h]

                        for g in _UNIQUE_GROUP_SIZES:
                            if dh % g != 0:
                                continue
                            G     = dh // g
                            K_grp = K_h.view(T, G, g)  # [T, G, g]
                            norms = K_grp.norm(
                                dim=-1, keepdim=True).clamp(min=1e-6)
                            U_grp = K_grp / norms       # unit-norm [T, G, g]

                            for j in range(G):
                                key = (layer_id, h, j, g)
                                accum.setdefault(key, []).append(
                                    U_grp[:, j, :])    # [T, g]

                del outputs
                torch.cuda.empty_cache()

        for (l, h, j, g), parts in accum.items():
            U = torch.cat(parts, dim=0).numpy().astype(np.float32)
            norms = np.linalg.norm(U, axis=1, keepdims=True)
            U     = U / np.maximum(norms, 1e-8)

            for tid, tname, tg, bt, K in TIERS:
                if tg != g:
                    continue
                kmeans_map[(l, h, j, tid)].partial_fit(U)

        del accum
        gc.collect()

    print("\nSaving codebooks ...")
    total_files = 0
    for (l, h, j, tid), km in tqdm(kmeans_map.items(), desc="saving"):
        centers = torch.tensor(km.cluster_centers_, dtype=torch.float32)
        centers = centers / centers.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        path = f"{SAVE_DIR}/layer{l}_head{h}_group{j}_tier{tid}.pt"
        torch.save(centers, path)
        total_files += 1

    print(f"\n{'='*60}")
    print(f"Done.  {total_files} codebook files  →  {SAVE_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()