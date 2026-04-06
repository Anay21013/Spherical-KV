import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch

def load_codebooks(
    codebook_dir: str,
    num_layers:   int,
    num_kv_heads: int,
    tiers,                     # list from build_tiers()
) -> Dict[Tuple[int, int, int], torch.Tensor]:
    """
    Load all per-(layer, head, tier) codebooks.
    """
    if not os.path.isdir(codebook_dir):
        raise FileNotFoundError(f"Codebook directory not found: {codebook_dir}")

    active_tiers = [t for t in tiers if t.tier_id != 0]   # skip drop
    codebooks: Dict[Tuple[int, int, int], torch.Tensor] = {}
    missing: list = []

    for layer in range(num_layers):
        for head in range(num_kv_heads):
            for tier in active_tiers:
                G   = tier.G
                g   = tier.g
                tid = tier.tier_id
                group_cbs = []

                for grp in range(G):
                    cb = (_try_load(codebook_dir,
                                    f"layer{layer}_head{head}_group{grp}_tier{tid}"))

                    if cb is None:
                        missing.append(
                            f"layer{layer}_head{head}_group{grp}_tier{tid}")
                        break

                    cb = cb.float()
                    if cb.ndim == 1:
                        cb = cb.view(-1, g)
                    elif cb.shape[-1] != g:
                        cb = cb.T.contiguous()

                    norms = cb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    cb    = cb / norms
                    group_cbs.append(cb)

                if len(group_cbs) == G:
                    codebooks[(layer, head, tid)] = torch.stack(group_cbs, dim=0)

    if missing:
        print(f"[codebook_loader] WARNING: {len(missing)} file(s) missing. "
              f"First: '{missing[0]}'")

    n_loaded   = len(codebooks)
    n_expected = num_layers * num_kv_heads * len(active_tiers)
    print(f"[codebook_loader] Loaded {n_loaded}/{n_expected} codebook entries "
          f"({num_layers}L x {num_kv_heads}H x {len(active_tiers)} tiers)")

    return codebooks


def get_codebook(
    codebooks: Dict[Tuple[int, int, int], torch.Tensor],
    layer:    int,
    kv_head:  int,
    tier_id:  int,
) -> Optional[torch.Tensor]:
    """Convenience accessor; returns None if absent."""
    return codebooks.get((layer, kv_head, tier_id))


def _try_load(directory: str, stem: str) -> Optional[torch.Tensor]:
    for ext in (".pt", ".pth", ".npy", ""):
        path = os.path.join(directory, stem + ext)
        if not os.path.exists(path):
            continue
        try:
            if ext == ".npy":
                return torch.from_numpy(np.load(path)).float()
            obj = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(obj, torch.Tensor):
                return obj.float()
            if isinstance(obj, dict):
                vals = list(obj.values())
                if vals and isinstance(vals[0], torch.Tensor):
                    return vals[0].float()
        except Exception as exc:
            print(f"[codebook_loader] Could not read '{path}': {exc}")
    return None