from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
import time
import statistics
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from codebook_loader import load_codebooks
from config import GROUP_SIZE, HEADER_BYTES, NUM_GROUPS, PAGE_SIZE
from spherical_kv_pipeline import SphericalKVPipeline, reference_codebook_decode

N_WARM = 2
N_MEAS = 8
N_TRIALS = 1


@contextlib.contextmanager
def nvtx_range(name: str):
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()



def load_model_and_tokenizer(name_or_path: str, device: torch.device):
    try:
        from transformers import AutoTokenizer, LlamaForCausalLM
    except ImportError:
        raise ImportError("pip install transformers>=4.40")

    print(f"[evaluate] Loading '{name_or_path}' ...")
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    model = LlamaForCausalLM.from_pretrained(
        name_or_path,
        dtype=torch.float32,
        device_map={"": device},
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


def get_eval_tokens(
    tokenizer,
    text:         Optional[str],
    dataset_name: Optional[str],
    num_tokens:   int,
) -> torch.Tensor:
    if text:
        ids = tokenizer.encode(text, return_tensors="pt").squeeze(0)
    elif dataset_name:
        ids = _load_dataset_tokens(tokenizer, dataset_name, num_tokens)
    else:
        raise ValueError("Provide --eval_text or --dataset")
    ids = ids[:num_tokens]
    print(f"[evaluate] Eval corpus: {ids.numel()} tokens  (dataset={dataset_name})")
    return ids


def _load_dataset_tokens(tokenizer, dataset_name: str, num_tokens: int) -> torch.Tensor:
    from datasets import load_dataset
    if dataset_name == "pg19":
        ds   = load_dataset("pg19", split="test", trust_remote_code=True, streaming=True)
        # text = " ".join(ds["text"][:5])
        texts = []
        for i, example in enumerate(ds):
            texts.append(example["text"])
            if i >= 4:   # first 5 books
                break
        text = " ".join(texts)
    elif dataset_name == "wikitext":
        ds   = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = " ".join(ds["text"])
    else:
        ds   = load_dataset(dataset_name, split="test")
        text = " ".join(ds["text"])
    return tokenizer.encode(text, return_tensors="pt").squeeze(0)


def eval_memory_footprint(
    pipeline: SphericalKVPipeline,
    prefill_ids: torch.Tensor,
) -> dict:
    if not pipeline.per_head_pages:
        pipeline.prefill(prefill_ids.to(pipeline.device))

    T  = pipeline.seq_len
    L  = pipeline.num_layers
    H  = pipeline.num_kv_heads
    dh = pipeline.head_dim
    G  = pipeline.num_groups

    # Dense FP16 baseline
    dense_K_bytes = 2 * T * L * H * dh
    dense_V_bytes = 2 * T * L * H * dh
    compressed_K_pages_bytes  = 0
    pointer_table_bytes       = 0

    for (layer, head), tier_list in pipeline.per_head_pages.items():
        for entry in tier_list:
            pages_tensor  = entry[0]
            ptable_tensor = entry[1]
            compressed_K_pages_bytes += pages_tensor.numel()   # header+theta+radius
            pointer_table_bytes      += ptable_tensor.numel() * 4  # [P,3] int32

    total_compressed_K_bytes = compressed_K_pages_bytes + pointer_table_bytes

    # V stays dense FP16; sum V_tier sizes across all tiers/heads.
    retained_V_bytes = sum(
        entry[4].numel() * 2
        for tier_list in pipeline.per_head_pages.values()
        for entry in tier_list
        if entry[4] is not None
    )

    kv_bytes_per_token_compressed = (
        (total_compressed_K_bytes + retained_V_bytes) / max(T, 1)
    )
    kv_bytes_per_token_dense = (dense_K_bytes + dense_V_bytes) / max(T, 1)

    return {
        "tokens":                      T,
        "dense_K_bytes":               dense_K_bytes,
        "dense_V_bytes":               dense_V_bytes,
        "compressed_K_pages_bytes":    compressed_K_pages_bytes,  # header+theta+radius
        "compressed_K_ptable_bytes":   pointer_table_bytes,
        "compressed_K_total_bytes":    total_compressed_K_bytes,
        "retained_V_bytes":            retained_V_bytes,
        "kv_bytes_per_token_dense":    kv_bytes_per_token_dense,
        "kv_bytes_per_token_sphkv":    kv_bytes_per_token_compressed,
        "compression_ratio_K":         dense_K_bytes / max(total_compressed_K_bytes, 1),
        "peak_KV_MB_dense":            (dense_K_bytes + dense_V_bytes) / 1e6,
        "peak_KV_MB_sphkv":            (total_compressed_K_bytes + retained_V_bytes) / 1e6,
    }



def eval_throughput(
    pipeline:    SphericalKVPipeline,
    prefill_ids: torch.Tensor,
    n_warm:  int = N_WARM,
    n_meas:  int = N_MEAS,
    n_trials: int = N_TRIALS,
) -> dict:
    device = pipeline.device
    is_cuda = device.type == "cuda"

    pipeline.prefill(prefill_ids.to(device))

    toks_per_sec_list: List[float] = []
    hbm_bytes_list:    List[float] = []

    for trial in range(n_trials):
        current_ids = prefill_ids.to(device).clone()

        for _ in range(n_warm):
            with torch.no_grad():
                out = pipeline.model(
                    input_ids=current_ids[:, -1:],
                    use_cache=False, return_dict=True,
                )
            next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
            current_ids = torch.cat([current_ids, next_id], dim=-1)
        if is_cuda:
            torch.cuda.synchronize()

        if is_cuda:
            mem_before = torch.cuda.memory_stats(device).get(
                "allocated_bytes.all.current", 0
            )
            t_start = torch.cuda.Event(enable_timing=True)
            t_end   = torch.cuda.Event(enable_timing=True)
            t_start.record()
        else:
            t_wall_start = time.perf_counter()

        for _ in range(n_meas):
            with torch.no_grad():
                with nvtx_range("page_lookup"):
                    pass
                out = pipeline.model(
                    input_ids=current_ids[:, -1:],
                    use_cache=False, return_dict=True,
                )
            next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
            current_ids = torch.cat([current_ids, next_id], dim=-1)

        if is_cuda:
            t_end.record()
            torch.cuda.synchronize()
            elapsed_ms  = t_start.elapsed_time(t_end)
            elapsed_s   = elapsed_ms / 1e3
            mem_after   = torch.cuda.memory_stats(device).get(
                "allocated_bytes.all.current", 0
            )
            hbm_delta   = max(mem_after - mem_before, 0)
            hbm_per_tok = hbm_delta / n_meas
        else:
            elapsed_s   = time.perf_counter() - t_wall_start
            hbm_per_tok = 0.0

        toks_per_sec_list.append(n_meas / elapsed_s)
        hbm_bytes_list.append(hbm_per_tok)

    tps_median  = statistics.median(toks_per_sec_list)
    hbm_median  = statistics.median(hbm_bytes_list)
    tps_p50     = tps_median
    tps_p95     = sorted(toks_per_sec_list)[int(0.95 * n_trials)]

    print(
        f"[eval_throughput] tok/s median={tps_median:.1f}  p95={tps_p95:.1f}  "
        f"bHBM/tok={hbm_median:.0f} bytes  (proxy; use ncu for hardware counters)"
    )

    return {
        "tok_s_median":      tps_median,
        "tok_s_p50":         tps_p50,
        "tok_s_p95":         tps_p95,
        "tok_s_all_trials":  toks_per_sec_list,
        "hbm_bytes_per_tok_proxy":   hbm_median,
        "n_warm":            n_warm,
        "n_meas":            n_meas,
        "n_trials":          n_trials,
    }


@torch.no_grad()
def eval_attention_quality(
    pipeline:         SphericalKVPipeline,
    prefill_ids:      torch.Tensor,
    num_decode_steps: int = 16,
) -> dict:
    device = pipeline.device
    prefill_ids = prefill_ids.to(device)
    from llama_hooks import capture_prefill_pass
    with torch.no_grad():
        kv_pairs_ref, _, _, _ = capture_prefill_pass(pipeline.model, prefill_ids)

    K_ref: Dict[Tuple[int, int], torch.Tensor] = {}
    V_ref: Dict[Tuple[int, int], torch.Tensor] = {}
    for li, (K, V) in enumerate(kv_pairs_ref):
        for h in range(pipeline.num_kv_heads):
            K_ref[(li, h)] = K[0, h].float().cpu()   # [T, dh]
            V_ref[(li, h)] = V[0, h].float().cpu()   # [T, dh]
    del kv_pairs_ref

    print(f"\n[eval_attention_quality] Prefilling {prefill_ids.shape[1]} tokens ...")
    pipeline.prefill(prefill_ids)

    l2_errs:      List[float] = []
    logit_drifts: List[float] = []
    dh = pipeline.head_dim

    current_ids = prefill_ids.clone()
    print(f"[eval_attention_quality] Evaluating {num_decode_steps} decode steps ...")

    for step in range(num_decode_steps):
        with torch.no_grad():
            out = pipeline.model(
                input_ids=current_ids,
                use_cache=True, output_attentions=False, return_dict=True,
            )
        next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
        current_ids = torch.cat([current_ids, next_id], dim=-1)

        kv = out.past_key_values

        for li in range(pipeline.num_layers):
            # K_layer, V_layer = kv[li]
            layer_cache = kv.layers[li]
            K_layer = layer_cache.keys
            V_layer = layer_cache.values
            for h in range(pipeline.num_kv_heads):
                k_new = K_layer[0, h, -1, :].to(device).float()
                v_new = V_layer[0, h, -1, :].to(device).float()
                q_vec = k_new   # proxy query (key used as stand-in)

                # ── Compressed path: use per-head-pages 6-tuple ──────────
                with nvtx_range("page_lookup"):
                    ph    = pipeline.per_head_pages.get((li, h), [])
                    # cb_lh resolved per-tier inside the loop below

                ctx_logit_parts: List[torch.Tensor] = []
                V_approx_parts:  List[torch.Tensor] = []

                with nvtx_range("angle_logits"):
                    # b_theta→tier_id map from pipeline's tier list
                    _bt_to_tid = {t.b_theta: t.tier_id
                                  for t in pipeline.tiers if t.tier_id != 0}
                    for entry in ph:
                        pt       = entry[0]
                        ptt      = entry[1]
                        b_theta  = entry[2]
                        n_tokens = entry[3]
                        V_tier   = entry[4]
                        K_tier   = entry[5]
                        tier_id   = _bt_to_tid.get(b_theta, 1)
                        cb_lh     = pipeline.codebooks.get((li, h, tier_id))
                        tier_obj  = pipeline.tiers[tier_id]
                        tier_G    = tier_obj.G
                        tier_g    = tier_obj.g
                        if pipeline.use_fused:
                            from spherical_kv_pipeline import _call_fused
                            raw = _call_fused(
                                pt, ptt, q_vec, cb_lh, b_theta,
                                dh, tier_G, tier_g,
                            )
                            ctx_logit_parts.append(raw.view(-1)[:n_tokens])
                        else:
                            _r_c, _th_c = K_tier
                            _r_c  = _r_c.to(device)            # [N, tier_G]
                            _th_c = _th_c.to(device)           # [N, tier_G]
                            _qg   = q_vec.view(tier_G, tier_g) # [G, g]
                            # gather codewords: [tier_G, N, tier_g]
                            _cw = cb_lh[
                                torch.arange(tier_G, device=device).unsqueeze(1),
                                _th_c.long().T,
                            ]
                            # dots [G, N] → logits [N]
                            _dots = (_cw * _qg.unsqueeze(1)).sum(-1)  # [G, N]
                            ctx_logit_parts.append(
                                (_r_c.T * _dots).sum(0) / math.sqrt(dh)
                            )
                        V_approx_parts.append(V_tier)

                ctx_logits_approx = (
                    torch.cat(ctx_logit_parts) if ctx_logit_parts
                    else torch.zeros(0, device=device)
                )
                new_logit_approx  = (k_new @ q_vec / math.sqrt(dh)).unsqueeze(0)
                all_logits_approx = torch.cat([ctx_logits_approx, new_logit_approx])

                with nvtx_range("softmax"):
                    attn_approx = torch.softmax(all_logits_approx, dim=0)

                V_approx_parts.append(v_new.unsqueeze(0))
                with nvtx_range("kv_read"):
                    attn_out_approx = attn_approx @ torch.cat(V_approx_parts, dim=0)

                # ── Exact reference: all prefill K dense ─────────────────
                K_full = torch.cat([K_ref[(li, h)].to(device), k_new.unsqueeze(0)], dim=0)
                V_full = torch.cat([V_ref[(li, h)].to(device), v_new.unsqueeze(0)], dim=0)
                exact_logits = (K_full @ q_vec) / math.sqrt(dh)
                exact_attn   = torch.softmax(exact_logits, dim=0)
                exact_out    = exact_attn @ V_full

                l2_errs.append(
                    (attn_out_approx - exact_out).pow(2).mean().sqrt().item()
                )

                n_ctx = len(ctx_logits_approx)
                if n_ctx > 0:
                    # Retained token indices in positional order.
                    _order = pipeline._ctx_token_order.get((li, h), [])
                    _ret_idx = [tok.index for tok in _order[:n_ctx]]
                    if _ret_idx and (li, h) in K_ref:
                        K_retained_exact = K_ref[(li, h)][_ret_idx].to(device).float()
                        exact_retained = (K_retained_exact @ q_vec) / math.sqrt(dh)
                        drift = (ctx_logits_approx - exact_retained).abs().mean().item()
                        logit_drifts.append(drift)

        del out
        if (step + 1) % 4 == 0:
            print(
                f"  step {step+1}/{num_decode_steps}  "
                f"mean_L2={sum(l2_errs)/len(l2_errs):.5f}  "
                f"mean_logit_drift={sum(logit_drifts)/len(logit_drifts):.5f}"
            )

    return {
        "mean_l2_attn_output":   sum(l2_errs) / max(len(l2_errs), 1),
        "max_l2_attn_output":    max(l2_errs) if l2_errs else 0.0,
        "mean_logit_drift":      sum(logit_drifts) / max(len(logit_drifts), 1),
        "max_logit_drift":       max(logit_drifts) if logit_drifts else 0.0,
        "num_measurements":      len(l2_errs),
    }


@torch.no_grad()
def eval_perplexity(
    model,
    tokenizer,
    pipeline:    SphericalKVPipeline,
    eval_ids:    torch.Tensor,
    prefill_len: int = 64,
    stride:      int = 256,
    max_len:     int = 2048,
) -> dict:
    device = pipeline.device
    eval_ids = eval_ids[:max_len].to(device)
    N = eval_ids.numel()

    total_nll  = 0.0
    num_tokens = 0
    pos        = 0

    print(f"\n[eval_perplexity] {N} tokens  prefill={prefill_len}  stride={stride}")

    while pos + prefill_len + 1 <= N:
        window_end  = min(pos + prefill_len + stride, N)
        context_ids = eval_ids[pos : pos + prefill_len].unsqueeze(0)
        target_ids  = eval_ids[pos + prefill_len : window_end]

        pipeline.prefill(context_ids)

        for t_idx in range(len(target_ids)):
            input_ids = eval_ids[pos : pos + prefill_len + t_idx + 1].unsqueeze(0)
            out = model(input_ids=input_ids, use_cache=False, return_dict=True)
            logits = out.logits[0, -2, :]
            target = target_ids[t_idx]
            total_nll -= F.log_softmax(logits, dim=-1)[target].item()
            num_tokens += 1

        pos += stride
        ppl = math.exp(total_nll / max(num_tokens, 1))
        print(f"  pos={pos}  tokens_scored={num_tokens}  ppl={ppl:.3f}")

    ppl = math.exp(total_nll / max(num_tokens, 1))
    print(f"\n[eval_perplexity] Final ppl={ppl:.4f}  ({num_tokens} tokens scored)")
    return {
        "perplexity": ppl,
        "total_nll":  total_nll,
        "num_tokens": num_tokens,
    }



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--codebook_dir",
                   default="llama_3.2_1B_codebooks/codebooks_llama_1b")
    p.add_argument("--device",             default="cuda" if torch.cuda.is_available()
                                                          else "cpu")
    p.add_argument("--eval_text",          default=None)
    p.add_argument("--dataset",            default="pg19",
                   help="pg19 (paper W1) or wikitext")
    p.add_argument("--num_eval_tokens",    type=int, default=2048)
    p.add_argument("--prefill_len",        type=int, default=256)
    p.add_argument("--decode_steps",       type=int, default=16)
    p.add_argument("--sink_tokens",        type=int, default=4)
    p.add_argument("--n_warm",             type=int, default=N_WARM,
                   help="warmup tokens excluded from throughput measurement")
    p.add_argument("--n_meas",             type=int, default=N_MEAS,
                   help="measurement window tokens for tok/s and bHBM")
    p.add_argument("--n_trials",           type=int, default=N_TRIALS,
                   help="trials for median tok/s")
    p.add_argument("--skip_perplexity",    action="store_true")
    p.add_argument("--skip_throughput",    action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path, device)
    cfg          = model.config
    num_layers   = cfg.num_hidden_layers
    num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim     = getattr(cfg, "head_dim",
                           cfg.hidden_size // cfg.num_attention_heads)
    print(
        f"\nModel: layers={num_layers}  kv_heads={num_kv_heads}  "
        f"head_dim={head_dim}\n"
    )

    from tiers import build_tiers
    tiers_list = build_tiers(head_dim)

    codebooks = load_codebooks(
        codebook_dir=args.codebook_dir,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        tiers=tiers_list,
    )

    pipeline = SphericalKVPipeline(
        model=model, tokenizer=tokenizer,
        codebooks=codebooks, device=device,
        head_dim=head_dim, group_size=GROUP_SIZE,
        sink_tokens=args.sink_tokens,
        use_fused=(device.type == "cuda"),
    )

    eval_ids    = get_eval_tokens(tokenizer, args.eval_text, args.dataset,
                                  args.num_eval_tokens)
    prefill_ids = eval_ids[: args.prefill_len].unsqueeze(0).to(device)

    sep = "=" * 64

    # ── 1. Memory footprint ──────────────────────────────────────────────
    print(f"\n{sep}\n  Memory Accounting\n{sep}")
    mem = eval_memory_footprint(pipeline, prefill_ids)
    for k, v in mem.items():
        print(f"  {k:<40s}  {v:,.3f}" if isinstance(v, float)
              else f"  {k:<40s}  {v:,}")

    # ── 2. Throughput + HBM  ─────────────────────────────────────────────
    if not args.skip_throughput:
        print(f"\n{sep}\n  HBM bytes/token Analysis\n{sep}")
        print(
            f"  Warmup={args.n_warm} tokens (excluded)  "
            f"Measurement={args.n_meas} tokens  Trials={args.n_trials}"
        )
        thr = eval_throughput(
            pipeline, prefill_ids,
            n_warm=args.n_warm, n_meas=args.n_meas, n_trials=args.n_trials,
        )
        for k, v in thr.items():
            if k == "tok_s_all_trials":
                print(f"  {k:<40s}  {[f'{x:.1f}' for x in v]}")
            elif isinstance(v, float):
                print(f"  {k:<40s}  {v:.3f}")
            else:
                print(f"  {k:<40s}  {v}")
        print(
            "  NOTE: hbm_bytes_per_tok_proxy uses torch.cuda.memory_stats.\n"
            "  For hardware DRAM counters run:\n"
            "    ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum \\\n"
            "        --nvtx --nvtx-include angle_logits python evaluate.py ..."
        )

    if not args.skip_perplexity:
        print(f"\n{sep}\n  Perplexity\n{sep}")
        ppl_result = eval_perplexity(
            model, tokenizer, pipeline,
            eval_ids, prefill_len=args.prefill_len,
        )
        for k, v in ppl_result.items():
            if isinstance(v, float):
                print(f"  {k:<40s}  {v:.4f}")
            else:
                print(f"  {k:<40s}  {v}")

    pipeline.uninstall()
    print(f"\n{sep}\n  Done.\n{sep}")


if __name__ == "__main__":
    main()