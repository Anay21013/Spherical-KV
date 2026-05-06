import gc
import json
import math
import re
import string
import time
from collections import Counter
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F



def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = ' '.join(s.split())
    return s


def _extract_short_answer(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    first_line = text.split('\n')[0].strip()
    if len(first_line) > 150:
        dot = first_line.find('.')
        if dot > 0:
            first_line = first_line[:dot]
    for prefix in ["Answer:", "The answer is:", "The answer is",
                   "Based on the provided", "Based on the context",
                   "According to the"]:
        if first_line.lower().startswith(prefix.lower()):
            first_line = first_line[len(prefix):].strip().lstrip(':,. ')
    return first_line.strip()


def exact_match(prediction: str, gold_answers: List[str]) -> float:
    pred_norm = normalize_answer(prediction)
    for gold in gold_answers:
        if normalize_answer(gold) == pred_norm:
            return 1.0
    return 0.0


def f1_score(prediction: str, gold_answers: List[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0
    best_f1 = 0.0
    for gold in gold_answers:
        gold_tokens = normalize_answer(gold).split()
        if not gold_tokens:
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_common = sum(common.values())
        if num_common == 0:
            continue
        precision = num_common / len(pred_tokens)
        recall = num_common / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
    return best_f1



def _build_prompt(context: str, question: str, tokenizer=None) -> str:
    system = ("You are a helpful assistant. Answer the question in a few words "
              "based only on the provided context. Do not explain.")
    user_msg = (f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"Give a short, direct answer.")
    if tokenizer is not None and hasattr(tokenizer, 'apply_chat_template'):
        try:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return f"{system}\n\n{user_msg}\n\nAnswer:"



def load_longbench_dataset(
    task: str = "hotpotqa", split: str = "test",
    max_samples: int = 100, tokenizer=None,
    max_context_tokens: int = 0,
) -> List[dict]:
    from datasets import load_dataset
    ds = load_dataset("THUDM/LongBench", task, split=split)

    samples, skipped = [], 0
    for example in ds:
        if len(samples) >= max_samples:
            break
        context = example.get("context", "")
        question = example.get("input", "")
        answers = example.get("answers", [])
        if isinstance(answers, str):
            try:
                answers = json.loads(answers)
            except json.JSONDecodeError:
                answers = [answers]
        if not isinstance(answers, list):
            answers = [str(answers)]

        simple = f"{context}\n\nQuestion: {question}\n\nAnswer:"
        if tokenizer is not None and max_context_tokens > 0:
            n_tokens = len(tokenizer.encode(simple))
            if n_tokens > max_context_tokens - 128:
                skipped += 1
                continue
        else:
            n_tokens = example.get("length", 0)

        samples.append({"context": context, "answers": answers,
                        "question": question, "length": n_tokens})

    print(f"[W2] Loaded {len(samples)} from LongBench/{task}"
          + (f" (skipped {skipped} > {max_context_tokens} tokens)" if skipped else ""))
    return samples


def build_qa_prompt_with_boundaries(
    context, question, tokenizer,
    n_distractors=0, distractor_texts=None, answer_position="natural",
):
    docs = [p.strip() for p in context.split("\n\n") if p.strip()] or [context.strip()]
    if n_distractors > 0:
        distractors = distractor_texts or [
            f"Supplementary background (passage {i+1}), unrelated."
            for i in range(n_distractors)][:n_distractors]
        if answer_position == "late":
            all_docs = distractors + docs
        elif answer_position == "middle":
            m = len(distractors) // 2
            all_docs = distractors[:m] + docs + distractors[m:]
        else:
            all_docs = docs + distractors
    else:
        all_docs = docs

    parts = [f"Document {i+1}:\n{d}" for i, d in enumerate(all_docs)]
    ctx_block = "\n\n---\n\n".join(parts)
    prompt = _build_prompt(ctx_block, question, tokenizer)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    T = input_ids.shape[1]
    boundaries = []
    for doc in docs:
        cs = prompt.find(doc)
        if cs < 0:
            continue
        ts = min(len(tokenizer.encode(prompt[:cs], add_special_tokens=False)) + 1, T - 1)
        te = min(ts + len(tokenizer.encode(doc, add_special_tokens=False)), T)
        boundaries.append((ts, te))
    return prompt, input_ids, boundaries



@torch.no_grad()
def _generate_dense_hf(model, tokenizer, prompt, device, max_new=64):
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    t0 = time.perf_counter()
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                         num_beams=1, temperature=1.0, top_p=1.0)
    gen_time = time.perf_counter() - t0
    gen_ids = out[0, ids.shape[1]:]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return _extract_short_answer(raw), len(gen_ids), gen_time


@torch.no_grad()
def _generate_sphkv_hf(model, tokenizer, prompt, device, max_new=64):
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    T = ids.shape[1]
    gen = []
    t0 = time.perf_counter()
    for step in range(max_new):
        if step == 0:
            tok = ids[:, -1:]
            pos = torch.tensor([[T - 1]], device=device)
        else:
            tok = torch.tensor([[gen[-1]]], device=device)
            pos = torch.tensor([[T + step - 1]], device=device)
        out = model(input_ids=tok, use_cache=False, position_ids=pos)
        nxt = out.logits[0, -1].float().argmax().item()
        if nxt == tokenizer.eos_token_id:
            break
        gen.append(nxt)
    gen_time = time.perf_counter() - t0
    if not gen:
        return "", 0, gen_time
    return (_extract_short_answer(tokenizer.decode(gen, skip_special_tokens=True)),
            len(gen), gen_time)



@torch.no_grad()
def _generate_dense_vllm(vllm_engine, tokenizer, prompt, max_new=64):
    """Dense via vLLM engine (handles memory, fast prefill+decode)."""
    from vllm import SamplingParams
    params = SamplingParams(max_tokens=max_new, temperature=0, top_p=1.0)
    t0 = time.perf_counter()
    outputs = vllm_engine.generate([prompt], params, use_tqdm=False)
    gen_time = time.perf_counter() - t0
    raw = outputs[0].outputs[0].text.strip()
    n_toks = len(outputs[0].outputs[0].token_ids)
    return _extract_short_answer(raw), n_toks, gen_time


@torch.inference_mode()
def _generate_sphkv_vllm(vllm_fwd, tokenizer, input_ids, pipeline,
                          budget_bpt, device, max_new=64, boundaries=None):
    """SphKV via VLLMDirectForward + our CUDA kernel (~32 tok/s)."""
    input_ids = input_ids.to(device)
    T = input_ids.shape[1]

    # Free previous sample
    if hasattr(pipeline, '_lut_pools'):
        pipeline._lut_pools.clear()
    if hasattr(pipeline, 'per_head_pages'):
        pipeline.per_head_pages.clear()
    if hasattr(pipeline, '_token_states'):
        pipeline._token_states = []
    torch.cuda.empty_cache()

    # 1. Prefill
    kv_pairs, prefill_logits = vllm_fwd.prefill_capture(input_ids)

    # 2. Build K list for pipeline
    post_rope_K_list = []
    for (K_post, V_post) in kv_pairs:
        B, H, Tlen, D = K_post.shape
        K_t = K_post.permute(0, 2, 1, 3).contiguous().view(B, Tlen, H * D)
        post_rope_K_list.append(K_t)

    # 3. Set budget
    import config as _cfg
    _cfg.BITS_PER_TOKEN = budget_bpt
    _cfg.GLOBAL_BUDGET_BITS = budget_bpt * T * vllm_fwd.num_layers * vllm_fwd.num_kv

    attn_w = list(getattr(vllm_fwd, '_last_prefill_reuse_q', None)
                  or [None] * vllm_fwd.num_layers)
    ho = list(getattr(vllm_fwd, '_last_prefill_head_out', None)
              or [None] * vllm_fwd.num_layers)
    pipeline.prefill(
        kv_pairs=kv_pairs, attn_weights=attn_w, head_outputs=ho,
        pre_rope_K_list=post_rope_K_list, seq_len=T, skip_patch=True,
        retrieval_boundaries=boundaries or [])

    seg_stats = _extract_segment_tier_stats(pipeline)

    first_tok = prefill_logits[0, -1].float().argmax().item()
    del kv_pairs, post_rope_K_list, attn_w, ho, prefill_logits
    torch.cuda.empty_cache()

    gen = []
    if first_tok == tokenizer.eos_token_id:
        return "", 0, 0.0, seg_stats
    gen.append(first_tok)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(1, max_new):
        tok_in = torch.tensor([[gen[-1]]], device=device)
        logits = vllm_fwd.decode_step_sphkv(tok_in, T + step - 1, pipeline)
        nxt = logits[0, -1].float().argmax().item()
        if nxt == tokenizer.eos_token_id:
            break
        gen.append(nxt)

    torch.cuda.synchronize()
    decode_time = time.perf_counter() - t0

    if not gen:
        return "", 0, 0.0, seg_stats
    return (_extract_short_answer(tokenizer.decode(gen, skip_special_tokens=True)),
            len(gen), decode_time, seg_stats)



_SEG_NAMES = {0: "prefix", 1: "retrieved", 2: "recent"}


def _extract_segment_tier_stats(pipeline):
    stats = {}
    if not hasattr(pipeline, "_retained_tokens"):
        return stats

    total_per_seg = {0: 0, 1: 0, 2: 0}

    for tok in pipeline._retained_tokens:
        seg = getattr(tok, "segment_id", 0)
        tid = getattr(tok, "new_tier_id", 0)
        seg_name = _SEG_NAMES.get(seg, f"seg{seg}")
        key = (seg_name, tid)
        stats[key] = stats.get(key, 0) + 1
        total_per_seg[seg] = total_per_seg.get(seg, 0) + 1

    stats["_total_retained"] = sum(total_per_seg.values())
    stats["_per_segment_total"] = {
        _SEG_NAMES.get(k, f"seg{k}"): v for k, v in total_per_seg.items()
    }
    return stats


def _format_segment_tier_table(stats):
    if not stats:
        return ""
    seg_totals = stats.get("_per_segment_total", {})
    lines = []
    lines.append(f"    {'Segment':<10} {'b1':>8} {'b2':>8} {'b3':>8} {'Total':>8}")
    for seg_name in ["prefix", "retrieved", "recent"]:
        b1 = stats.get((seg_name, 1), 0)
        b2 = stats.get((seg_name, 2), 0)
        b3 = stats.get((seg_name, 3), 0)
        total = seg_totals.get(seg_name, 0)
        if total > 0:
            lines.append(f"    {seg_name:<10} {b1:>8} {b2:>8} {b3:>8} {total:>8}")
    return "\n".join(lines)



@torch.inference_mode()
def evaluate_w2(
    model, tokenizer, pipeline,
    samples: List[dict], device: torch.device,
    max_new_tokens: int = 64,
    mode: str = "dense",
    n_distractors: int = 0,
    answer_position: str = "natural",
    budget_bpt: float = 96.0,
    vllm_fwd=None,
    vllm_engine=None,
) -> dict:
    all_em, all_f1 = [], []
    total_gen_tokens = 0
    total_decode_time = 0.0
    total_ctx_tokens = 0
    agg_seg_stats = {}   # paper §3.3: segment-wise tier counts (sphkv only)
    use_vllm = (vllm_fwd is not None) or (vllm_engine is not None)

    if pipeline is not None and mode != "dense":
        try:
            import config as _cfg
            _cfg.BITS_PER_TOKEN = budget_bpt
        except Exception:
            pass

    for i, sample in enumerate(samples):
        prompt, input_ids, boundaries = build_qa_prompt_with_boundaries(
            sample["context"], sample["question"], tokenizer,
            n_distractors=n_distractors, answer_position=answer_position)

        prefill_len = input_ids.shape[1]
        total_ctx_tokens += prefill_len

        try:
            if mode == "dense":
                if vllm_engine is not None:
                    pred, n_tok, dt = _generate_dense_vllm(
                        vllm_engine, tokenizer, prompt, max_new_tokens)
                else:
                    pred, n_tok, dt = _generate_dense_hf(
                        model, tokenizer, prompt, device, max_new_tokens)
            else:
                if vllm_fwd is not None:
                    pred, n_tok, dt, samp_stats = _generate_sphkv_vllm(
                        vllm_fwd, tokenizer, input_ids, pipeline,
                        budget_bpt, device, max_new_tokens,
                        boundaries=boundaries)
                    for k, v in samp_stats.items():
                        if not isinstance(k, tuple):
                            continue
                        agg_seg_stats[k] = agg_seg_stats.get(k, 0) + v
                else:
                    pipeline.prefill(input_ids.to(device),
                                     retrieval_boundaries=boundaries)
                    pred, n_tok, dt = _generate_sphkv_hf(
                        model, tokenizer, prompt, device, max_new_tokens)

            total_gen_tokens += max(n_tok, 1)
            total_decode_time += dt
        except Exception as e:
            pred = ""
            total_gen_tokens += 1
            print(f"  [W2] Sample {i} error: {e}")

        em = exact_match(pred, sample["answers"])
        f1 = f1_score(pred, sample["answers"])
        all_em.append(em)
        all_f1.append(f1)

        if i < 3:
            gold_str = " | ".join(sample["answers"][:2]) if isinstance(sample["answers"], list) else str(sample["answers"])
            pred_show = (pred[:120] + "...") if len(pred) > 120 else pred
            gold_show = (gold_str[:120] + "...") if len(gold_str) > 120 else gold_str
            print(f"    [{mode} sample {i}] F1={f1*100:.1f}%")
            print(f"      gold: {gold_show!r}")
            print(f"      pred: {pred_show!r}")

        if pipeline is not None and hasattr(pipeline, '_patched') and pipeline._patched:
            pipeline.uninstall()

        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            print(f"  [{mode}] {i+1}/{len(samples)}  "
                  f"EM={100*sum(all_em)/len(all_em):.1f}%  "
                  f"F1={100*sum(all_f1)/len(all_f1):.1f}%")

        if mode != "dense" and (i + 1) % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    avg_em = sum(all_em) / max(len(all_em), 1)
    avg_f1 = sum(all_f1) / max(len(all_f1), 1)

    tok_s = total_gen_tokens / max(total_decode_time, 1e-6)

    bKV = 0.0
    src = model if model is not None else vllm_fwd
    if src is not None:
        if hasattr(src, 'config'):
            cfg = src.config
            L = cfg.num_hidden_layers
            H = getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)
            dh = getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)
        else:
            L, H, dh = src.num_layers, src.num_kv, src.dh
        if mode == "dense":
            bKV = L * H * (dh * 2 + dh * 2)
        else:
            bKV = L * H * (math.ceil(budget_bpt / 8) + dh * 2)

    hBM_per_tok = bKV  # resident bytes ~ HBM bytes per decode step (no dense reconstruction)

    seg_table = ""
    if agg_seg_stats:
        print(f"\n  ── Segment-wise retention + tiering (controller evidence, §3.3) ──")
        seg_table = _format_segment_tier_table(_make_pretty_seg_stats(agg_seg_stats))
        print(seg_table)

    return {
        "mode": mode, "workload": "w2",
        "em": avg_em, "f1": avg_f1,
        "Q": avg_f1 * 100,
        "tok_s": tok_s, "bKV": bKV,
        "hBM_per_tok": hBM_per_tok,
        "budget_bpt": budget_bpt if mode != "dense" else None,
        "n_samples": len(samples), "n_distractors": n_distractors,
        "answer_position": answer_position,
        "avg_context_len": total_ctx_tokens / max(len(samples), 1),
        "decode_time_s": total_decode_time,
        "em_all": all_em, "f1_all": all_f1,
        "segment_tier_counts": agg_seg_stats,   # paper §3.3 evidence
        "segment_tier_table": seg_table,
    }


def _make_pretty_seg_stats(agg):
    """Re-add _per_segment_total summary after aggregation across samples."""
    per_seg = {}
    for (seg_name, tid), n in agg.items():
        if isinstance(seg_name, str):
            per_seg[seg_name] = per_seg.get(seg_name, 0) + n
    out = dict(agg)
    out["_per_segment_total"] = per_seg
    return out
