import re
import string
from collections import Counter
from typing import List, Dict, Optional, Tuple

import torch


def load_longbench_dataset(
    task:        str = "hotpotqa",
    split:       str = "test",
    max_samples: int = 100,
) -> List[dict]:
    """
    Load a LongBench task.

    Returns list of dicts with keys:
        input, context, answers, question, length
    """
    from datasets import load_dataset

    ds = load_dataset("THUDM/LongBench", task, split=split,
                      trust_remote_code=True)

    samples = []
    for i, example in enumerate(ds):
        if i >= max_samples:
            break

        sample = {
            "input":    example.get("input", ""),
            "context":  example.get("context", ""),
            "answers":  example.get("answers", [example.get("answer", "")]),
            "question": example.get("input", ""),
            "length":   example.get("length", 0),
        }

        if isinstance(sample["answers"], str):
            sample["answers"] = [sample["answers"]]

        samples.append(sample)

    print(f"[W2] Loaded {len(samples)} samples from LongBench/{task}")
    return samples



DOCUMENT_SEP = "\n\n---\n\n"
RETRIEVAL_HEADER = "Document {idx}:\n"


def build_qa_prompt_with_boundaries(
    context:   str,
    question:  str,
    tokenizer,
    n_distractors: int = 0,
    distractor_texts: Optional[List[str]] = None,
    answer_position:  str = "natural",
) -> Tuple[str, torch.Tensor, List[Tuple[int, int]]]:
    """
    Build QA prompt and extract retrieval boundaries as token index ranges.

    Parameters
    ----------
    context         : The retrieved context (may contain multiple docs separated by \\n\\n)
    question        : The question
    tokenizer       : HF tokenizer for computing token boundaries
    n_distractors   : Number of distractors to append (Section 3.3)
    distractor_texts: Distractor passages (if None, use dummy text)
    answer_position : "early", "middle", "late", "natural" -- controls where
                      the real context is placed among distractors

    Returns
    -------
    prompt_text:          Full prompt string
    input_ids:            [1, T] tensor
    retrieval_boundaries: list of (start_tok, end_tok) for retrieved doc spans
    """
    prefix = "Read the following documents and answer the question.\n\n"

    # Split context into documents
    docs = _split_documents(context)

    # Add distractors
    if n_distractors > 0:
        distractors = distractor_texts or [
            _make_dummy_distractor(i) for i in range(n_distractors)
        ]
        distractors = distractors[:n_distractors]

        if answer_position == "early":
            all_docs = docs + distractors
        elif answer_position == "late":
            all_docs = distractors + docs
        elif answer_position == "middle":
            mid = len(distractors) // 2
            all_docs = distractors[:mid] + docs + distractors[mid:]
        else:  # natural
            all_docs = docs + distractors
    else:
        all_docs = docs

    # Build prompt with document markers
    doc_parts = []
    for idx, doc in enumerate(all_docs):
        doc_parts.append(f"{RETRIEVAL_HEADER.format(idx=idx+1)}{doc}")

    context_block = DOCUMENT_SEP.join(doc_parts)
    suffix = f"\n\nQuestion: {question}\n\nAnswer:"
    prompt_text = prefix + context_block + suffix

    # Tokenize to get token-level boundaries
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    T = input_ids.shape[1]

    # Find retrieval boundaries (token positions of the real docs, not distractors)
    retrieval_boundaries = _find_retrieval_boundaries(
        prompt_text, tokenizer, docs, n_distractors, answer_position, T
    )

    return prompt_text, input_ids, retrieval_boundaries


def _split_documents(context: str) -> List[str]:
    """Split context into individual documents."""
    # Try common separators
    for sep in ["\n\n\n", "\n\n---\n\n", "\n---\n", "\n\n"]:
        parts = context.split(sep)
        if len(parts) > 1:
            return [p.strip() for p in parts if p.strip()]
    return [context.strip()]


def _make_dummy_distractor(idx: int) -> str:
    """Generate a simple distractor passage."""
    return (
        f"This is supplementary background information (passage {idx+1}). "
        f"It discusses related topics but does not contain the answer to "
        f"the question being asked. The content here is meant to serve as "
        f"a distractor to test the model's ability to locate relevant "
        f"information among multiple documents."
    )


def _find_retrieval_boundaries(
    prompt_text: str,
    tokenizer,
    real_docs:   List[str],
    n_distractors: int,
    answer_position: str,
    T: int,
) -> List[Tuple[int, int]]:
    """
    Find token-level start/end for the real retrieved documents.
    Returns list of (start_tok_idx, end_tok_idx) tuples.
    """
    boundaries = []

    for doc in real_docs:
        # Find the character position of this doc in the prompt
        char_start = prompt_text.find(doc)
        if char_start < 0:
            continue
        char_end = char_start + len(doc)

        # Convert character positions to token positions
        # Encode prefix up to char_start to get token offset
        prefix_text = prompt_text[:char_start]
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        tok_start = len(prefix_ids) + 1  # +1 for BOS if present

        doc_text = prompt_text[char_start:char_end]
        doc_ids = tokenizer.encode(doc_text, add_special_tokens=False)
        tok_end = tok_start + len(doc_ids)

        # Clamp to valid range
        tok_start = max(0, min(tok_start, T - 1))
        tok_end = max(tok_start + 1, min(tok_end, T))

        boundaries.append((tok_start, tok_end))

    if not boundaries and T > 0:
        # Fallback: mark middle 60% as retrieved
        start = int(T * 0.1)
        end = int(T * 0.7)
        boundaries.append((start, end))

    return boundaries


# =========================================================================
# EM and F1 scoring (SQuAD-style)
# =========================================================================

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = ' '.join(s.split())
    return s


def exact_match(prediction: str, gold_answers: List[str]) -> float:
    pred_norm = normalize_answer(prediction)
    for gold in gold_answers:
        if normalize_answer(gold) == pred_norm:
            return 1.0
    return 0.0


def f1_score(prediction: str, gold_answers: List[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    best_f1 = 0.0
    for gold in gold_answers:
        gold_tokens = normalize_answer(gold).split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_common = sum(common.values())
        if num_common == 0:
            continue
        precision = num_common / len(pred_tokens) if pred_tokens else 0
        recall = num_common / len(gold_tokens) if gold_tokens else 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            best_f1 = max(best_f1, f1)
    return best_f1


# =========================================================================
# W2 evaluation
# =========================================================================

@torch.no_grad()
def evaluate_w2(
    model,
    tokenizer,
    pipeline,
    samples:        List[dict],
    device:         torch.device,
    max_new_tokens: int = 64,
    mode:           str = "dense",
    n_distractors:  int = 0,
    answer_position: str = "natural",
) -> dict:
    """
    Run W2 evaluation with retrieval boundary tagging.

    For SphericalKV: passes retrieval_boundaries to prefill() so the
    controller uses segment_id=1 (weight=1.5) for retrieved passages.
    """
    all_em = []
    all_f1 = []
    all_seg_stats = []

    for i, sample in enumerate(samples):
        prompt_text, input_ids, retrieval_boundaries = \
            build_qa_prompt_with_boundaries(
                sample["context"], sample["question"], tokenizer,
                n_distractors=n_distractors,
                answer_position=answer_position)

        input_ids = input_ids.to(device)

        if pipeline is not None and mode != "dense":
            pipeline.prefill(input_ids,
                             retrieval_boundaries=retrieval_boundaries)

        # Generate
        try:
            generated = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
            gen_ids = generated[0, input_ids.shape[1]:]
            prediction = tokenizer.decode(gen_ids,
                                          skip_special_tokens=True).strip()
        except Exception as e:
            prediction = ""
            print(f"  [W2] Generation error at sample {i}: {e}")

        em = exact_match(prediction, sample["answers"])
        f1 = f1_score(prediction, sample["answers"])
        all_em.append(em)
        all_f1.append(f1)

        # Collect segment stats if pipeline has retained tokens
        if pipeline is not None and hasattr(pipeline, '_retained_tokens'):
            seg_counts = {0: 0, 1: 0, 2: 0}
            for ts in pipeline._retained_tokens:
                seg_counts[ts.segment_id] = seg_counts.get(ts.segment_id, 0) + 1
            all_seg_stats.append(seg_counts)

        if pipeline is not None and pipeline._patched:
            pipeline.uninstall()

        if (i + 1) % 10 == 0:
            print(f"  [{mode}] {i+1}/{len(samples)}  "
                  f"EM={sum(all_em)/len(all_em):.3f}  "
                  f"F1={sum(all_f1)/len(all_f1):.3f}")

    # Aggregate segment stats
    seg_summary = {}
    if all_seg_stats:
        for seg_id in [0, 1, 2]:
            vals = [s.get(seg_id, 0) for s in all_seg_stats]
            seg_summary[seg_id] = sum(vals) / max(len(vals), 1)

    return {
        "mode":          mode,
        "em":            sum(all_em) / max(len(all_em), 1),
        "f1":            sum(all_f1) / max(len(all_f1), 1),
        "n_samples":     len(samples),
        "n_distractors": n_distractors,
        "answer_position": answer_position,
        "em_all":        all_em,
        "f1_all":        all_f1,
        "seg_retention": seg_summary,
    }