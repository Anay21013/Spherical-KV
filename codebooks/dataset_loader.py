from datasets import load_dataset

def load_c4(num_samples: int):
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    texts = []
    for example in ds:
        text = example["text"]
        if text and len(text) > 200:
            texts.append(text)
        if len(texts) >= num_samples:
            break
    return texts
 
 
def load_wikitext(num_samples: int):
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
    return [t for t in ds["text"][:num_samples] if t and len(t) > 200]


def load_wikipedia_qa_style(num_samples: int, min_chars: int = 4000,
                            snippet_chars: int = 600):
    """
    Wikipedia paragraphs concatenated into LongBench-style multi-document
    prompts. Synthesizes the prompt structure (instruction + concatenated
    passages + generic question) WITHOUT using any LongBench data — no
    leakage, but matched domain (Wikipedia, where most LongBench QA
    passages come from) and matched attention pattern (model attends in
    "find-the-answer" mode rather than "predict-next-word" mode).

    Used for importance-stratified codebook calibration: the per-key
    attention scores captured during forward will reflect retrieval-style
    importance rather than language-modeling importance, so the codebook
    centroids end up where they're needed at inference for W2.

    Each prompt:
      <instruction>
      Passage 1: <wiki snippet>
      Passage 2: <wiki snippet>
      ...
      Question: <generic prompt>
      Answer:

    Returns: list of strings.
    """
    ds = load_dataset("wikipedia", "20220301.en", split="train",
                      streaming=True, trust_remote_code=True)

    instruction = (
        "Answer the question based on the passages below. "
        "Only give me the answer and do not output any other words.\n\n"
    )
    # Generic question — content doesn't matter much; what matters is the
    # *presence* of a question at the end so the model attends in QA mode.
    generic_questions = [
        "Question: What is the main topic discussed in the passages above?",
        "Question: Which entity is most prominently described?",
        "Question: What event or fact stands out across these passages?",
        "Question: When did the central event in the passages take place?",
        "Question: Where is the location most associated with these passages?",
    ]
    suffix = "\n\nAnswer:"

    out, buf, buf_len, qi = [], [], 0, 0
    for example in ds:
        text = example.get("text", "")
        if not text or len(text) < 200:
            continue
        # Truncate each article to a fixed-ish length so one long article
        # doesn't dominate a single calibration prompt.
        snippet = text[:snippet_chars].strip()
        buf.append(f"Passage {len(buf) + 1}: {snippet}")
        buf_len += len(snippet)

        if buf_len >= min_chars:
            q = generic_questions[qi % len(generic_questions)]
            qi += 1
            prompt = instruction + "\n\n".join(buf) + "\n\n" + q + suffix
            out.append(prompt)
            buf, buf_len = [], 0
            if len(out) >= num_samples:
                break

    return out
