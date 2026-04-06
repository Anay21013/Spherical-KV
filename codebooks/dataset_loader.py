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
