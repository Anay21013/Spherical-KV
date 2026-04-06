MODEL_NAME  = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE      = "cuda"
 
SEQ_LEN     = 512
NUM_SAMPLES = 2000

TIERS = [
    (1,        "High", 16,  6,       64),
    (2,        "Mid",  16,  4,       16),
    (3,        "Low",  32,  3,        8),
]
 
SAVE_DIR = "codebooks_llama_1b"