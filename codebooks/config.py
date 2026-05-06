MODEL_NAME  = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE      = "cuda"
 
SEQ_LEN     = 2048
NUM_SAMPLES = 200

TIERS = [
    (1,        "High", 16,  6,       64),
    (2,        "Mid",  16,  4,       16),
    (3,        "Low",  32,  3,        8),
]
 
SAVE_DIR = "codebooks_llama_8b_v9"
