# Spherical KV — Supplementary Code

Code for reproducing the experiments in the paper:
**"Spherical KV: Angle-Domain Attention with Rate-Distortion KV Cache Compression"**

## Setup

```bash
conda create -n sphkv python=3.11 -y
conda activate sphkv
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
huggingface-cli login
```

### requirements.txt

```
transformers>=4.45.0
datasets
scikit-learn
numpy
tqdm
matplotlib
seaborn
```

### Hardware

| Task | Minimum GPU | Recommended |
|------|-------------|-------------|
| Codebook training (1B) | 4 GB VRAM | RTX 3050+ |
| Codebook training (8B+) | 24 GB VRAM | A6000 / A100 |
| W1 experiments (8K ctx) | 24 GB VRAM | A100-80GB |
| W1 experiments (32K+ ctx) | 48 GB VRAM | A100-80GB |
| W2/W3 experiments | 24 GB VRAM | A100-80GB |
| CUDA kernel compilation | CUDA 12.1+ with nvcc | — |

## Reproducing Results

All experiments are run through `experiment_runner.py`. Results are saved to `experiment_results/`.

### Step 0: Train codebooks (one-time per model)

```bash
python generate.py
```

Codebooks are saved to the directory specified by `SAVE_DIR` in `config.py`. Training takes ~2h on A100 (MiniBatchKMeans) or ~13h (full KMeans). Train once, reuse across all experiments.

To change the target model, edit `config.py`:
```python
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SAVE_DIR   = "codebooks/codebooks_llama_8b"
```

### Step 1: W1 — Long-context language modeling (Table 4, Figure 4)

PG-19 token-level NLL and perplexity with strided sliding-window protocol.

```bash
python experiment_runner.py \
  --models meta-llama/Llama-3.1-8B-Instruct \
  --codebook_dirs codebooks/codebooks_llama_8b \
  --workloads w1 \
  --context_lengths 8192 32768 \
  --modes dense sphkv sphkv_recon sphkv_angle sphkv_rd \
  --budgets 48 56 64 80 96 112 128 160 \
  --n_warm 8 --n_meas 64 --n_trials 3 \
  --device cuda
```

### Step 2: W2 — Retrieval QA (Table 5, Figure 5 A4)

Multi-hop QA on HotpotQA and 2WikiMultiHopQA with distractor and position sweeps.

```bash
python experiment_runner.py \
  --models meta-llama/Llama-3.1-8B-Instruct \
  --codebook_dirs codebooks/codebooks_llama_8b \
  --workloads w2 \
  --modes dense sphkv \
  --budgets 60 \
  --w2_task hotpotqa \
  --w2_max_samples 50 \
  --w2_distractors 0 3 5 \
  --w2_positions early middle late \
  --device cuda
```

Repeat with `--w2_task 2wikimqa` for the second dataset.

### Step 3: W3 — Agentic rollouts (Table 6, Figure 5 A5)

Multi-step tool-use trajectories measuring behavioral divergence.

```bash
python experiment_runner.py \
  --models meta-llama/Llama-3.1-8B-Instruct \
  --codebook_dirs codebooks/codebooks_llama_8b \
  --workloads w3 \
  --modes dense sphkv \
  --budgets 60 \
  --w3_source toolbench \
  --w3_max_episodes 30 \
  --w3_max_steps 10 \
  --w3_seeds 3 \
  --device cuda
```

### Step 4: Ablations (Figure 5 A0-A3)

```bash
python experiment_runner.py \
  --models meta-llama/Llama-3.1-8B-Instruct \
  --codebook_dirs codebooks/codebooks_llama_8b \
  --workloads w1 \
  --context_lengths 8192 32768 \
  --modes dense sphkv sphkv_recon sphkv_angle sphkv_rd \
         keepdrop quant_only decoupled \
  --budgets 48 64 80 112 160 \
  --device cuda
```

### HBM traffic measurement (Table A.2)

```bash
ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum \
    --nvtx --nvtx-include angle_logits \
    --csv --log-file ncu_report.csv \
    python experiment_runner.py --modes sphkv --budgets 60
```

## File Structure

```
spherical_kv_pipeline.py     Main pipeline: ADA attention + RDR allocation (Algorithm 1)
decode_kernel.cu             Fused CUDA decode kernel with per-position Q back-rotation
fused_decode.cpp             C++ binding for the CUDA kernel
fused_decode.py              Python wrapper for the kernel

allocation.py                Greedy knapsack tier allocator (Algorithm 1)
distortion_proxy.py          Calibrated distortion proxy (Appendix C)
calibrate_lambda.py          Offline calibration of tier-specific lambda coefficients

config.py                    Tier definitions, model config, budget parameters
tiers.py                     Tier dataclass and builder
token_state.py               Per-token state tracking (tier, segment, age)
paging.py                    Page layout and bitpacking for compressed KV
pagebuilder.py               Page construction from quantized codes
pointer_table.py             Page table for kernel dispatch
bitpacking.py                Bit-level packing utilities

generate.py                  Codebook training (pre-RoPE K-means on C4)
codebook_loader.py           Load trained codebooks from disk
quantization.py              Spherical quantization (radius + angular codes)
spherical_parameterization.py  K -> (r, theta) decomposition per group

llama_hooks.py               Pre-RoPE K capture hooks + patched decode forward
resuse_proxy.py              Token reuse proxy (EMA attention weights)
stability_proxy.py           Logit stability proxy for drift detection

experiment_runner.py         Full experiment harness for W1/W2/W3
evaluate.py                  Standalone evaluation with strided perplexity
results.py                   Result aggregation and paper figure generation
visualize.py                 Interactive tier allocation dashboard

dataset_loader.py            PG-19 / C4 data loading
```

## Configuration

Key parameters in `config.py`:

| Parameter | Description |
|-----------|-------------|
| `MODEL_NAME` | HuggingFace model identifier |
| `TIERS` | List of (tier_id, name, group_size, b_theta, K_centroids) |
| `SEQ_LEN` | Sequence length for codebook training |
| `NUM_SAMPLES` | Number of C4 samples for codebook training |
| `SAVE_DIR` | Output directory for trained codebooks |

Tier definitions follow the paper (Section 2.2):

| Tier | Name | Group size (g) | b_theta | Centroids (K) |
|------|------|----------------|---------|----------------|
| 1 | High | 16 | 6 | 64 |
| 2 | Mid | 16 | 4 | 16 |
| 3 | Low | 32 | 3 | 8 |

## Expected Results

Results to be filled after experiments on target hardware.

### W1: Language Modeling (PG-19)

| Model | Context | Dense PPL | SphKV PPL | PPL ratio | Speedup | KV Reduction |
|-------|---------|-----------|-----------|-----------|---------|--------------|
| | 8K | | | | | |
| | 32K | | | | | |
| | 128K | | | | | |

### W2: Retrieval QA (HotpotQA)

| Model | Distractors | Dense EM | SphKV EM | Dense F1 | SphKV F1 |
|-------|-------------|----------|----------|----------|----------|
| | 0 | | | | |
| | 3 | | | | |
| | 5 | | | | |

### W3: Agentic Rollouts

| Model | Dense success | SphKV success | Disagree rate |
|-------|--------------|---------------|---------------|
| | | | |

## License

This code is provided for review purposes only.
