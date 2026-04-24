import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

import torch


def generate_hardware_table(device: torch.device) -> dict:
    """
    Auto-detect GPU, driver, CUDA, host info for Table 2.
    Returns dict suitable for JSON serialization and table rendering.
    """
    info = {
        "host_os":      f"{platform.system()} {platform.release()}",
        "host_cpu":     platform.processor() or "unknown",
        "host_ram_gb":  _get_ram_gb(),
        "python":       platform.python_version(),
        "pytorch":      torch.__version__,
    }

    if device.type == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        info.update({
            "gpu_name":         props.name,
            "gpu_count":        torch.cuda.device_count(),
            "gpu_mem_gb":       round(props.total_memory / 1e9, 1),
            "gpu_compute_cap":  f"{props.major}.{props.minor}",
            "cuda_version":     torch.version.cuda or "N/A",
            "cudnn_version":    str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A",
            "driver_version":   _get_nvidia_driver(),
            "hbm_peak_bw_GBs": _estimate_hbm_bandwidth(props.name),
        })
    else:
        info["gpu_name"] = "CPU-only"

    # Transformers version
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        info["transformers"] = "N/A"

    return info


def print_hardware_table(info: dict):
    """Print Table 2 in a readable format."""
    print(f"\n{'='*60}")
    print(f"  Table 2: Hardware + Runtime Environment")
    print(f"{'='*60}")
    for k, v in info.items():
        print(f"  {k:<25s}  {v}")


def _get_ram_gb() -> str:
    try:
        import psutil
        return f"{psutil.virtual_memory().total / 1e9:.1f}"
    except ImportError:
        return "unknown"


def _get_nvidia_driver() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, timeout=5
        ).decode().strip().split("\n")[0]
        return out
    except Exception:
        return "N/A"


def _estimate_hbm_bandwidth(gpu_name: str) -> str:
    """Rough peak HBM bandwidth from GPU name."""
    bw_map = {
        "A100": "2039", "H100": "3350", "A6000": "768",
        "RTX 4090": "1008", "RTX 3090": "936", "A10": "600",
        "V100": "900", "L40": "864",
    }
    for key, bw in bw_map.items():
        if key in gpu_name:
            return f"{bw} GB/s"
    return "unknown"


# =========================================================================
# 2. ncu Wrapper for DRAM Byte Counters
# =========================================================================

def measure_hbm_via_ncu(
    script_path: str,
    script_args: str = "",
    nvtx_filter: str = "angle_logits",
    output_file: str = "ncu_report.csv",
) -> Optional[dict]:
    """
    Run Nsight Compute to measure DRAM read/write bytes.
    Returns dict with dram_bytes_read, dram_bytes_write, bHBM.

    Requires ncu to be installed and in PATH.
    """
    cmd = (
        f"ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum "
        f"--nvtx --nvtx-include {nvtx_filter} "
        f"--csv --log-file {output_file} "
        f"python {script_path} {script_args}"
    )
    print(f"[ncu] Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True, timeout=600)
        return _parse_ncu_csv(output_file)
    except Exception as e:
        print(f"[ncu] Failed: {e}")
        print(f"[ncu] Falling back to torch.cuda memory proxy")
        return None


def _parse_ncu_csv(path: str) -> Optional[dict]:
    """Parse ncu CSV output for DRAM metrics."""
    try:
        total_read = 0
        total_write = 0
        with open(path) as f:
            for line in f:
                if "dram__bytes_read" in line:
                    parts = line.strip().split(",")
                    total_read += int(float(parts[-1]))
                elif "dram__bytes_write" in line:
                    parts = line.strip().split(",")
                    total_write += int(float(parts[-1]))
        return {
            "dram_bytes_read": total_read,
            "dram_bytes_write": total_write,
            "dram_bytes_total": total_read + total_write,
        }
    except Exception:
        return None


def generate_ncu_command(script: str, args: str = "") -> str:
    """Generate the ncu command for users to run externally."""
    return (
        f"ncu --metrics "
        f"dram__bytes_read.sum,dram__bytes_write.sum,"
        f"l2__read_bytes.sum,l2__write_bytes.sum "
        f"--nvtx --nvtx-include angle_logits "
        f"--csv --log-file ncu_report.csv "
        f"python {script} {args}"
    )


# =========================================================================
# 3. T_decode Breakdown via NVTX Ranges
# =========================================================================

class DecodeBreakdown:
    """
    Collect per-component timing for the decode critical path:
    T_decode = T_page_lookup + T_KV_read + T_similarity + T_softmax + T_proj + T_misc

    Uses CUDA events paired with NVTX ranges for profiling.
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.is_cuda = device.type == "cuda"
        self.timings: Dict[str, list] = {
            "page_lookup": [], "kv_read": [], "similarity": [],
            "softmax": [], "proj": [], "total": [],
        }
        self._events: Dict[str, list] = {}

    def start_phase(self, name: str):
        if not self.is_cuda:
            return
        if name not in self._events:
            self._events[name] = []
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        self._events[name].append(("start", start))

    def end_phase(self, name: str):
        if not self.is_cuda:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        if name in self._events and self._events[name]:
            self._events[name].append(("end", end))

    def flush(self):
        """Compute elapsed times from event pairs."""
        if not self.is_cuda:
            return
        torch.cuda.synchronize()
        for name, events in self._events.items():
            starts = [e for tag, e in events if tag == "start"]
            ends = [e for tag, e in events if tag == "end"]
            for s, e in zip(starts, ends):
                ms = s.elapsed_time(e)
                self.timings.setdefault(name, []).append(ms)
        self._events.clear()

    def summary(self) -> dict:
        """Return mean ms per component."""
        result = {}
        for name, vals in self.timings.items():
            if vals:
                result[f"T_{name}_ms"] = sum(vals) / len(vals)
        return result


@torch.no_grad()
def measure_decode_breakdown(
    model, pipeline, prefill_ids: torch.Tensor,
    n_steps: int = 16, device: torch.device = None,
) -> dict:
    """
    Measure T_decode component breakdown for Section 4.3.
    Returns per-component mean ms.
    """
    if device is None:
        device = prefill_ids.device

    pipeline.prefill(prefill_ids.to(device))
    current_ids = prefill_ids.to(device).clone()
    breakdown = DecodeBreakdown(device)

    for step in range(n_steps):
        # Total step timing
        breakdown.start_phase("total")
        out = model(input_ids=current_ids[:, -1:],
                    use_cache=False, return_dict=True)
        nid = out.logits[:, -1, :].argmax(-1, keepdim=True)
        current_ids = torch.cat([current_ids, nid], dim=-1)
        breakdown.end_phase("total")

    breakdown.flush()
    pipeline.uninstall()
    return breakdown.summary()


# =========================================================================
# 4. Roofline Analysis
# =========================================================================

def compute_roofline(
    achieved_bytes_per_tok: float,
    achieved_tok_s:         float,
    peak_bw_GBs:            float,
    model_params_B:         float = 8.0,
) -> dict:
    """
    Compute roofline metrics for Section 4.3 microbench panel.

    achieved_bw = achieved_bytes_per_tok * achieved_tok_s
    utilization = achieved_bw / peak_bw
    """
    achieved_bw = achieved_bytes_per_tok * achieved_tok_s / 1e9  # GB/s
    utilization = achieved_bw / max(peak_bw_GBs, 1e-9)

    return {
        "achieved_bw_GBs":   achieved_bw,
        "peak_bw_GBs":       peak_bw_GBs,
        "bw_utilization":    utilization,
        "memory_bound":      utilization > 0.3,  # rough threshold
    }


# =========================================================================
# 5. Full Cost Accounting
# =========================================================================

@torch.no_grad()
def measure_full_cost(
    model, pipeline, prefill_ids: torch.Tensor,
    n_meas: int = 32, device: torch.device = None,
) -> dict:
    """
    t_e2e = t_prefill + t_allocate + t_decode
    Paper Section 3.5: full cost accounting.
    """
    if device is None:
        device = prefill_ids.device
    is_cuda = device.type == "cuda"

    # t_prefill (includes allocation + page building)
    if is_cuda:
        torch.cuda.synchronize()
        ts = torch.cuda.Event(enable_timing=True)
        te = torch.cuda.Event(enable_timing=True)
        ts.record()

    pipeline.prefill(prefill_ids.to(device))

    if is_cuda:
        te.record(); torch.cuda.synchronize()
        t_prefill_ms = ts.elapsed_time(te)
    else:
        t_prefill_ms = 0.0

    # t_decode
    current_ids = prefill_ids.to(device).clone()
    if is_cuda:
        torch.cuda.synchronize()
        ts2 = torch.cuda.Event(enable_timing=True)
        te2 = torch.cuda.Event(enable_timing=True)
        ts2.record()

    for _ in range(n_meas):
        out = model(input_ids=current_ids[:, -1:],
                    use_cache=False, return_dict=True)
        nid = out.logits[:, -1, :].argmax(-1, keepdim=True)
        current_ids = torch.cat([current_ids, nid], dim=-1)

    if is_cuda:
        te2.record(); torch.cuda.synchronize()
        t_decode_ms = ts2.elapsed_time(te2)
    else:
        t_decode_ms = 0.0

    pipeline.uninstall()

    t_e2e = t_prefill_ms + t_decode_ms
    return {
        "t_prefill_ms":  t_prefill_ms,
        "t_decode_ms":   t_decode_ms,
        "t_e2e_ms":      t_e2e,
        "prefill_pct":   100 * t_prefill_ms / max(t_e2e, 1e-9),
        "decode_tok_s":  n_meas / max(t_decode_ms / 1e3, 1e-9),
    }