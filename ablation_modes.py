from __future__ import annotations
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, Optional

import config as _cfg
import distortion_proxy as _dp
import token_state as _ts



def _mutate_dict(target: dict, updates: dict) -> None:
    for k, v in updates.items():
        target[k] = v


def _snapshot_dict(d: dict) -> dict:
    return {k: v for k, v in d.items()}


def _set_lambdas(lt: Dict[int, float], lr: Dict[int, float]) -> Dict[str, Any]:
    saved = {
        "LAMBDA_THETA_cfg": _snapshot_dict(_cfg.LAMBDA_THETA),
        "LAMBDA_THETA_dp":  _snapshot_dict(_dp.LAMBDA_THETA),
        "LAMBDA_R_cfg":     _snapshot_dict(_cfg.LAMBDA_R),
        "LAMBDA_R_dp":      _snapshot_dict(_dp.LAMBDA_R),
    }
    # config.LAMBDA_THETA and distortion_proxy.LAMBDA_THETA currently reference
    # the SAME dict object, but we defensively update both in case someone
    # reassigns one of them later.
    for target in {id(_cfg.LAMBDA_THETA): _cfg.LAMBDA_THETA,
                   id(_dp.LAMBDA_THETA):  _dp.LAMBDA_THETA}.values():
        _mutate_dict(target, lt)
    for target in {id(_cfg.LAMBDA_R): _cfg.LAMBDA_R,
                   id(_dp.LAMBDA_R):  _dp.LAMBDA_R}.values():
        _mutate_dict(target, lr)
    return saved


def _restore_lambdas(saved: Dict[str, Any]) -> None:
    if "LAMBDA_THETA_cfg" in saved:
        _cfg.LAMBDA_THETA.clear()
        _cfg.LAMBDA_THETA.update(saved["LAMBDA_THETA_cfg"])
    if "LAMBDA_THETA_dp" in saved:
        _dp.LAMBDA_THETA.clear()
        _dp.LAMBDA_THETA.update(saved["LAMBDA_THETA_dp"])
    if "LAMBDA_R_cfg" in saved:
        _cfg.LAMBDA_R.clear()
        _cfg.LAMBDA_R.update(saved["LAMBDA_R_cfg"])
    if "LAMBDA_R_dp" in saved:
        _dp.LAMBDA_R.clear()
        _dp.LAMBDA_R.update(saved["LAMBDA_R_dp"])



def _pin_tier_hook(target_tier_id: int) -> Callable:
    """
    Hook: force every non-sink, non-dropped retained token to ``target_tier_id``.
    """
    def _hook(pipeline):
        n_changed = 0
        for ts in pipeline._retained_tokens:
            if ts.protected:
                continue
            if ts.new_tier_id == 0:
                continue  # dropped; do not resurrect
            if ts.new_tier_id != target_tier_id:
                ts.new_tier_id = int(target_tier_id)
                n_changed += 1
        return n_changed
    _hook.__name__ = f"pin_tier_b{target_tier_id}"
    return _hook


def _layer_modal_tier_hook() -> Callable:
    """
    Hook (UniformHead)
    """
    def _hook(pipeline):
        by_layer: Dict[int, Counter] = defaultdict(Counter)
        for ts in pipeline._retained_tokens:
            if ts.protected or ts.new_tier_id == 0:
                continue
            by_layer[ts.layer][ts.new_tier_id] += 1
        layer_tier: Dict[int, int] = {}
        for li, ctr in by_layer.items():
            if ctr:
                layer_tier[li] = ctr.most_common(1)[0][0]
        n_changed = 0
        for ts in pipeline._retained_tokens:
            if ts.protected or ts.new_tier_id == 0:
                continue
            target = layer_tier.get(ts.layer)
            if target is not None and ts.new_tier_id != target:
                ts.new_tier_id = int(target)
                n_changed += 1
        return n_changed
    _hook.__name__ = "layer_modal_tier"
    return _hook



ABLATION_MODES = {
    "keepdrop":     "A2: KeepDrop (binary keep/drop, flat lambdas)",
    "quant_only":   "A2: Quant-only (retain all, pin to b3)",
    "decoupled":    "A2: Decoupled (flat-lambda keep/drop, then pin to b3)",
    "uniform_head": "A3: UniformHead (layer-modal tier across heads)",
    "noseg":        "A4: NoSeg (flat segment weights)",
    "nogate":       "A5: NoGate (GAMMA=0, COOLDOWN_STEPS=0)",
    # sphkv_angle is handled as a "negative control" but installs like an
    # ablation (needs pin hook).  experiment_runner dispatches it here too.
    "sphkv_angle":  "A2-B: AngleOnly (ADA kernel, uniform tier)",
}


_RETAIN_ALL_MODES = {"quant_only"}

_ANGLE_ONLY_MODES = {"sphkv_angle"}


def apply_ablation_mode(mode: str,
                        pipeline=None) -> Dict[str, Any]:
    saved: Dict[str, Any] = {"mode": mode}

    if pipeline is not None:
        saved["pre_pages_hook"] = pipeline._pre_pages_hook

    if mode == "keepdrop":
        saved.update(_set_lambdas(
            lt={1: 0.30, 2: 0.30, 3: 0.30},
            lr={1: 0.01, 2: 0.01, 3: 0.01},
        ))

    elif mode == "decoupled":
        saved.update(_set_lambdas(
            lt={1: 0.30, 2: 0.30, 3: 0.30},
            lr={1: 0.01, 2: 0.01, 3: 0.01},
        ))
        if pipeline is not None:
            pipeline._pre_pages_hook = _pin_tier_hook(target_tier_id=3)

    elif mode == "quant_only":
        saved["BITS_PER_TOKEN_cfg"] = _cfg.BITS_PER_TOKEN
        if _cfg.BITS_PER_TOKEN < 500.0:
            _cfg.BITS_PER_TOKEN = 9999.0
        if pipeline is not None:
            pipeline._pre_pages_hook = _pin_tier_hook(target_tier_id=3)

    elif mode == "uniform_head":
        if pipeline is not None:
            pipeline._pre_pages_hook = _layer_modal_tier_hook()

    elif mode == "noseg":
        saved["SEGMENT_WEIGHTS"] = _snapshot_dict(_dp.SEGMENT_WEIGHTS)
        for k in list(_dp.SEGMENT_WEIGHTS.keys()):
            _dp.SEGMENT_WEIGHTS[k] = 1.0

    elif mode == "nogate":
        saved["GAMMA_cfg"]           = _cfg.GAMMA
        saved["GAMMA_dp"]            = _dp.GAMMA
        saved["COOLDOWN_STEPS_cfg"]  = _cfg.COOLDOWN_STEPS
        saved["COOLDOWN_STEPS_ts"]   = _ts.COOLDOWN_STEPS
        _cfg.GAMMA          = 0.0
        _dp.GAMMA           = 0.0
        _cfg.COOLDOWN_STEPS = 0
        _ts.COOLDOWN_STEPS  = 0

    elif mode == "sphkv_angle":
        if pipeline is not None:
            pipeline._pre_pages_hook = _pin_tier_hook(target_tier_id=3)

    else:
        raise ValueError(f"Unknown ablation mode: {mode}. "
                         f"Known: {list(ABLATION_MODES.keys())}")

    return saved


def restore_ablation_mode(saved: Dict[str, Any],
                          pipeline=None) -> None:
    if not saved:
        return

    _restore_lambdas(saved)

    if "SEGMENT_WEIGHTS" in saved:
        _dp.SEGMENT_WEIGHTS.clear()
        _dp.SEGMENT_WEIGHTS.update(saved["SEGMENT_WEIGHTS"])

    if "BITS_PER_TOKEN_cfg" in saved:
        _cfg.BITS_PER_TOKEN = saved["BITS_PER_TOKEN_cfg"]

    if "GAMMA_cfg" in saved:
        _cfg.GAMMA = saved["GAMMA_cfg"]
    if "GAMMA_dp" in saved:
        _dp.GAMMA = saved["GAMMA_dp"]
    if "COOLDOWN_STEPS_cfg" in saved:
        _cfg.COOLDOWN_STEPS = saved["COOLDOWN_STEPS_cfg"]
    if "COOLDOWN_STEPS_ts" in saved:
        _ts.COOLDOWN_STEPS = saved["COOLDOWN_STEPS_ts"]

    if pipeline is not None and "pre_pages_hook" in saved:
        pipeline._pre_pages_hook = saved["pre_pages_hook"]


def post_allocation_ablation(mode: str, pipeline) -> None:
    """DEPRECATED: semantics moved to ``pipeline._pre_pages_hook``."""
    return


__all__ = [
    "ABLATION_MODES",
    "apply_ablation_mode",
    "restore_ablation_mode",
    "post_allocation_ablation",
]
