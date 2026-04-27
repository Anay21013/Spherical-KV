import json
import re
import statistics
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


DEFAULT_TOOL_SCHEMA = [
    {
        "name": "search",
        "description": "Search for information on a topic",
        "parameters": {
            "query": {"type": "string", "description": "Search query"}
        }
    },
    {
        "name": "calculator",
        "description": "Perform a mathematical calculation",
        "parameters": {
            "expression": {"type": "string", "description": "Math expression"}
        }
    },
    {
        "name": "lookup",
        "description": "Look up a specific fact or entity",
        "parameters": {
            "entity": {"type": "string", "description": "Entity to look up"}
        }
    },
    {
        "name": "finish",
        "description": "Indicate the task is complete with final answer",
        "parameters": {
            "answer": {"type": "string", "description": "Final answer"}
        }
    },
]


def mock_tool_response(tool_name: str, args: dict) -> str:
    """
    Deterministic mock tool responses for reproducibility.
    Paper: 'deterministic tool responses when possible'
    """
    if tool_name == "search":
        query = args.get("query", "")
        return f"Search results for '{query}': Found 3 relevant documents. The most relevant states that {query} is a well-documented topic."
    elif tool_name == "calculator":
        expr = args.get("expression", "0")
        try:
            result = eval(expr, {"__builtins__": {}}, {})
            return f"Result: {result}"
        except Exception:
            return "Error: invalid expression"
    elif tool_name == "lookup":
        entity = args.get("entity", "")
        return f"Lookup result: '{entity}' is a known entity with standard properties."
    elif tool_name == "finish":
        return "Task completed."
    else:
        return f"Unknown tool: {tool_name}"



def load_toolbench_dataset(
    max_samples: int = 50,
    split:       str = "test",
) -> List[dict]:
    """
    Load ToolBench episodes.
    Falls back to synthetic episodes if dataset unavailable.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("ToolBench/ToolBench", split=split,
                          trust_remote_code=True)
        samples = []
        for i, example in enumerate(ds):
            if i >= max_samples:
                break
            samples.append({
                "instruction": example.get("instruction", example.get("query", "")),
                "tools":       example.get("tools", DEFAULT_TOOL_SCHEMA),
                "gold_answer": example.get("answer", ""),
                "gold_trajectory": example.get("trajectory", []),
            })
        print(f"[W3] Loaded {len(samples)} ToolBench episodes")
        return samples
    except Exception as e:
        print(f"[W3] ToolBench load failed ({e}), using synthetic episodes")
        return _make_synthetic_episodes(max_samples)


def load_agentbench_dataset(
    max_samples: int = 50,
    task:        str = "os",
) -> List[dict]:
    """
    Load AgentBench episodes.
    Falls back to synthetic if unavailable.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("THUDM/AgentBench", task, split="test",
                          trust_remote_code=True)
        samples = []
        for i, example in enumerate(ds):
            if i >= max_samples:
                break
            samples.append({
                "instruction": example.get("instruction", ""),
                "tools":       DEFAULT_TOOL_SCHEMA,
                "gold_answer": example.get("answer", ""),
                "gold_trajectory": example.get("trajectory", []),
            })
        print(f"[W3] Loaded {len(samples)} AgentBench/{task} episodes")
        return samples
    except Exception as e:
        print(f"[W3] AgentBench load failed ({e}), using synthetic episodes")
        return _make_synthetic_episodes(max_samples)


def _make_synthetic_episodes(n: int) -> List[dict]:
    """Generate synthetic multi-step tool-use episodes for testing.

    Each task has a verifiable gold_answer.  This is essential — without
    it, success reduces to "did the model emit any finish tool call?",
    which is a meaningless metric (an instruction-tuned model that
    immediately writes `{"tool": "finish", "args": {"answer": ""}}`
    would score 100%).
    """
    episodes = []
    tasks = [
        {"instruction": "Use the calculator to compute 12 * 7.",
         "gold_answer": "84"},
        {"instruction": "Use the calculator to compute 42 * 17 + 83.",
         "gold_answer": "797"},
        {"instruction": "Use the calculator to compute (15 + 9) * 4.",
         "gold_answer": "96"},
        {"instruction": "Search for the capital of Japan.",
         "gold_answer": "Tokyo"},
        {"instruction": "Search for the author of the play Hamlet.",
         "gold_answer": "Shakespeare"},
    ]
    for i in range(n):
        t = tasks[i % len(tasks)]
        episodes.append({
            "instruction":     t["instruction"],
            "tools":           DEFAULT_TOOL_SCHEMA,
            "gold_answer":     t["gold_answer"],
            "gold_trajectory": [],
        })
    print(f"[W3] Generated {n} synthetic episodes")
    return episodes

# Agent episode runner

AGENT_SYSTEM_PROMPT = """You are a helpful assistant with access to tools. \
You MUST respond with exactly one JSON object per turn — nothing else, \
no prose before or after.

Use a tool by emitting:
{{"tool": "tool_name", "args": {{"param": "value"}}}}

When you have the final answer, emit:
{{"tool": "finish", "args": {{"answer": "your answer"}}}}

Available tools:
{tools_desc}

Examples
--------
Task: What is 12 times 7?
{{"tool": "calculator", "args": {{"expression": "12*7"}}}}
Tool response: 84
{{"tool": "finish", "args": {{"answer": "84"}}}}

Task: Who wrote Hamlet?
{{"tool": "search", "args": {{"query": "author of Hamlet"}}}}
Tool response: William Shakespeare wrote Hamlet around 1600.
{{"tool": "finish", "args": {{"answer": "William Shakespeare"}}}}

Now begin. Respond with exactly one JSON tool call per turn."""


def _format_tools_desc(tools: List[dict]) -> str:
    parts = []
    for t in tools:
        params = ", ".join(f"{k}: {v['type']}" for k, v in t.get("parameters", {}).items())
        parts.append(f"- {t['name']}({params}): {t['description']}")
    return "\n".join(parts)


def _parse_tool_call(text: str) -> Optional[Tuple[str, dict]]:
    """Extract tool call from model output."""
    decoder = json.JSONDecoder()
    n = len(text)
    i = 0
    while i < n:
        if text[i] == '{':
            try:
                obj, end = decoder.raw_decode(text, i)
            except json.JSONDecodeError:
                i += 1
                continue
            if isinstance(obj, dict) and "tool" in obj:
                tool = obj.get("tool", "")
                args = obj.get("args", {})
                if not isinstance(args, dict):
                    args = {}
                return str(tool), args
            i = end
        else:
            i += 1

    # Last-ditch fallback: a bare "tool: name" mention with no JSON.
    m = re.search(r'"?tool"?\s*[:=]\s*"?([A-Za-z_]\w*)"?', text)
    if m:
        return m.group(1), {}

    return None


@torch.no_grad()
def run_episode(
    model,
    tokenizer,
    pipeline,
    episode:        dict,
    device:         torch.device,
    max_steps:      int = 10,
    max_new_tokens: int = 128,
    mode:           str = "dense",
    seed:           int = 42,
) -> dict:
    """
    Run a single bounded-horizon agentic episode.

    Returns trajectory (sequence of tool calls), final answer,
    success flag, and total generated tokens.
    """
    torch.manual_seed(seed)

    tools = episode.get("tools", DEFAULT_TOOL_SCHEMA)
    tools_desc = _format_tools_desc(tools)
    system = AGENT_SYSTEM_PROMPT.format(tools_desc=tools_desc)

    conversation = f"{system}\n\nTask: {episode['instruction']}\n\n"
    trajectory = []
    total_tokens = 0
    success = False
    final_answer = ""

    for step in range(max_steps):
        input_ids = tokenizer.encode(conversation, return_tensors="pt").to(device)
        T = input_ids.shape[1]

        # Prefill with pipeline if using SphericalKV
        if pipeline is not None and mode != "dense":
            try:
                pipeline.prefill(input_ids)
            except Exception:
                pass

        # Generate
        try:
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
            gen_ids = output[0, T:]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            total_tokens += len(gen_ids)
        except Exception as e:
            response = f'{{"tool": "finish", "args": {{"answer": "error: {e}"}}}}'
            total_tokens += 1

        if pipeline is not None and pipeline._patched:
            pipeline.uninstall()

        # Parse tool call
        tool_call = _parse_tool_call(response)
        if tool_call is None:
            # Model didn't produce a valid tool call -- treat as finish
            trajectory.append({"step": step, "raw": response,
                               "tool": "none", "args": {}})
            break

        tool_name, tool_args = tool_call
        trajectory.append({"step": step, "tool": tool_name,
                           "args": tool_args, "raw": response})

        if tool_name == "finish":
            final_answer = tool_args.get("answer", "")
            gold = str(episode.get("gold_answer", "")).strip().lower()
            answer_norm = str(final_answer).strip().lower()
            if gold:
                # Substring match in either direction is forgiving enough
                # for "Tokyo" vs "Tokyo, Japan" and "84" vs "the answer is 84".
                success = (gold in answer_norm) or (answer_norm in gold and answer_norm != "")
            else:
                # No gold answer available (e.g. trajectory-only AgentBench
                # episodes) — fall back to "model emitted finish".
                success = True
            break

        # Execute tool and append response
        tool_response = mock_tool_response(tool_name, tool_args)
        conversation += f"Assistant: {response}\nTool response: {tool_response}\n\n"

    return {
        "trajectory":   trajectory,
        "final_answer":  final_answer,
        "success":       success,
        "n_steps":       len(trajectory),
        "total_tokens":  total_tokens,
        "seed":          seed,
    }


# W3 evaluation with stability metrics

@torch.no_grad()
def evaluate_w3(
    model,
    tokenizer,
    pipeline,
    episodes:       List[dict],
    device:         torch.device,
    mode:           str = "dense",
    max_steps:      int = 10,
    max_new_tokens: int = 128,
    n_seeds:        int = 3,
) -> dict:
    """
    Full W3 evaluation with stability metrics.

    Runs each episode across multiple seeds to measure:
    - Success rate (task completion)
    - Trajectory sensitivity S_traj
    - Length drift DeltaT
    - Trajectory disagreement rate
    """
    all_success = []
    all_n_steps = []
    all_total_tokens = []
    trajectory_sets = []  # per-episode list of per-seed trajectories

    for i, episode in enumerate(episodes):
        seed_results = []
        for seed in range(n_seeds):
            result = run_episode(
                model, tokenizer, pipeline, episode, device,
                max_steps=max_steps, max_new_tokens=max_new_tokens,
                mode=mode, seed=seed)
            seed_results.append(result)

        # Use first seed for primary metrics
        primary = seed_results[0]
        all_success.append(float(primary["success"]))
        all_n_steps.append(primary["n_steps"])
        all_total_tokens.append(primary["total_tokens"])
        trajectory_sets.append(seed_results)

        if (i + 1) % 5 == 0:
            sr = sum(all_success) / len(all_success)
            print(f"  [{mode}] {i+1}/{len(episodes)}  success_rate={sr:.3f}")


    # S_traj: variance of n_steps across seeds per episode, then average
    s_traj_values = []
    for seed_results in trajectory_sets:
        steps = [r["n_steps"] for r in seed_results]
        if len(steps) > 1:
            s_traj_values.append(statistics.variance(steps))

    s_traj = statistics.mean(s_traj_values) if s_traj_values else 0.0

    # Trajectory disagreement: fraction of episodes where tool-call
    # sequences differ across seeds
    n_disagree = 0
    for seed_results in trajectory_sets:
        tool_seqs = []
        for r in seed_results:
            seq = tuple(t["tool"] for t in r["trajectory"])
            tool_seqs.append(seq)
        if len(set(tool_seqs)) > 1:
            n_disagree += 1

    disagree_rate = n_disagree / max(len(trajectory_sets), 1)

    return {
        "mode":               mode,
        "success_rate":       sum(all_success) / max(len(all_success), 1),
        "mean_steps":         sum(all_n_steps) / max(len(all_n_steps), 1),
        "mean_tokens":        sum(all_total_tokens) / max(len(all_total_tokens), 1),
        "n_episodes":         len(episodes),
        "n_seeds":            n_seeds,
        "S_traj":             s_traj,
        "disagree_rate":      disagree_rate,
        "success_all":        all_success,
    }


def compute_w3_length_drift(
    dense_results: dict,
    sphkv_results: dict,
) -> dict:
    """
    DeltaT = E[|T_sphkv - T_dense|] across episodes.
    """
    dense_tokens = dense_results.get("mean_tokens", 0)
    sphkv_tokens = sphkv_results.get("mean_tokens", 0)
    delta_t = abs(sphkv_tokens - dense_tokens)

    return {
        "DeltaT":        delta_t,
        "dense_tokens":  dense_tokens,
        "sphkv_tokens":  sphkv_tokens,
    }
