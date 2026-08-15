"""Run one (model, policy, bit-width) configuration end to end.

The gold run is special: it writes the reference artifacts - router logits, router
weights, router inputs, and a compressed gold output distribution - that every later run
is scored against. Candidate runs refuse to start if those artifacts are missing, because
comparing against a stale or absent gold is how silent nonsense gets into a results table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from . import verify
from .capture import capture_routing
from .config import ExperimentConfig, environment
from .data import describe, load_token_stream, ppl_batches, routing_batches
from .evaluate import OutputReference, evaluate_lm
from .metrics import compare_routing
from .quantize import build_policy, sample_placebo_modules, to_torchao_config
from .registry import get_spec, parameter_census, resolve_topology

# Gold artifacts are split by consumer rather than written as one bundle. A candidate run
# needs the router logits and the small output reference; it never reads the captured
# router inputs, which are the largest tensor here and are only used by the Part 2
# offline attribution. Keeping them in a separate file stops every candidate run from
# paging in hundreds of megabytes it will not touch.
ARTIFACTS = "artifacts.pt"  # router logits, weights, groups, topology, fingerprint
REFERENCE = "output_reference.pt"  # compressed gold output distribution
INPUTS = "router_inputs.pt"  # Part 2 only
METRICS = "metrics.json"

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable. These models need a GPU; on the TAU cluster "
            "submit through Slurm rather than running on the login node."
        )
    return requested


def load_model(cfg: ExperimentConfig, spec, quant_config):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_id, trust_remote_code=spec.trust_remote_code, cache_dir=cfg.cache_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id,
        torch_dtype=DTYPES[cfg.dtype],
        device_map="auto" if cfg.device == "cuda" else cfg.device,
        quantization_config=quant_config,
        trust_remote_code=spec.trust_remote_code,
        cache_dir=cfg.cache_dir,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def _meta_skeleton(cfg: ExperimentConfig, spec):
    """Instantiate the architecture on the meta device, so no weights are materialised.

    Downloads the config only. Used to learn module names before the real load, which the
    quantization policy needs in order to expand its skip patterns into exact FQNs.
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    hf_config = AutoConfig.from_pretrained(
        spec.model_id, trust_remote_code=spec.trust_remote_code, cache_dir=cfg.cache_dir
    )
    with init_empty_weights():
        return AutoModelForCausalLM.from_config(
            hf_config, trust_remote_code=spec.trust_remote_code
        )


def _check_alignment(gold: dict, routing, topology: dict, cfg: ExperimentConfig) -> None:
    """Refuse to compare a candidate against a gold run that saw different tokens.

    Downstream, the only structural check is that gold and candidate logit tensors have
    matching shapes. That is satisfied by any two runs with the same sequence count and
    length, so a changed seed, corpus or tokenizer would sail through and yield routing
    metrics that look entirely reasonable while comparing unrelated tokens.
    """
    expected = gold.get("fingerprint")
    if expected is None:
        raise ValueError(
            f"Gold artifacts at {cfg.gold_dir / ARTIFACTS} predate the alignment check. "
            "Re-run the gold policy so the token fingerprint is recorded."
        )
    if expected != routing.fingerprint:
        raise ValueError(
            f"Routing tokens do not match the gold run ({routing.fingerprint} vs "
            f"{expected}). corpus/seed/routing_sequences/routing_seq_len and the "
            "tokenizer must all be identical to the gold run."
        )

    gold_topology = gold.get("topology", {})
    for key in ("num_experts", "top_k", "num_router_layers"):
        if key in gold_topology and gold_topology[key] != topology[key]:
            raise ValueError(
                f"Topology differs from gold: {key} is {topology[key]} here but "
                f"{gold_topology[key]} in gold."
            )


def run(cfg: ExperimentConfig, progress: bool = True) -> dict:
    device = _resolve_device(cfg.device)
    spec = get_spec(cfg.model_key)
    torch.manual_seed(cfg.seed)

    is_gold = cfg.policy == "gold"
    if not is_gold and not (cfg.gold_dir / ARTIFACTS).exists():
        raise FileNotFoundError(
            f"Gold artifacts missing at {cfg.gold_dir / ARTIFACTS}. "
            f"Run the gold policy for {cfg.model_key} first - every metric is relative to it."
        )

    # A quantized policy targets modules by exact FQN, so the names have to be known
    # before the real load. One meta-device skeleton serves both that and the placebo draw.
    placebo: tuple[str, ...] = ()
    module_fqns: tuple[str, ...] = ()
    if not is_gold:
        skeleton = _meta_skeleton(cfg, spec)
        module_fqns = tuple(fqn for fqn, _ in skeleton.named_modules() if fqn)
        if cfg.policy == "placebo":
            placebo = sample_placebo_modules(skeleton, spec, seed=cfg.placebo_seed)
        del skeleton

    policy = build_policy(cfg.policy, spec, cfg.bits, placebo_modules=placebo)
    quant_config = to_torchao_config(policy, module_fqns)

    print(f"[{cfg.model_key}/{cfg.run_name}] {policy.describe()}")
    model, tokenizer = load_model(cfg, spec, quant_config)

    # -- correctness gates, before anything expensive is computed --------------------
    audit = verify.check(model, spec, policy, strict=True)
    bit_check = verify.check_bit_width(model, policy)
    if not bit_check.get("passed", True):
        raise verify.VerificationError(
            f"Bit-width check failed: {bit_check['violations']}. A weight shows more "
            f"distinct values per channel than INT{policy.bits} allows."
        )
    topology = resolve_topology(model, spec)
    census = parameter_census(model, spec)
    print(
        f"  routers: {topology['num_router_layers']} layers, "
        f"{topology['num_experts']} experts, top-{topology['top_k']}, "
        f"{census['router_fraction'] * 100:.4f}% of params"
    )

    # -- data ------------------------------------------------------------------------
    tokens = load_token_stream(tokenizer, cfg.corpus, cache_dir=cfg.cache_dir)
    ppl = ppl_batches(tokens, cfg.ppl_seq_len, cfg.ppl_stride, cfg.max_ppl_windows)
    routing = routing_batches(tokens, cfg.routing_sequences, cfg.routing_seq_len, cfg.seed)

    # Check against gold before the forward passes, not after: a mismatched corpus or
    # seed should cost seconds, not a full capture and perplexity sweep.
    reference = None
    gold_artifacts = None
    if not is_gold:
        gold_artifacts = torch.load(
            cfg.gold_dir / ARTIFACTS, map_location="cpu", weights_only=False
        )
        _check_alignment(gold_artifacts, routing, topology, cfg)
        ref_path = cfg.gold_dir / REFERENCE
        if ref_path.exists():
            reference = OutputReference.from_dict(
                torch.load(ref_path, map_location="cpu", weights_only=False)
            )

    # -- routing capture ---------------------------------------------------------------
    captured = capture_routing(
        model,
        spec,
        routing.batches,
        device=device,
        capture_inputs=cfg.capture_inputs,
        input_stride=cfg.input_stride,
        progress=progress,
    )

    # -- language modelling ------------------------------------------------------------
    lm_metrics, built_reference = evaluate_lm(
        model,
        ppl,
        device=device,
        reference=reference,
        top_m=cfg.top_m,
        position_stride=cfg.position_stride,
        build_reference=is_gold,
        progress=progress,
    )

    # -- comparison against gold -------------------------------------------------------
    result: dict = {
        "config": cfg.to_dict(),
        "environment": environment(),
        "model_id": spec.model_id,
        "policy": policy.name,
        "bits": policy.bits,
        "audit": audit,
        "bit_width_check": bit_check,
        "topology": topology,
        "parameter_census": census,
        "dataset": describe(tokens, routing, cfg.corpus),
        "lm": lm_metrics,
    }

    if is_gold:
        cfg.gold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "logits": captured["logits"],
                "router_weights": captured["router_weights"],
                "groups": routing.groups,
                "topology": topology,
                "fingerprint": routing.fingerprint,
            },
            cfg.gold_dir / ARTIFACTS,
        )
        if built_reference is not None:
            torch.save(built_reference.to_dict(), cfg.gold_dir / REFERENCE)
        if captured.get("inputs"):
            torch.save(
                {
                    "inputs": captured["inputs"],
                    "input_stride": captured.get("input_stride"),
                },
                cfg.gold_dir / INPUTS,
            )
        # Self-comparison proves the metric code reports zero difference for identical
        # inputs; if this is not exactly zero, something upstream is nondeterministic.
        result["routing"] = compare_routing(
            captured["logits"], captured["logits"], topology["top_k"],
            groups=routing.groups, n_boot=cfg.n_boot, seed=cfg.seed,
        )
        result["self_check"] = {
            "kl": result["routing"]["pooled"]["kl"]["mean"],
            "top1_error": result["routing"]["pooled"]["top1_error"]["mean"],
            "passed": (
                abs(result["routing"]["pooled"]["kl"]["mean"]) < 1e-9
                and abs(result["routing"]["pooled"]["top1_error"]["mean"]) < 1e-9
            ),
        }
    else:
        result["routing"] = compare_routing(
            gold_artifacts["logits"],
            captured["logits"],
            topology["top_k"],
            groups=np.asarray(gold_artifacts["groups"]),
            n_boot=cfg.n_boot,
            seed=cfg.seed,
        )
        if cfg.capture_inputs and captured.get("inputs"):
            cfg.run_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "inputs": captured["inputs"],
                    "router_weights": captured["router_weights"],
                    "input_stride": captured.get("input_stride"),
                },
                cfg.run_dir / INPUTS,
            )

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.run_dir / METRICS
    out_path.write_text(json.dumps(result, indent=2, default=_json_default))
    print(f"  wrote {out_path}")
    _print_summary(result)
    return result


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _print_summary(result: dict) -> None:
    lm = result["lm"]
    line = f"  PPL {lm['perplexity']:.4f}"
    if "top1_agreement" in lm:
        line += f" | top-1 agreement {lm['top1_agreement'] * 100:.2f}%"
        line += f" | output KL {lm['output_kl_topm']:.5f}"
    print(line)

    routing = result.get("routing", {}).get("pooled")
    if routing:
        print(
            f"  routing KL {routing['kl']['mean']:.6f} "
            f"[{routing['kl']['ci_low']:.6f}, {routing['kl']['ci_high']:.6f}] | "
            f"top-1 flip {routing['top1_error']['mean'] * 100:.2f}% | "
            f"jaccard {routing['jaccard_distance']['mean']:.4f}"
        )
