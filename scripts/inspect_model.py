#!/usr/bin/env python3
"""Architecture discovery: confirm the registry matches the real checkpoint.

Run this before any experiment. It answers the questions that would otherwise be
assumptions: where the routers actually live, how many experts they choose between, what
fraction of the model they are, and - importantly - whether the experts are individual
nn.Linear modules or fused 3D parameter stacks, because the latter changes what a
quantization walker can see.

Loads on the meta device by default, so it needs no GPU and downloads only the config.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from moequant.quantize import build_policy, quantizable_modules
from moequant.registry import (
    MODEL_SPECS,
    find_routers,
    get_spec,
    layer_index,
    parameter_census,
    router_weight,
)


def build_skeleton(spec, cache_dir: str | None):
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        spec.model_id, trust_remote_code=spec.trust_remote_code, cache_dir=cache_dir
    )
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=spec.trust_remote_code
        )
    return model, config


def expert_layout(model: nn.Module) -> dict:
    """Detect whether experts are separate Linears or a fused parameter stack.

    transformers 5.x fuses MoE experts into 3D `nn.Parameter` tensors. Quantization
    walkers that only swap `nn.Linear` skip those silently, leaving the experts in full
    precision while still reporting a successful quantized load.
    """
    fused: list[str] = []
    linear_experts = 0
    for fqn, module in model.named_modules():
        if "expert" not in fqn.lower():
            continue
        if isinstance(module, nn.Linear):
            linear_experts += 1
        for pname, param in module.named_parameters(recurse=False):
            if param.ndim == 3:
                fused.append(f"{fqn}.{pname} {tuple(param.shape)}")

    return {
        "linear_expert_modules": linear_experts,
        "fused_3d_parameters": fused,
        "layout": "fused-3d" if fused else "individual-linear",
        "warning": (
            "Experts are fused 3D parameters. An nn.Linear-based quantizer will skip "
            "them entirely. Pin transformers to 4.x or target parameters by FQN."
            if fused
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_key", choices=sorted(MODEL_SPECS))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--out", default=None, help="Write the report as JSON here")
    parser.add_argument(
        "--show-tree", action="store_true", help="Print every module name (very long)"
    )
    args = parser.parse_args()

    spec = get_spec(args.model_key)
    print(f"Model: {spec.model_id}")
    print(f"Notes: {spec.notes}\n")

    model, config = build_skeleton(spec, args.cache_dir)

    if args.show_tree:
        for fqn, _ in model.named_modules():
            print(" ", fqn)
        print()

    routers = find_routers(model, spec)
    census = parameter_census(model, spec)
    layout = expert_layout(model)

    shapes = {tuple(router_weight(m).shape) for m in routers.values()}
    print(f"Routers matched by {spec.router_pattern!r}: {len(routers)}")
    print(f"  layers      : {sorted(layer_index(f) for f in routers)}")
    print(f"  weight shape: {shapes}")
    print(f"  module type : {sorted({type(m).__name__ for m in routers.values()})}")
    print(f"  output kind : {spec.router_output_kind}")

    print("\nParameter census")
    print(f"  total     : {census['total_params']:,}")
    print(f"  routers   : {census['router_params']:,} ({census['router_fraction'] * 100:.4f}%)")
    print(
        f"  protected : {census['protected_params']:,} "
        f"({census['protected_fraction'] * 100:.4f}%)  <- what `mixed` keeps in high precision"
    )

    print("\nExpert layout")
    print(f"  {layout['layout']} ({layout['linear_expert_modules']} Linear expert modules)")
    if layout["warning"]:
        print(f"  WARNING: {layout['warning']}")

    print("\nPolicy preview (INT4)")
    quantizable = quantizable_modules(model, spec)
    for name in ("uniform", "mixed"):
        policy = build_policy(name, spec, 4)
        import re as _re

        skips = [_re.compile(p) for p in policy.skip_patterns]
        would_quantize = [f for f, _ in quantizable if not any(s.fullmatch(f) for s in skips)]
        routers_hit = sum(1 for f in would_quantize if spec.is_router(f))
        print(
            f"  {name:8s}: {len(would_quantize)} modules quantized, "
            f"{routers_hit}/{len(routers)} routers included"
        )

    report = {
        "model_id": spec.model_id,
        "num_routers": len(routers),
        "router_fqns": list(routers),
        "router_shapes": [list(s) for s in shapes],
        "router_module_types": sorted({type(m).__name__ for m in routers.values()}),
        "parameter_census": census,
        "expert_layout": layout,
        "config": {
            k: v
            for k, v in vars(config).items()
            if isinstance(v, (int, float, str, bool, type(None)))
        },
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
