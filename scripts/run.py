#!/usr/bin/env python3
"""Run Part 1 configurations.

Gold always runs first when included, because every other policy is scored against the
artifacts it writes.

    python scripts/run.py --config configs/olmoe.yaml --policies gold uniform mixed --bits 4
    python scripts/run.py --config configs/olmoe.yaml --policies uniform mixed --bits 8 4 3
"""

from __future__ import annotations

import argparse
import sys
import traceback

from moequant.config import ExperimentConfig
from moequant.quantize import POLICIES, SUPPORTED_BITS
from moequant.runner import run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--policies", nargs="+", default=["gold", "uniform", "mixed"], choices=POLICIES
    )
    parser.add_argument(
        "--bits",
        nargs="+",
        type=int,
        default=[4],
        choices=SUPPORTED_BITS,
        help="Bit-widths to sweep. Ignored for the gold policy.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--routing-sequences", type=int, default=None)
    parser.add_argument("--max-ppl-windows", type=int, default=None)
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue the sweep after a failure instead of stopping",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    overrides = {
        "device": args.device,
        "results_dir": args.results_dir,
        "cache_dir": args.cache_dir,
        "seed": args.seed,
        "routing_sequences": args.routing_sequences,
        "max_ppl_windows": args.max_ppl_windows,
    }

    # Gold is bit-width independent and must exist before anything is compared to it.
    jobs: list[tuple[str, int | None]] = []
    if "gold" in args.policies:
        jobs.append(("gold", None))
    for policy in args.policies:
        if policy == "gold":
            continue
        for bits in args.bits:
            jobs.append((policy, bits))

    failures: list[str] = []
    for policy, bits in jobs:
        cfg = ExperimentConfig.from_yaml(args.config, policy=policy, bits=bits, **overrides)
        label = f"{cfg.model_key}/{cfg.run_name}"
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
        try:
            run(cfg, progress=not args.quiet)
        except Exception as exc:  # noqa: BLE001 - a sweep should report, not vanish
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            if not args.keep_going:
                print(f"\nStopping after failure in {label}. Use --keep-going to continue.")
                return 1

    if failures:
        print(f"\n{len(failures)} of {len(jobs)} runs failed:")
        for line in failures:
            print(f"  - {line}")
        return 1

    print(f"\nAll {len(jobs)} runs completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
