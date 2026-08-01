"""Experiment configuration and run-environment capture."""

from __future__ import annotations

import dataclasses
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ExperimentConfig:
    """One (model, policy, bit-width) run."""

    model_key: str
    policy: str = "gold"
    bits: int | None = None

    dtype: str = "bfloat16"
    device: str = "cuda"

    corpus: str = "wikitext2"
    ppl_seq_len: int = 1024
    ppl_stride: int | None = None
    max_ppl_windows: int | None = None

    routing_sequences: int = 128
    routing_seq_len: int = 256

    capture_inputs: bool = True
    input_stride: int = 8

    top_m: int = 64
    position_stride: int = 16

    seed: int = 42
    n_boot: int = 200

    results_dir: str = "results"
    cache_dir: str | None = None
    placebo_seed: int = 0

    extra: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> ExperimentConfig:
        payload = yaml.safe_load(Path(path).read_text()) or {}
        known = {f.name for f in dataclasses.fields(cls)}
        extra = {k: v for k, v in payload.items() if k not in known}
        payload = {k: v for k, v in payload.items() if k in known}
        payload.update({k: v for k, v in overrides.items() if v is not None})
        cfg = cls(**payload)
        cfg.extra.update(extra)
        return cfg

    @property
    def run_name(self) -> str:
        if self.policy == "gold":
            return "gold"
        return f"{self.policy}_int{self.bits}"

    @property
    def run_dir(self) -> Path:
        return Path(self.results_dir) / self.model_key / self.run_name

    @property
    def gold_dir(self) -> Path:
        return Path(self.results_dir) / self.model_key / "gold"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - git may be absent; not worth failing a run over
        return None


def environment() -> dict:
    """Everything needed to reproduce a run, recorded alongside its results."""
    info: dict = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
            info["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        pass
    for module in ("transformers", "torchao", "datasets", "bitsandbytes"):
        try:
            info[module] = __import__(module).__version__
        except Exception:  # noqa: BLE001 - absence is informative, not fatal
            info[module] = None
    return info
