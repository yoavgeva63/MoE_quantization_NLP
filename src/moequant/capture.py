"""Forward hooks that record what each router saw and what it decided.

Two things are captured per router:

* **logits** - the routing scores, which drive every Part 1 metric.
* **inputs** - the hidden states arriving at the router. Part 1 never uses these, but
  recording them costs one hook and turns Part 2 into offline analysis instead of a
  second round of cluster jobs. They are strided down to keep memory sane.

Batches are deliberately single-sequence: with no padding there are no masked positions
to track, and gold and candidate runs see identical tokens in identical order, so
captured rows line up by construction.
"""

from __future__ import annotations

import torch
from torch import nn

from .registry import ModelSpec, find_routers, layer_index, router_weight


class RouterCapture:
    """Context manager collecting per-layer router logits (and optionally inputs).

    Usage::

        with RouterCapture(model, spec) as cap:
            for batch in batches:
                model(**batch)
        logits = cap.logits()      # {layer_index: [num_tokens, num_experts]}
    """

    def __init__(
        self,
        model: nn.Module,
        spec: ModelSpec,
        capture_inputs: bool = True,
        input_stride: int = 8,
        store_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.model = model
        self.spec = spec
        self.capture_inputs = capture_inputs
        self.input_stride = max(1, int(input_stride))
        self.store_dtype = store_dtype

        self.routers = find_routers(model, spec)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._logits: dict[str, list[torch.Tensor]] = {fqn: [] for fqn in self.routers}
        self._inputs: dict[str, list[torch.Tensor]] = {fqn: [] for fqn in self.routers}
        self._pending_input: dict[str, torch.Tensor] = {}

    # -- hook bodies ----------------------------------------------------------------

    def _make_pre_hook(self, fqn: str):
        def pre_hook(module, args):  # noqa: ANN001 - torch hook signature
            if not args:
                return
            hidden = args[0]
            if not isinstance(hidden, torch.Tensor):
                return
            flat = hidden.reshape(-1, hidden.shape[-1]).detach()
            # DeepSeek-style routers give no logits in their output, so we keep the
            # activation around to recompute them in the forward hook.
            if self.spec.router_output_kind == "recompute":
                self._pending_input[fqn] = flat
            if self.capture_inputs:
                self._inputs[fqn].append(
                    flat[:: self.input_stride].to("cpu", torch.float16, copy=True)
                )

        return pre_hook

    def _make_hook(self, fqn: str, module: nn.Module):
        def hook(mod, args, output):  # noqa: ANN001 - torch hook signature
            logits = self._extract_logits(fqn, mod, output)
            if logits is None:
                return
            flat = logits.reshape(-1, logits.shape[-1]).detach()
            self._logits[fqn].append(flat.to("cpu", self.store_dtype, copy=True))

        return hook

    def _extract_logits(self, fqn: str, module: nn.Module, output) -> torch.Tensor | None:
        if self.spec.router_output_kind == "logits":
            if isinstance(output, torch.Tensor):
                return output
            if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
                return output[0]
            raise TypeError(
                f"Router {fqn} returned {type(output).__name__}, expected a logit tensor. "
                "Set router_output_kind='recompute' for this architecture."
            )

        # "recompute": the module hides its logits, so rebuild them from input @ W.T.
        hidden = self._pending_input.pop(fqn, None)
        if hidden is None:
            return None
        weight = router_weight(module)
        return torch.nn.functional.linear(hidden.to(weight.dtype), weight)

    # -- lifecycle ------------------------------------------------------------------

    def __enter__(self) -> RouterCapture:
        for fqn, module in self.routers.items():
            self._handles.append(module.register_forward_pre_hook(self._make_pre_hook(fqn)))
            self._handles.append(module.register_forward_hook(self._make_hook(fqn, module)))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._pending_input.clear()

    # -- results --------------------------------------------------------------------

    def logits(self) -> dict[int, torch.Tensor]:
        """Concatenated router logits keyed by transformer layer index."""
        out: dict[int, torch.Tensor] = {}
        for fqn, chunks in self._logits.items():
            if not chunks:
                continue
            out[layer_index(fqn)] = torch.cat(chunks, dim=0)
        if not out:
            raise RuntimeError(
                "No router logits captured. Did the forward pass actually run, and does "
                "the registry pattern match the real module names?"
            )
        return out

    def inputs(self) -> dict[int, torch.Tensor]:
        """Concatenated (strided) router input activations keyed by layer index."""
        out: dict[int, torch.Tensor] = {}
        for fqn, chunks in self._inputs.items():
            if chunks:
                out[layer_index(fqn)] = torch.cat(chunks, dim=0)
        return out

    def router_weights(self) -> dict[int, torch.Tensor]:
        """Dense copies of each router's weight matrix, for the Part 2 decomposition."""
        from .verify import dequantize

        return {
            layer_index(fqn): dequantize(router_weight(module)).cpu()
            for fqn, module in self.routers.items()
        }


@torch.no_grad()
def capture_routing(
    model: nn.Module,
    spec: ModelSpec,
    batches: list[dict[str, torch.Tensor]],
    device: torch.device | str,
    capture_inputs: bool = True,
    input_stride: int = 8,
    progress: bool = True,
) -> dict:
    """Run the model over `batches` and return router logits, inputs, and weights."""
    from tqdm.auto import tqdm

    model.eval()
    iterator = tqdm(batches, desc="routing", disable=not progress)

    with RouterCapture(model, spec, capture_inputs, input_stride) as cap:
        for batch in iterator:
            inputs = {k: v.to(device) for k, v in batch.items()}
            model(**inputs)

        result = {
            "logits": cap.logits(),
            "router_weights": cap.router_weights(),
        }
        if capture_inputs:
            result["inputs"] = cap.inputs()
            result["input_stride"] = input_stride
    return result
