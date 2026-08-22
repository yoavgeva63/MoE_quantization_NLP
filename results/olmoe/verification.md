# olmoe - offline verification

Machine-checked from the saved artifacts, with no GPU and no model download. Regenerate with `python scripts/verify_offline.py --results-dir results/olmoe`.

**All 12 runs pass.**

## Router weights, every channel of every layer

Read from each run's stored `router_weights`, which are the dequantized weights that run actually used. `levels` is the largest number of distinct values found in any single output channel; under per-axis symmetric INT-*N* quantization a channel cannot exceed `2**N`.

| Run | Layers | Channels inspected | Max levels in a channel | Allowed | Layers bit-identical to gold | Verdict |
|-----|--------|--------------------|-------------------------|---------|------------------------------|---------|
| gold BF16 | 16 | 1,024 | 1,161 | n/a (protected) | **16/16** | PASS |
| mixed INT8 | 16 | 1,024 | 1,161 | n/a (protected) | **16/16** | PASS |
| placebo INT8 | 16 | 1,024 | 60 | 256 | **0/16** | PASS |
| uniform INT8 | 16 | 1,024 | 60 | 256 | **0/16** | PASS |
| attention INT4 | 16 | 1,024 | 16 | 16 | **0/16** | PASS |
| mixed INT4 | 16 | 1,024 | 1,161 | n/a (protected) | **16/16** | PASS |
| placebo INT4 | 16 | 1,024 | 16 | 16 | **0/16** | PASS |
| uniform INT4 | 16 | 1,024 | 16 | 16 | **0/16** | PASS |
| attention INT3 | 16 | 1,024 | 8 | 8 | **0/16** | PASS |
| mixed INT3 | 16 | 1,024 | 1,161 | n/a (protected) | **16/16** | PASS |
| placebo INT3 | 16 | 1,024 | 8 | 8 | **0/16** | PASS |
| uniform INT3 | 16 | 1,024 | 8 | 8 | **0/16** | PASS |

## What this establishes

- **gold BF16** (identical to gold in every layer): bit-identical to gold in 16/16 layers
- **mixed INT8** (identical to gold in every layer (routers protected)): bit-identical to gold in 16/16 layers
- **placebo INT8** (quantized in every layer, at most 2**bits levels per channel): quantized in all 16 layers, at most 60 of 256 levels per channel
- **uniform INT8** (quantized in every layer, at most 2**bits levels per channel): quantized in all 16 layers, at most 60 of 256 levels per channel
- **attention INT4** (quantized in every layer, at most 2**bits levels per channel): quantized in all 16 layers, at most 16 of 16 levels per channel
- **mixed INT4** (identical to gold in every layer (routers protected)): bit-identical to gold in 16/16 layers
- **placebo INT4** (quantized in every layer, at most 2**bits levels per channel): quantized in all 16 layers, at most 16 of 16 levels per channel
- **uniform INT4** (quantized in every layer, at most 2**bits levels per channel): quantized in all 16 layers, at most 16 of 16 levels per channel
- **attention INT3** (quantized in every layer, at most 2**bits levels per channel): quantized in all 16 layers, at most 8 of 8 levels per channel
- **mixed INT3** (identical to gold in every layer (routers protected)): bit-identical to gold in 16/16 layers
- **placebo INT3** (quantized in every layer, at most 2**bits levels per channel): quantized in all 16 layers, at most 8 of 8 levels per channel
- **uniform INT3** (quantized in every layer, at most 2**bits levels per channel): quantized in all 16 layers, at most 8 of 8 levels per channel

Every `mixed` run's router weights are bit-identical to gold in every layer (4 runs checked), and every `uniform` and `placebo` run's routers sit at or below their INT-*N* level ceiling in every channel with no layer matching gold (8 runs checked). That is direct evidence for the single-variable claim - `mixed` and `uniform` differ in the routers and nothing else - computed from the artifacts rather than from the quantizer's own bookkeeping.

The INT8 rows show fewer than 256 levels, which is expected rather than a shortfall: with a per-channel scale of `max|w| / 127` and roughly Gaussian router weights, the 2048 values in a channel do not reach the extreme codes.

## What the placebo control protected

The chosen FQNs are not recorded in any `metrics.json` - only `num_protected_modules` - so they are re-derived here from a meta-device skeleton. Deterministic given `placebo_seed`, and reproducible with `python scripts/verify_offline.py --results-dir results/<model>`.

- `placebo_seed` = 0
- Protected module: `model.layers.14.mlp.experts.10.down_proj`
- Modules protected: **1**, against 16 routers
- Parameters protected: 2,097,152 against a router budget of 2,097,152 (**1.000000x**)
- Candidate pool: 3,072 modules, with distinct sizes 2,097,152

The selection is a **single draw**, not the greedy accumulation the code's loop suggests: every eligible candidate is already within tolerance of the whole router budget on its own, so the first module drawn satisfies the stopping condition and the tolerance band is never exercised.

### Activation exposure

Parameter count is the only dimension on which this control is matched. A router is read by every token in every layer; an expert projection is read only by the tokens routed to its own expert, in its own layer. Measured from the gold router logits:

| Protected module | Layer | Expert | Tokens on its path | Token x layer exposure gap vs routers |
|------------------|-------|--------|--------------------|---------------------------------------|
| `model.layers.14.mlp.experts.10.down_proj` | 14 | 10 | 11.43% of 32,768 | **140x** |

The routers are read at 100% of tokens in all 16 layers. Counting token x layer forward events, they are exposed by the factor in the last column more than the placebo's protected parameters. This is the honest limitation of the control: it rules out "any high-precision parameters help", and does not address "any high-precision island on the every-token, every-layer main path helps".
