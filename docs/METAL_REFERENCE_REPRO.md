# Independent V4.1 reference on a Mac

`tests/repro/v41_reference.py` runs the pinned official 1017 Python forward on CPU
using the same GGUF, Engram and optional restore/posttrain files as the engine.
It produces all-position, full-vocabulary logits in the engine's `<i32 S,V><f32>`
format. No Metal/CUDA engine kernels are called by the reference.

This is an implementation comparison using identical quantized weights. It does
not measure quantization quality against the original 510 GB FP checkpoint, nor
does a completed run automatically satisfy the maintainer's numerical merge gate.

## Setup

Use an isolated Python 3.14 environment. The fetcher downloads about 6.5 MB of
official code/config/tokenizer files, pinned by revision and SHA-256; it never
downloads weights or replaces an existing mismatched file. Paths below should
point outside the source tree for outputs and downloaded files.

```sh
uv venv --python 3.14 "$REF_ENV"
uv pip install --python "$REF_ENV/bin/python" -r tests/repro/v41_reference_requirements.txt
python3 tests/repro/v41_reference_fetch.py "$REF_SOURCE"
"$REF_ENV/bin/python" tests/repro/v41_reference_test.py --gguf "$MODEL_PATH"
```

The [official source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170)
is pinned at `1bae0e1088993b9b82de08f59de5be5a34e42170`. File hashes live in
`tests/repro/v41_reference_sources.json` and are checked again before execution.

## Run and compare

Run one model process at a time. Keep the macOS watchdog from PR #6 around both
reference and engine commands; use 110000 MiB for native full-model Metal runs.
A three-layer/eight-token smoke run is available with `--layers 3 --ntok 8`.
Its output is explicitly diagnostic and cannot be used as a full-model result.

```sh
"$REF_ENV/bin/python" tests/repro/v41_reference.py \
  --reference "$REF_SOURCE" --gguf "$MODEL_PATH" --engram-dir "$ENGRAM_PATH" \
  --sidecar "$ZCHAIN_PATH" --sidecar "$POSTTRAIN_PATH" \
  --ids tests/repro/metal_score_ids.txt --ntok 128 --out "$REF_OUT"

./ds4 --metal -m "$MODEL_PATH" --engram-dir "$ENGRAM_PATH" \
  --zchain "$ZCHAIN_PATH" --posttrain "$POSTTRAIN_PATH" --vision-gguf off \
  --mem-budget-mb 110000 --score-ids tests/repro/metal_score_ids.txt \
  --v41-chunk 1 --score-out "$ENGINE_OUT"

make -C gguf-tools anchor_metrics
gguf-tools/bench/anchor_metrics --ref-raw "$REF_OUT" \
  --ids tests/repro/metal_score_ids.txt --student "$ENGINE_OUT"
```

The default preserves the official bf16 embedding dtype. Also run a separate
output with `--embedding-dtype exact`: the native Q4_K lookup retains expanded
fp32 values (`metal/v41_dense.metal`), as does the existing teacher with
`--exact-weights`. This changes the initial residual/normalization dtype, so keep
the two references labeled and compare all native revisions against both. Do not
select whichever reference happens to favor a candidate.

Repeat native scoring at chunk 8, and run the baseline with exactly the same
files and IDs. Report NLL, KL, distribution overlap and same-top-token rate for
each revision. Use the same explicit sidecar list on both sides; omitting it
selects a different model. The legacy V4 `caliper_ref.sh` and V4 logprob fixtures
are not valid substitutes for this 40-layer V4.1 comparison.

## Scope of the CPU adaptation

- Official Transformer, Block, Attention, Compressor, Indexer, MoE and Engram
  forwards are retained. Vision and the unused speculative draft towers are not
  built for teacher-forced text scoring.
- Linear and sparse attention use the repository teacher's independent PyTorch
  reference math. CUDA-only FP8/FP4 activation quantization and Sinkhorn use its
  scalar-formula PyTorch references. Unsupported kernel calls fail explicitly.
- Dense Q4_K uses the existing NumPy reference decoder. DQV3 shared-codebook
  experts use NumPy unpacking, checked against scalar bit addressing on real
  12/13-bit payloads. No engine dequantizer supplies forward values.
- Quantized linear weights stay expanded in fp32. The token embedding retains
  its official bf16 dtype by default (or exact fp32 when requested), and `wo_a`
  retains bf16 for its direct einsum (the existing teacher's `--exact-weights`
  exception). Subsequent quantized linear outputs retain bf16 rounding; this
  is not a claim of bitwise identity with CUDA or Metal reduction order.
- Official Engram hash constants are regenerated and compared exactly with the
  GGUF. Large tables are read by row from the existing safetensors shards.
- Restore gains/router bias and low-rank adapters are explicit. Unknown sidecar
  formats fail; routed low-rank adapters are currently unsupported.
- Parameters are loaded per layer/expert and released after use. The read-only
  model mapping is closed and reopened after each layer to bound RSS. The JSON
  beside the logits records configuration, versions, source hashes, timing and NLL.

The fixed 128-ID fixture is a short regression probe. It covers mixed compressor
ratios but not long-context top-k selection or representative model quality.

## Local result, 2026-10-31

M5 Max, 128 GiB, macOS 26.6.2, Python 3.14.7, six CPU threads. The full 40-layer,
128-token bf16-embedding reference completed in 215 s, with 8.17 GiB peak physical
footprint and 9.65 GiB peak RSS, normal memory pressure and no swap growth. It generated
128 × 129280 finite logits. The decoder/adapter checks pass, including 135 exact
scalar/vector comparisons across real 12/13-bit expert payloads. Reference versus
itself gives zero KL, unit overlap and 100% top-token agreement.

Same model on both sides: `DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative.gguf`,
`grrb-code_fit_n15360-engine` restore, `posttrain-code-20261022`, and Engram enabled.
Native scoring used the 110000 MiB budget and a memory watchdog. Metrics below
come from `anchor_metrics`; NLL uses 127 next-token targets, distribution metrics
use all 128 rows. This fixture is too short for a quality claim.

| Native revision / chunk | Mean NLL | Mean KL to reference | Mean overlap | Same top token |
| --- | ---: | ---: | ---: | ---: |
| Baseline `8bf9394` / 1 | 1.9667 | 0.03278 | 0.9366 | 96.88% |
| Baseline `8bf9394` / 8 | 1.9638 | 0.05357 | 0.9289 | 94.53% |
| Upload-only `27657e1` / 1 | 1.9611 | 0.04208 | 0.9314 | 97.66% |
| PR #7 `0bfc98c` / 1 and 8 | 1.9179 | 0.10299 | 0.9130 | 92.97% |

The default CPU reference NLL is 1.9856. A second CPU run reproduced its full
logits byte for byte in 213 s. PR #7's chunk outputs are byte-identical, including
a repeat of the previously recorded files. Its logits SHA-256 is
`52268236361d12a72e20b1d284d6d4589f15eb5629e15df6b3d816e2bfd623ba`.
The reference logits SHA-256 is
`116b29bf851e4062ddd7a37146e00156dd531c8380dace3201fae0e5c8128a09`.

The exact-embedding diagnostic completed in 210 s with 6.91 GiB peak physical
footprint and no swap growth. Its reference NLL is 1.9838, and the logits SHA-256 is
`691f9fb25c3119d2555c8c4ecb3069e313a1d85aab07ed473e7a14bddd1ed358`.
All other reference settings and native output files are unchanged:

| Native revision / chunk | Mean KL to exact-embedding reference | Mean overlap | Same top token |
| --- | ---: | ---: | ---: |
| Baseline `8bf9394` / 1 | 0.08046 | 0.9125 | 94.53% |
| Baseline `8bf9394` / 8 | 0.06580 | 0.9245 | 96.09% |
| Upload-only `27657e1` / 1 | 0.08459 | 0.9157 | 94.53% |
| PR #7 `0bfc98c` / 1 and 8 | 0.05136 | 0.9302 | 94.53% |

**The numerical merge gate remains open.** The candidate ranking reverses when
only the embedding policy changes: PR #7 is farther from the default bf16
reference but closer to the exact-embedding reference. The latter matches the
native Q4_K lookup and the existing teacher's `--exact-weights` setting. This
sensitivity prevents attributing the gap to a specific kernel or concluding that
PR #7 regresses from the default table alone. Lower fixture NLL does not establish
correctness. Both exceed the same-weight KL target of roughly 1e-4–1e-3 stated in
`gguf-tools/scripts/v41_engine_parity_spark.sh`; the metric tool's generic 0.90
overlap target is not a substitute. Differences due to the CPU adaptations and
engine behavior still need to be isolated before interpreting this as a specific
kernel defect or accepting the rounding change. The reference uses the quantized
model, not original FP weights; an official CUDA-kernel comparison remains separate.
