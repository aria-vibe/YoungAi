# Metal upload ordering and verification parity

This change is separate from PR #6's memory/lifetime fixes. `fable5.md` remains
maintainer-owned. Keep detailed timing logs and model outputs outside the tree.

## Root cause and upload audit

`v41_compress_source` writes every source layer's group positions into the same
`st->posg`, then queues RoPE readers. Normal single-token decode encodes all layers
in one batch. On the 1017 model, sources L2/L8/L14 have ratio 2 and L20 has ratio 1:
at odd position p, the former require p-1 and the latter writes p. The old Metal
zerocopy wrapper immediately overwrites the shared bytes before prior readers run.
The model-free `plain decode posg RoPE` probe reproduces this without DSpark.

Metal tensor writes now snapshot the host bytes before returning and enqueue the
copy after prior readers. Inputs must be uploaded before encoding their consumers;
end/synchronize the batch before CPU reads through `contents()`. Idle writes remain
immediate. Pages append disjoint 256-byte-aligned slices and reset only after their
last command buffer completes or errors. At most sixteen 1 MiB pages are retained;
large/overflow uploads are retained only by their command buffers. Cleanup releases
the pool after draining. No callback touches pool state.

Audited V4.1 call families (including async/zerocopy wrappers):

| Call site | Ordering and source lifetime |
| --- | --- |
| `core_v41_forward.c`: tok, pos, vrows, pre_mix | Written before begin/body; CPU arrays may be freed after the snapshot. |
| `core_v41_attn.c`: posg | Written before both RoPE readers; later layers reuse the destination. This is the reproduced hazard. |
| `core_v41_engram.c`: eraw | Per-layer upload precedes dequantization; pinned CPU data lives through the forward. |
| `core_v41_draft.c`: slots and cache initialization | Initialization precedes readers; queued snapshots preserve each successive slot value. |
| `core_v41_draft_amp.c` / `core_v41_amp.c`: adapters | Load/concatenate fresh adapter tensors before attaching or using them; host load buffers may then be reused. |
| `core_v41_multi.c` / `core_v41_mgraph.c`: batch inputs | Written before the batch/body; Metal capture is unavailable and uses direct dispatch. |
| `core_v41_vision.c`: weights, patches, image rows | Written before projection/selection; image delimiter rows are disjoint from queued image copies; synchronize before publication. |

No V4.1 caller intentionally encodes an input reader before its upload or accesses
a queued destination through `contents()`. The API comment states the ordering
contract so new callers cannot depend on late host writes.

## Arithmetic and model-free validation

The Q4_K verification kernel shares packed weights/scales across 2–8 rows and uses
the single-row lane/block traversal and SIMD reduction. Reassociation and implicit
FMA contraction are disabled locally in both Q4_K kernels: otherwise compiler
choices can change the result with NT even when the source expression is identical.
This may change baseline single-row rounding and requires the model-level gate.
The skip flag remains grid-wide (`skip[0]`).

Attention receives split size/cap from the host. `ratio > 0` explicitly selects
causal per-row visibility; `ratio == 0` preserves frozen index fixtures, independently
of the half-prefill precision switch. Frozen backward fixtures request zero.

```sh
make -j6 ds4 ds4-server ds4_test
make linecount
./ds4_test --metal-upload --metal-v41 --server --engine-units --rax --tp-allreduce
make test
```

The upload suite checks batch/pending order, real compressor-position RoPE,
stack-source snapshots, page rollover/reuse, tensor views and unaligned copies.
Q4_K exact comparisons cover 256/1024/5120 input dimensions, 1/4 groups, rows 2–8,
rounded/unrounded outputs and output strides. Attention checks ratios 1/2/4/128,
early/split/compressor boundaries, long positions and both precision settings.
Existing primitives also compare against scalar reference math and finite differences.

## Performance and full-model gates

Build the same bounded microbenchmark on each revision and run sequentially on an
idle GPU. Its matrix is 5120 by 32768 (90 MiB packed Q4_K); compare medians per row
count. This measures kernel time including submission, not end-to-end DSpark speed.

```sh
make -f Makefile -f tests/repro/metal_q4_bench.mk -j6 ds4-metal-q4-bench
./ds4-metal-q4-bench
```

For end-to-end tests, use the same 1017 model, sidecar, posttrain, Engram, token IDs,
greedy settings, 110000 MiB budget and memory watchdog. Use a short prompt for
baseline DSpark so it fits; compare repeated cold/cache-hit requests. Apply the
separate memory PR when testing combined 8K behavior. Record output hashes,
acceptance and decode speed. `startup` logs separate prefill, pre-draft work and
draft initialization; the server's `prompt_ms` includes time through first emission.
The earlier raw logs show 17.1 s vs 12.3 s inside prefill, not draft initialization.

Compare fixed teacher-forced IDs at chunks 1 and 8 and run upload-only/arithmetic-only
ablations. Batch agreement is not independent quality validation. The inherited
`caliper_ref.sh` expects a 43-layer V4 reference and `--logprob-vectors` uses V4
fixtures. A matching V4.1 reference-forward/logprob comparison remains a merge gate;
missing weights/fixtures or skipped suites must be reported, never treated as passes.
