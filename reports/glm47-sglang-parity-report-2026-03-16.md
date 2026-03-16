# GLM-4.7 SGLang Parity Report

Date: 2026-03-16

## Summary

GLM-4.7 post-train parity checks show that:

- Megatron-native checkpoint scoring works.
- HF/ms-swift-native scoring on the exported artifact works.
- SGLang serving starts and handles requests, but its scored outputs diverge catastrophically from both Megatron and HF on the merged GLM-4.7 export.

This is a real framework/export bug candidate, not a Modal scheduling issue.

## Working Evaluation Surfaces

Held-out eval data came from the deterministic DPO eval split generated for:

- `run_id = glm47-dpo-beta010`
- checkpoint `checkpoint-30`
- export `checkpoint-30-hf-merged`

Validated metrics:

- Megatron-native mean sequence logprob across the 16-example held-out slice:
  - `-1.1957163220662301`
- HF/ms-swift-native mean sequence logprob across the 16-example held-out slice:
  - `-1.1938755980029576`

These two paths are close enough to use as the native baseline.

## SGLang Result

Operational fix required before SGLang would start quickly:

- `--disable-cuda-graph` was not sufficient.
- `--disable-piecewise-cuda-graph` also had to be set explicitly.

After that fix, SGLang loaded the merged GLM-4.7 export, started serving, and completed a teacher-forced one-example scoring run.

Observed one-example parity result for `distilabel-math-dpo-256-0`:

- Megatron-native mean token logprob: `-1.0937994550195373`
- HF/ms-swift-native mean token logprob: `-1.1178475046466065`
- SGLang mean token logprob: `-11.92868423461914`

Per-token deltas on that example:

- HF vs Megatron mean absolute delta: `0.0988701641016022`
- SGLang vs HF mean absolute delta: `10.810836729972534`
- SGLang vs Megatron mean absolute delta: `10.834884779599603`

## Raw SGLang Evidence

The first teacher-forced SGLang payload written by the evaluator shows:

- `output_token_ids_logprobs = [[[-11.92868423461914, 78596, "Conditional"]]]`
- `output_token_logprobs = [[-11.92868423461914, 0, "!"]]`

That means SGLang is scoring the gold next token `"Conditional"` at roughly `-11.93` while also greedily emitting token id `0` (`"!"`) under the same context.

The committed SGLang per-example artifact for that run contains the same `-11.92868423461914` value for every response token, producing the catastrophic parity gap above.

## Conclusion

Current GLM-4.7 status:

- Megatron checkpoint save/load path: usable
- HF export + HF inference path: usable
- SGLang on the merged GLM-4.7 export: functionally broken for parity/logprob checks

The likely problem is either:

- the GLM-4.7 merged export artifact as consumed by SGLang, or
- SGLang's handling of this GLM-4.7 MoE export format

It is not explained by Modal capacity or startup behavior.
