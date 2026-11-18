"""Image-build patcher: fix LoRA-B buffer sizing on quantized column-parallel layers.

Root cause of "LoRA B output dim ... does not match base partition prefix dim"
under --quantization fp8:

  * mem_pool.get_lora_b_shape derives the effective TP for non-MoE column
    modules from _row_parallel_shard_tp, an INPUT-sharding probe
    (input_size // input_size_per_partition).
  * On bf16, UnquantizedLinearMethod never sets input_size_per_partition, so
    the probe falls back to the global tp_size and the code path below then
    corrects the output dim via the output-side probe. Works.
  * Fp8LinearMethod.create_weights DOES set layer.input_size_per_partition
    (== input_size for column-parallel layers, whose input is unsharded), so
    the probe returns 1, the whole sharding branch is skipped, and LoRA-B is
    sized at the FULL output dim while the base layer stays TP-sharded.
    set_lora_info then fails (e.g. shared_experts.gate_up_proj: B=4096 vs
    per-rank partitions [1024, 1024] -> 2048).

Fix: for non-MoE column-parallel modules, use the base module's
output_size_per_partition (probed by _column_parallel_out_partition) as the
authoritative per-rank LoRA-B output dim. It exists on both bf16 and
quantized layers and is exactly what set_lora_info validates against.
Applied at image build via `python /tmp/sglang_fp8_lora_fix.py`.
"""

from pathlib import Path

P = Path("/sgl-workspace/sglang/python/sglang/srt/lora/mem_pool.py")

OLD = '''        if (
            effective_tp_size > 1
            and module_name not in ROW_PARALLELISM_LINEAR_LORA_NAMES
            and module_name not in REPLICATED_LINEAR_LORA_NAMES
        ):
            # If the base column-parallel module is fully REPLICATED (its actual
            # output_size_per_partition still equals the full output_dim -- e.g. the
            # dense MLP gate_up under --moe-dense-tp-size 1), its output is NOT
            # sharded, so keep LoRA-B at the full output dim. Dividing by the global
            # tp_size here undersizes B and crashes set_lora_info ("LoRA B output dim
            # != base partition prefix dim"). Non-MoE only; MoE shards by moe_tp_size.
            probed_out = (
                None
                if self.is_moe_module(module_name)
                else self._column_parallel_out_partition(
                    module_name, base_model, layer_idx
                )
            )
            if probed_out is not None and probed_out == output_dim:
                pass  # replicated base: keep full B output dim
            else:
                output_dim = self._column_parallel_lora_b_per_rank_dim(
                    module_name, output_dim, effective_tp_size
                )
'''

NEW = '''        if (
            module_name not in ROW_PARALLELISM_LINEAR_LORA_NAMES
            and module_name not in REPLICATED_LINEAR_LORA_NAMES
            and not self.is_moe_module(module_name)
        ):
            # The base module's output_size_per_partition is the ground truth
            # for LoRA-B's per-rank output dim (replicated OR TP-sharded), and
            # it is quant-independent. Do NOT gate this on the input-sharding
            # probe above: Fp8LinearMethod sets input_size_per_partition ==
            # input_size on column-parallel layers, which makes that probe
            # return 1 and previously skipped sharding entirely, sizing LoRA-B
            # at the full output dim against a TP-sharded base and crashing
            # set_lora_info ("LoRA B output dim != base partition prefix dim").
            probed_out = self._column_parallel_out_partition(
                module_name, base_model, layer_idx
            )
            if probed_out is not None:
                output_dim = probed_out
            elif effective_tp_size > 1:
                output_dim = self._column_parallel_lora_b_per_rank_dim(
                    module_name, output_dim, effective_tp_size
                )
        elif (
            effective_tp_size > 1
            and module_name not in ROW_PARALLELISM_LINEAR_LORA_NAMES
            and module_name not in REPLICATED_LINEAR_LORA_NAMES
        ):
            # MoE modules keep the moe_tp_size sharding path.
            output_dim = self._column_parallel_lora_b_per_rank_dim(
                module_name, output_dim, effective_tp_size
            )
'''

src = P.read_text()
assert src.count(OLD) == 1, f"expected 1 match, got {src.count(OLD)}"
P.write_text(src.replace(OLD, NEW))
print("patched", P)
