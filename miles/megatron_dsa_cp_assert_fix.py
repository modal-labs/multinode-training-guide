"""Relax megatron-core's blanket "no CP for DSA" assert for the TileLang backend.

megatron-core's MCoreMLATransformerConfig.__post_init__ refuses
context_parallel_size > 1 whenever experimental_attention_variant == "dsa".
That is correct for the generic megatron-core DSAttention module (it has no
CP collectives), but Megatron-Bridge's GLM-5 TileLang path replaces that
module with CP-capable code (see bridge models/glm5/tilelang/tilelang_mla.py:
CP-gathered K, CP-local q RoPE replicate/slice, indexer varlen bounds
scattered over the CP group — the slime/baseten allgather-CP scheme).

This patch gates the assert on the DSA kernel backend: CP stays forbidden
for the megatron backend, and is allowed for tilelang. finalize() calls
__post_init__ on the provider instance, which carries dsa_attention_backend,
so the getattr sees it; the default keeps the assert enforced otherwise.

Correctness oracle for runs using this: train_rollout_logprob_abs_diff.
The trainer rescoring of SGLang-generated tokens catches wrong CP attention
math immediately (~0.01 healthy vs >1 broken).
"""

from pathlib import Path

TARGET = Path("/root/Megatron-LM/megatron/core/transformer/transformer_config.py")

OLD = '''        elif self.experimental_attention_variant == "dsa":
            assert (
                self.context_parallel_size == 1
            ), "Currently context parallelism is not supported by DSAttention!"'''

NEW = '''        elif self.experimental_attention_variant == "dsa":
            assert (
                self.context_parallel_size == 1
                or getattr(self, "dsa_attention_backend", "megatron") == "tilelang"
            ), (
                "Context parallelism with DSA requires the TileLang backend "
                "(megatron-core DSAttention has no CP collectives)."
            )'''

src = TARGET.read_text()
if NEW in src:
    print("already patched, skipping")
else:
    assert src.count(OLD) == 1, f"expected 1 match, got {src.count(OLD)}"
    TARGET.write_text(src.replace(OLD, NEW))
    print(f"patched {TARGET}: DSA CP assert gated on tilelang backend")
