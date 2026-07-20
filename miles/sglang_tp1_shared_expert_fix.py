"""Fix SGLANG_SHARED_EXPERT_TP1 double-add in deepseek_v2.py.

With a TP1-replicated shared expert, upstream adds its output to the MoE
result "after the all-reduce" so each TP rank contributes it only once.
But `should_skip_post_experts_all_reduce` can skip that explicit all-reduce
in favor of a *deferred/replaced* reduction (FlashInfer AllReduce Fusion,
dp-attention's reduce-scatterv), in which case the "post-all-reduce" add
actually lands *before* the real reduction and the replicated shared output
is summed once per TP rank (8-32x here) — corrupting every MoE layer.

Fix: fold the shared output into the pre-reduction add scaled by
1/tp_size. Any linear reduction (all-reduce, fused all-reduce,
reduce-scatterv) then reconstitutes exactly one copy. tp_size is a power
of two, so the bf16 scaling is exact and there is no precision cost.

Verified empirically on GLM-5.2-FP8: TP1 + AllReduce Fusion and
TP1 + dp-attention both produced garbage output before this patch.
"""

from pathlib import Path

TARGET = Path("/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py")

SENTINEL = "MILES_TP1_SHARED_EXPERT_FIX"

# Pre-reduction arg: pass the scaled shared output instead of None when TP1.
OLD_FUSE_ARG = "None if self._shared_expert_tp1 else shared_output,"
NEW_FUSE_ARG = (
    "(shared_output * (1.0 / self.tp_size) if shared_output is not None else None) "
    "if self._shared_expert_tp1 else shared_output,  # " + SENTINEL
)

# Post-reduction adds: now double-counting (the scaled copy is already in),
# so neuter them.
OLD_POST_ADD_DUAL = (
    "        if self._shared_expert_tp1:\n"
    "            final_hidden_states += shared_output\n"
)
OLD_POST_ADD_NORMAL = (
    "        if shared_output is not None and self._shared_expert_tp1:\n"
    "            final_hidden_states += shared_output\n"
)
NEW_POST_ADD = (
    "        if False:  # " + SENTINEL + ": folded into pre-reduction add\n"
    "            final_hidden_states += shared_output\n"
)


def main() -> None:
    src = TARGET.read_text()
    if SENTINEL in src:
        print("already patched, skipping")
        return

    n_fuse = src.count(OLD_FUSE_ARG)
    assert n_fuse == 2, f"expected 2 maybe_fuse TP1 args, found {n_fuse}"
    src = src.replace(OLD_FUSE_ARG, NEW_FUSE_ARG)

    n_dual = src.count(OLD_POST_ADD_DUAL)
    assert n_dual == 1, f"expected 1 dual-stream post-add, found {n_dual}"
    src = src.replace(OLD_POST_ADD_DUAL, NEW_POST_ADD)

    n_normal = src.count(OLD_POST_ADD_NORMAL)
    assert n_normal == 1, f"expected 1 forward_normal post-add, found {n_normal}"
    src = src.replace(OLD_POST_ADD_NORMAL, NEW_POST_ADD)

    TARGET.write_text(src)
    print(f"patched {TARGET}: 2 pre-reduction args, 2 post-adds neutered")


if __name__ == "__main__":
    main()
