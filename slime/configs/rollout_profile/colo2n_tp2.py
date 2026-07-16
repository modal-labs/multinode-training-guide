"""COLOCATE variant: 2 nodes (16 GPU shared), 8xTP2 engines, dp-off, 64k ctx.

The apples-to-apples colocate twin of the noncolocate `tp2` champion: same engine
recipe (TP2, dp-attention off), same 64k context, but colocate (all 16 GPUs
time-share inference+training, sync). Tests whether reclaiming the 77%-idle train
node via colocate beats noncolocate-async at the same 2-node budget.

mem_fraction dropped to 0.6 (engine KV must coexist with Megatron weights/optimizer
on the same GPUs); back off further or to 32k ctx if it OOMs at startup — and note
that as the finding (TP2+64k may not fit colocate). Warm train_time comes from a
multi-step ref (bmba6iva ~247s on 16 GPU), not this num_rollout=1 step-0 run.
"""

from configs.rollout_profile._base import make_colocate, modal  # noqa: F401

slime = make_colocate(
    "colo2n_tp2",
    rollout_num_gpus=16,
    rollout_num_gpus_per_engine=2,
    sglang_enable_dp_attention=False,
    sglang_mem_fraction_static=0.6,
    rollout_max_context_len=65536,
)
