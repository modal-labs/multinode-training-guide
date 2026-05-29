"""Kimi-K2.5 LoRA — same parallelism as ``kimi_k25_lora``, but PPO uses
sglang rollout log_probs as ``π_old`` directly (no train-side recompute,
no TIS).

Run:
    EXPERIMENT_CONFIG=kimi_k25_lora_rollout_logprobs modal run -d miles/modal_train.py::train

Tradeoffs vs ``kimi_k25_lora``:
  + Skips the pre-train log_prob recompute forward pass (saves ~10-20% wall
    clock per rollout).
  + Mathematically correct off-policy IS: ``π_old`` is the actual behavior
    policy (sglang Marlin int4) that sampled the trajectory.
  - The cross-engine kernel gap (Marlin int4 vs fake-QAT bf16) now lives
    inside the PPO IS ratio. As ``‖B_down‖`` grows, the kernel gap grows;
    once it crosses ``eps_clip`` (~0.2 in logprob), PPO starts capping the
    update on kernel-rounding boundaries rather than RL signal.
  - ``use_tis`` is mutually exclusive with ``use_rollout_logprobs`` in miles
    (arguments.py:2068-2069), so the explicit TIS clamp is gone.

Use when:
  - The kernel gap is verified small at all training steps (e.g. after Sol1
    bf16 inference lands), so PPO clip won't fire on kernel rounding.
  - Compute savings on the recompute pass actually matter.

Otherwise prefer ``kimi_k25_lora`` (PPO on within-train ratio + TIS clamp on
cross-engine). See analysis/README.md §8.4.
"""

from configs.kimi_k25_lora import _Miles as _LoRAMiles, modal  # noqa: F401


class _Miles(_LoRAMiles):
    wandb_group = "kimi-k25-lora-rollout-logprobs"

    use_rollout_logprobs = True
    use_tis = False


miles = _Miles()
