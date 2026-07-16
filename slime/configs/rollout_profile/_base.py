"""Factory for SGLang rollout-perf profile variants.

Every variant in this package is a thin wrapper around :func:`make_slime`: it
inherits the noncolocate two-node SWE rollout config
(``w_qwen3_6_swe_noncolocate_2n``) and overrides only the SGLang/rollout knob(s)
under test. The profile *invariants* are baked in here so variants stay
one-liners and stay comparable across the study:

  * ``num_rollout = 1``      -- profile step-0 generation only (we test rollout,
                                not training). Step 0 is cold-cache; for
                                steady-state engine numbers bump to 2 and
                                profile step 1 (see GUIDE.md "cold step").
  * ``eval_interval = None``  -- no eval leg stealing the shared engines.
  * per-variant W&B group + dump dir -- runs and ``rollout_*.pt`` dumps from
                                different variants never collide.

The factory also applies two SGLang companion-flag safety rules, because a
contradictory engine config silently wastes a whole GPU run:

  * speculative algorithm None  -> null the EAGLE companion flags
    (``num_steps`` / ``eagle_topk`` / ``num_draft_tokens``).
  * dp-attention off            -> null ``sglang_dp_size``,
    ``sglang_enable_dp_lm_head`` and ``sglang_ep_size`` (all three are part of
    the baseline's DP-attention layout and are invalid / meaningless as pure
    tensor parallelism). An explicit override always wins over the auto-null.

Launch (from ``multinode-training-guide/``):

    EXPERIMENT_CONFIG=rollout_profile.baseline \
        uv run --no-dev modal run -d slime/modal_train.py::train

Analyze: see
``agentic_rl/profiles/swe_sglang_rollout_perf_profile/GUIDE.md``.
"""

from configs import w_qwen3_6_swe_noncolocate_2n as _nc
from configs import w_qwen3_6_swe_colocate_2n as _co
from configs.base import CHECKPOINTS_PATH, run_tag

# Same Modal image / dev overlay as the config under test.
modal = _nc.modal

# The SGLang/rollout knobs the noncolocate_2n baseline ships with. One source of
# truth for "what baseline means" -- variants and the guide both read this.
BASELINE_KNOBS = {
    "rollout_num_gpus": 8,             # the whole rollout node; keep fixed (12/16 GPUs is not a clean node multiple)
    "rollout_num_gpus_per_engine": 8,  # TP per engine; #engines = rollout_num_gpus // this
    "sglang_mem_fraction_static": 0.7,
    "sglang_enable_dp_attention": True,
    "sglang_dp_size": 8,
    "sglang_ep_size": 8,
    "sglang_enable_dp_lm_head": True,
    "sglang_speculative_algorithm": "EAGLE",
    "sglang_disable_custom_all_reduce": False,
}

_SPEC_COMPANIONS = (
    "sglang_speculative_num_steps",
    "sglang_speculative_eagle_topk",
    "sglang_speculative_num_draft_tokens",
)
_DP_COMPANIONS = (
    "sglang_dp_size",
    "sglang_enable_dp_lm_head",
    "sglang_ep_size",
)


def _profile_fields(tag, overrides):
    """The profile invariants (1 rollout, no eval, per-variant group/dump) + the
    caller's overrides, as a fields dict ready for a ``_Slime`` constructor."""
    fields = dict(
        num_rollout=1,
        eval_interval=None,
        skip_eval_before_train=True,
        wandb_group=tag,
        save_debug_rollout_data=(
            f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{tag}/rollout_{{rollout_id}}.pt"
        ),
    )
    fields.update(overrides)
    return fields


def _apply_companion_nulls(fields, overrides, base_cls):
    """SGLang companion-flag safety rules (shared by both factories): null the
    EAGLE / DP-EP companion flags when the parent feature is off, unless the
    caller explicitly overrode them. ``base_cls`` supplies the defaults."""

    def _null_unless_overridden(*keys):
        for k in keys:
            if k not in overrides:  # an explicit override always wins
                fields[k] = None

    spec_default = getattr(base_cls, "sglang_speculative_algorithm", "EAGLE")
    if fields.get("sglang_speculative_algorithm", spec_default) is None:
        _null_unless_overridden(*_SPEC_COMPANIONS)

    dp_default = getattr(base_cls, "sglang_enable_dp_attention", True)
    if not fields.get("sglang_enable_dp_attention", dp_default):
        _null_unless_overridden(*_DP_COMPANIONS)


def make_slime(variant: str, **overrides):
    """Build a profile ``_Slime`` for ``variant`` (= W&B group + dump-dir name),
    applying ``overrides`` on top of the noncolocate_2n baseline.

    ``variant`` should be a short kebab/snake label, e.g. ``"baseline"``,
    ``"mem085"``, ``"nodp"``, ``"tp4"``. The run tag is
    ``qwen3.6-swe-rollout-profile-<variant>-<launch-ts>``.
    """
    tag = run_tag(f"qwen3.6-swe-rollout-profile-{variant}")
    fields = _profile_fields(tag, overrides)
    _apply_companion_nulls(fields, overrides, _nc._Slime)
    cfg = _nc._Slime(**fields)
    cfg._profile_variant = variant  # underscore => not a SLIME CLI arg
    return cfg


def make_colocate(variant: str, **overrides):
    """Build a profile ``_Slime`` from the **colocate** two-node base
    (``w_qwen3_6_swe_colocate_2n``) — all GPUs time-share inference + training,
    sync (``async_mode=False``). Use to profile colocate vs noncolocate-async at
    a given node budget.

    Node count is set via ``actor_num_nodes`` (colocate uses ALL its GPUs for the
    engines: ``rollout_num_gpus`` auto = ``actor_num_nodes * 8``). The colocate
    base ships 32k context; pass ``rollout_max_context_len=65536`` to match the
    noncolocate study for an apples-to-apples comparison. Same companion-null
    rules as ``make_slime``.

    NB: still ``num_rollout=1`` (clean step-0 rollout, pre-weight-update so no
    resync collapse). Step-0 *train* is cold, so read warm train_time from a
    multi-step reference (e.g. bmba6iva) rather than this run's train phase.
    """
    tag = run_tag(f"qwen3.6-swe-rollout-profile-{variant}")
    fields = _profile_fields(tag, overrides)
    _apply_companion_nulls(fields, overrides, _co._Slime)
    cfg = _co._Slime(**fields)
    cfg._profile_variant = variant  # underscore => not a SLIME CLI arg
    return cfg
