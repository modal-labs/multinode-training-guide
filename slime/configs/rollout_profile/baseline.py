"""Baseline: the exact w_qwen3_6_swe_noncolocate_2n SGLang knobs, num_rollout=1.

This is the reference point every other variant in this package is compared
against. No SGLang overrides -- only the profile invariants (1 rollout, no eval)
from ``make_slime``.
"""

from configs.rollout_profile._base import make_slime, modal  # noqa: F401

slime = make_slime("baseline")
