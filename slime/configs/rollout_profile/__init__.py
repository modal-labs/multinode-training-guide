# Rollout-perf profile configs: short (num_rollout=1) variants of
# w_qwen3_6_swe_noncolocate_2n that vary one SGLang/rollout knob at a time.
#
# Launch one with:
#   EXPERIMENT_CONFIG=rollout_profile.<variant> \
#       uv run --no-dev modal run -d slime/modal_train.py::train
#
# Methodology + analysis pipeline:
#   agentic_rl/profiles/swe_sglang_rollout_perf_profile/GUIDE.md
