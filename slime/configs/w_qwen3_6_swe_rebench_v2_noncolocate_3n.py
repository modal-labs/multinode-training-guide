"""Qwen3.6-35B-A3B SWE-rebench-V2 (Python) agentic RL — noncolocate, three nodes.

Dataset sibling of ``w_qwen3_6_swe_noncolocate_3n``: identical topology and engine
recipe (1 train + 2 rollout nodes, dp-attention OFF, 8× TP2 SGLang behind
sgl-router — see that config's docstring), only the **data** changes. Trains on the
FULL harbor build of ``nebius/SWE-rebench-V2`` (Python subset, ~7.2k tasks, no eval
holdout), published by ``agentic_rl.environment.convert2slime.swerebench``;
evals a transfer slice on the published ``swegym_lite`` held-out set.

SWE-rebench-V2 is SWE-bench-style; the converter renders each row into a harbor
task dir (prebuilt ``image_name`` via a Dockerfile-wrapper, reset+apply test_patch,
``install_config.test_cmd``, a stdlib pytest grader → F2P/P2P resolution). Scope is
Python-only because the public ``SWE-rebench/SWE-bench-fork`` only registers the
Python ``parse_log_pytest*`` parsers (other languages need Nebius's non-public
registry). Everything else (model, checkpoint, agent env, algorithm, optimizer,
eval defaults) is inherited from the 3n config → ``w_qwen3_6_swe_colocate_1n``.

Before the first run: publish the dataset to ``junlin-modal/swe-rebench-v2`` and
``download_data`` (see convert2slime/README.md). Reward shaping is the repo default
(``fractional`` over F2P∪P2P); set ``ASYNC_RL_REWARD_SHAPE=binary`` in the
``environment`` dict for the clean resolved-only signal.

    EXPERIMENT_CONFIG=w_qwen3_6_swe_rebench_v2_noncolocate_3n \
        uv run --no-dev modal run -d slime/modal_train.py::train
"""

from configs import w_qwen3_6_swe_noncolocate_3n as _base
from configs.base import CHECKPOINTS_PATH, run_tag
from configs.datasets import eval_datasets, pull, subsample, train_path

# Same Modal image / dev overlay as the SWE-Gym 3n config.
modal = _base.modal

# W&B run name; run_tag() appends a launch timestamp so dumps don't collide.
_RUN_TAG = run_tag("qwen3.6-35b-a3b-swe-rebench-v2-noncolocate-3n")

# Datasets (one HF repo each; see configs/datasets.py). ALL swe_rebench_v2 Python
# tasks are training (no in-distribution holdout), so eval is a transfer slice on
# the published swegym_lite held-out set (a separate repo + tasks/ tree).
_TRAIN = "swe_rebench_v2"
_EVAL = [("swegym_lite", None), ("swebench_verified", None)]  # transfer/generalization eval; set [] to disable


class _Slime(_base._Slime):
    # ── Data (repoint the inherited SWE-Gym dataset at SWE-rebench-V2) ─────────
    prompt_data = train_path(_TRAIN)
    async_mode = False
    eval_config = {
        "defaults": {
            "n_samples_per_eval_prompt": 1,
            "temperature": 0.6,  # low-but-nonzero: Qwen3 degenerates at greedy
            "top_p": 1.0,
        },
        "datasets": eval_datasets(_EVAL),  # n subsamples eval.jsonl in download_data
    }

    # ── WandB / debug dumps ────────────────────────────────────────────────────
    wandb_group = _RUN_TAG
    save_debug_rollout_data = (
        f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{_RUN_TAG}/rollout_{{rollout_id}}.pt"
    )

    def download_data(self) -> None:
        """Pull the SWE-rebench-V2 repo into /data/swe_rebench_v2/ and subsample
        the eval slice. Conversion/publish is offline (convert2slime/swerebench.py)."""
        for key in {_TRAIN, *(k for k, _ in _EVAL)}:
            pull(key)
        for key, n in _EVAL:
            if n is not None:
                subsample(key, n)


slime = _Slime()
