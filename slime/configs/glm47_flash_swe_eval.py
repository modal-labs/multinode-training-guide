"""GLM-4.7-Flash — eval-only run of the agentic SWE setup (no training).

Eval twin of ``glm47_flash_swe_rebench_async``: same model / checkpoint / agent
harness / GLM rollout recipe (dp-attention + HiCache + EAGLE), flipped to
eval-only. Evaluates the same held-out suites as ``w_qwen3_6_swe_eval_2n``
(SWE-bench Verified / Pro / Multilingual, Terminal-Bench 2.1, SWE-Gym-Lite).

Runs SYNC + COLOCATE: the training config is fully-async, which has no eval
branch, and eval carries no training memory so the actor + rollout engines can
time-share one 2-node allocation. The eval suites are harbor-format HF repos
(``prompt``/``label``/``metadata``, ``task_type=harbor``), unlike the native
SWE-rebench training data — so the eval rewires the data keys and adds
``ASYNC_RL_TASK_ROOT`` (harbor resolves ``metadata.task_path`` against it).

    EXPERIMENT_CONFIG=glm47_flash_swe_eval \\
        uv run --no-dev modal run -d slime/modal_train.py::train
"""

from configs.base import CHECKPOINTS_PATH, run_tag
from configs.datasets import eval_datasets, pull, subsample, train_path
from configs.glm47_flash_swe_rebench_async import _Slime, modal  # noqa: F401

# W&B run name; run_tag() appends a launch timestamp so dumps don't collide.
_RUN_TAG = run_tag("glm4.7-flash-swe-eval")

# Same suites as w_qwen3_6_swe_eval_2n. None evals the full held-out eval.jsonl;
# an int subsamples it in download_data. Comment a line out to eval fewer suites.
_EVAL = [
    ("swebench_verified", None),
    ("swebenchpro", None),
    ("swebench_multilingual", None),
    ("terminal_bench_2_1", None),
    ("swegym_lite", None),   # in-distribution held-out (30)
    # ("usaco", 50),
]


class _SlimeEval(_Slime):
    # ── Eval-only, sync + colocate (fully-async has no eval branch) ─────────────
    async_mode = False
    colocate = True
    rollout_function_path = None  # default sync rollout drives the custom-generate hook
    num_rollout = 0
    eval_interval = 1             # any non-None value arms the eval-only branch
    skip_eval_before_train = False
    lr_decay_iters = 1
    sglang_server_concurrency = 32

    # 2 nodes colocated: the TP2/PP2/CP4 actor + GLM rollout engines time-share 16 GPU.
    # Eval has no optimizer/activation memory, so colocate fits even though training
    # is non-colocated; mem_fraction backs off to leave room for the resident weights.
    # Pin a FULL node (8 GPU): Modal requires multi-node H200 functions to use all 8
    # GPU/node, and TP2×PP2×CP4=16 needs 2×8=16 GPU. Don't inherit actor_num_gpus_per_node
    # from the training base (it may carry a partial-node value that breaks both the
    # parallelism math and Modal's whole-node rule).
    actor_num_nodes = 4
    actor_num_gpus_per_node = 8
    rollout_num_gpus = 32
    sglang_mem_fraction_static = 0.6

    # sglang_speculative_algorithm = None
    # sglang_speculative_num_steps = None
    # sglang_speculative_eagle_topk = None
    # sglang_speculative_num_draft_tokens = None

    sglang_speculative_algorithm = "EAGLE"
    sglang_speculative_num_steps = 3
    sglang_speculative_eagle_topk = 1
    sglang_speculative_num_draft_tokens = 4

    # ── Eval data: the suites are harbor-format (prompt/label/metadata,
    # task_type=harbor) → HarborEnv, not the native-swerebench training schema.
    input_key = "prompt"                       # harbor rows use "prompt" (training used "input")
    prompt_data = train_path("swegym_lite")    # unused at num_rollout=0; exists after download_data
    eval_max_response_len = 8192
    environment = {**_Slime.environment, "ASYNC_RL_TASK_ROOT": "/data"}  # harbor task_path root

    eval_config = {
        "defaults": {"n_samples_per_eval_prompt": 1, "temperature": 0.6, "top_p": 1.0},
        "datasets": eval_datasets(_EVAL),
    }

    # ── WandB / dumps ───────────────────────────────────────────────────────────
    wandb_group = _RUN_TAG
    save_debug_rollout_data = (
        f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{_RUN_TAG}/rollout_{{rollout_id}}.pt"
    )

    def download_data(self) -> None:
        """Pull each eval suite's repo into /data/<key>/, then subsample where asked."""
        for key, _ in _EVAL:
            pull(key)
        for key, n in _EVAL:
            if n is not None:
                subsample(key, n)


slime = _SlimeEval()
