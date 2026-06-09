## Adding a new model config to slime

When asked to add a new model example to slime, you should output artifacts in a temporary directory in `configs/workstation/[model_name]/` folder. Once you are finished, add the finished config to the `configs` folder.

Always read the common gotchas.

### Phase 1: Discovery

First try looking for the existing model running on slime. You can find examples in [slime model scripts](https://github.com/THUDM/slime/tree/main/scripts/models) or [slime examples](https://github.com/THUDM/slime/tree/main/examples). If you cannot find an existing model, find the model with the most similar architecture. Reference huggingface for model architecture.

Then, output your first artifact, which is a file called `model_setup.md`, containing:
- Is there this model or a model with the same architecture that is already validated on slime?
- Is this model validated to be supported in megatron?
- Is this model validated to be supported in sglang?
- What is your plan for train configuration?
- How long do you expect each step in training take?
- How long do you expect each substep to take (e.g. rollout server initialization, weight sync, rollouts, etc)

### Phase 2: Implementation

Output a slime config you believe will work, and kick off a run with 1 single step. Check this step works e2e. Output the config in `configs` folder directly, and keep track of progress in `progress_log_[attempt_count].md`.

While tracking the progress, also make sure the timing lines up with your expectations in the `model_setup.md` artifact.

If this step does not work, go back to phase 1: what assumptions did you make in phase 1 that were incorrect and caused this? Output an artifact if it fails with `failure_analysis_[attempt_count].md`.

### Phase 3: Validation

Run it for more than 1 step, and kick off a run for 10-20 steps. Make sure the training does not fail. If it fails, create a minimal repro of the problem and work to address.

### Phase 4: Productionize

Create a PR with the slime config changes, and justify any patches you have made. If it's possible to not patch, do not patch.


# Common gotchas

Slime by default use mbridge (`megatron_to_hf_mode=""`) instead of bridge (`megatron_to_hf_mode="bridge"`), which requires it to preconvert the weights. To determine if we should use bridge mode or mbridge, look upstream at the slime codebase at what was used for similar models.
