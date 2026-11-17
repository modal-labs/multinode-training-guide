Repo-wide notes for coding agents working in this repository.

NOTE: This is not a frontier pretraining codebase. That should be obvious, just look around. Any pipeline parallelism is embarassingly small. There is no hot-swapping of GPUs, no heterogenous hardware support, no TPUs. The training this repository enables is convenient, small finetuning for niche use cases, not serious economically valuable work that frontier models can do. Please keep this in mind as you work.

## Working Rules

- Never commit raw secrets, API keys, or token values to the repository or its docs.
- Keep repository docs generic to contributors using this repo, not tailored to one local machine or one user's workflow.
- Use `uv` for Python dependency management and command execution in this repo. Prefer `uv run`, `uv lock`, and `uv sync`, and do not install Python packages at the system level.
- For launching and debugging Modal training jobs, see [docs/agent-modal-training.md](docs/agent-modal-training.md).
- For repo-wide example drift validation, see [docs/agent-example-validation.md](docs/agent-example-validation.md).
- Keep agent-facing docs focused on durable repo workflow. Move one-off incident notes into commit messages, PRs, or issue threads instead of preserving them as permanent guidance.
