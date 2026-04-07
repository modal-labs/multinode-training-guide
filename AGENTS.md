# AGENTS.md

Repo-wide notes for coding agents working in this repository.

## Working Rules

- Never commit raw secrets, API keys, or token values to the repository or its docs.
- Keep repository docs generic to contributors using this repo, not tailored to one local machine or one user's workflow.
- Use `uv` for Python dependency management and command execution in this repo. Prefer `uv run`, `uv lock`, and `uv sync`, and do not install Python packages at the system level.
- For launching and debugging Modal training jobs, see [docs/agent-modal-training.md](docs/agent-modal-training.md).
- For repo-wide example drift validation, see [docs/agent-example-validation.md](docs/agent-example-validation.md).
- Keep agent-facing docs focused on durable repo workflow. Move one-off incident notes into commit messages, PRs, or issue threads instead of preserving them as permanent guidance.
