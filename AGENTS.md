# AGENTS.md

## Cloud-specific instructions

This repo is a collection of **Modal multinode training examples** — there is no local application server or database. All computation runs remotely on [Modal](https://modal.com)'s GPU cloud via `modal run`.

### Development workflow

1. **Lint/format**: `ruff check .` and `ruff format --check .` (CI runs `ruff check --fix` and `ruff format` then auto-commits).
2. **Syntax validation**: `python3 -m py_compile <file>` on any changed `.py` file.
3. **Deploy/run**: `modal run <example>/modal_train.py` — requires a Modal account token (`MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET`). Without credentials, you can still lint, format, and syntax-check all code locally.

### Key caveats

- There is **no `requirements.txt` or `pyproject.toml`** at the repo root. All training dependencies are declared inline in each example's `modal.Image` definition and only installed inside Modal containers.
- The only local Python dependencies needed for development are `ruff` (linting) and `modal` (SDK for import validation and the CLI).
- `ruff check` will report pre-existing lint warnings in vendored code (nanoGPT from Karpathy, etc.) — these are expected and handled by CI's auto-fix.
- Each example directory follows the pattern: `modal_train.py` (entrypoint) + `train.py` (training logic) + `README.md`. See `STYLE_GUIDE.md` for conventions.
- The `PATH` must include `~/.local/bin` for user-installed pip packages (`ruff`, `modal`).
