# Thin Config Wrappers

Local configs are TOML wrappers around upstream SLIME CLI.
`configs/` is the only runtime config source.

## Goals
- Use SLIME `scripts/models/*.sh` `MODEL_ARGS` as source of truth.
- Keep local files focused on experiment intent (resource shape + chosen flags).
- Pass arbitrary upstream flags via `.args` files and CLI passthrough.

## TOML schema
Required:
- `model_id` (e.g. `Qwen/Qwen3-4B-Instruct-2507`)
- `model_args_script` (path relative to SLIME `scripts/models`)

Optional:
- `extends` (another config name)
- `model_args_env` (env vars used before sourcing `model_args_script`)
- `args_files` (list of `.args` files)
- `args` (inline token list)
- `sync` (bool, default false)
- `train_script` (defaults from `sync`)
- `app_name`, `gpu`, `n_nodes`, `wandb_project`, `wandb_run_name_prefix`

## `.args` format
- Parsed with `shlex`.
- `#` comments supported.
- Any SLIME flag is allowed.

## Runtime passthrough
From `run.py` / `modal_train.py`:
- `--slime-args "<raw flags>"`
- `--slime-args-file <path>`
- `--train-script <path>`

## MODEL_ARGS root resolution
1. `SLIME_MODEL_ARGS_ROOT` env var if set
2. Installed SLIME package path
3. Local fallback: `../dev/slime/scripts/models`
