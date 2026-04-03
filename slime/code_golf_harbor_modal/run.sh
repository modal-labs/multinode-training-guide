#!/bin/bash
set -euo pipefail

SCRIPT="slime/code_golf_harbor_modal/modal_train.py"
MODE="${1:-}"

case "$MODE" in
  prepare)
    modal run "${SCRIPT}"::prepare_dataset
    ;;
  download)
    modal run "${SCRIPT}"::download_model --config qwen-8b-multi
    ;;
  train)
    modal run "${SCRIPT}"::train_multi_node --config qwen-8b-multi
    ;;
  serve)
    modal serve "${SCRIPT}"::serve_latest_checkpoint
    ;;
  eval)
    modal run "${SCRIPT}"::eval_latest_checkpoint --n-concurrent "${2:-256}" --n-tasks "${3:-500}"
    ;;
  *)
    echo "Usage: $0 {prepare|download|train|serve|eval [n_concurrent] [n_tasks]}"
    exit 1
    ;;
esac
