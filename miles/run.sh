#!/usr/bin/env bash

set -euo pipefail

CONFIG="${1:-hello-qwen-0-6b}"
MODE="${2:-single}"

export MODAL_ENVIRONMENT="${MODAL_ENVIRONMENT:-peyton-agents}"

modal run miles/modal_train.py::prepare_dataset --config "${CONFIG}"
modal run miles/modal_train.py::download_model --config "${CONFIG}"

if [[ "${MODE}" == "multi" ]]; then
  modal run miles/modal_train.py::train_multi_node --config "${CONFIG}"
else
  modal run miles/modal_train.py::train_single_node --config "${CONFIG}"
fi
