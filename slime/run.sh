#!/bin/bash
# Run SLIME GRPO training on Modal
#
# Usage:
#   ./slime/run.sh <config-name> [gpu-override]
#   ./slime/run.sh glm-4-7
#   ./slime/run.sh glm-4-7 H100:8

set -e

CONFIG="${1:?Usage: $0 <config-name> [gpu-override]}"
GPU_OVERRIDE="${2:-}"

# cd to script directory (slime/)
cd "$(dirname "$0")"

# Extract config values
APP_NAME=$(python -c "from configs import get_config; print(get_config('$CONFIG').app_name)")
GPU=$(python -c "from configs import get_config; print(get_config('$CONFIG').gpu)")
N_NODES=$(python -c "from configs import get_config; print(get_config('$CONFIG').n_nodes)")

# Override GPU if specified
if [ -n "$GPU_OVERRIDE" ]; then
    echo "Overriding GPU: $GPU -> $GPU_OVERRIDE"
    # Update the gpu= line right before train_multi_node's @modal.experimental.clustered
    # Match pattern: gpu="...", and replace with override (only first occurrence)
    sed -i "0,/gpu=\"[A-Z0-9]*:[0-9]*\",  # GLM/s//gpu=\"$GPU_OVERRIDE\",  # GLM/" modal_train.py
    GPU="$GPU_OVERRIDE"
fi

echo "Config:  $CONFIG"
echo "App:     $APP_NAME"
echo "GPU:     $GPU"
echo "Nodes:   $N_NODES"

SLIME_APP_NAME="$APP_NAME" modal run -d modal_train.py::train_multi_node --config "$CONFIG"
