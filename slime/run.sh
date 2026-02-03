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
    python -c "
import re
with open('modal_train.py', 'r') as f:
    content = f.read()
content = re.sub(r'gpu=\"[A-Z0-9]*:[0-9]*\",  # GLM', 'gpu=\"$GPU_OVERRIDE\",  # GLM', content, count=1)
with open('modal_train.py', 'w') as f:
    f.write(content)
"
    GPU="$GPU_OVERRIDE"
fi

# Update cluster size to match config's n_nodes
python -c "
import re
with open('modal_train.py', 'r') as f:
    content = f.read()
content = re.sub(r'@modal\.experimental\.clustered\(\d+, rdma=True\)', '@modal.experimental.clustered($N_NODES, rdma=True)', content)
with open('modal_train.py', 'w') as f:
    f.write(content)
"

echo "Config:  $CONFIG"
echo "App:     $APP_NAME"
echo "GPU:     $GPU"
echo "Nodes:   $N_NODES"

SLIME_APP_NAME="$APP_NAME" modal run -d modal_train.py::train_multi_node --config "$CONFIG"
