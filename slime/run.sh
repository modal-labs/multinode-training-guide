#!/bin/bash
# Run SLIME GRPO training on Modal
#
# Usage:
#   ./slime/run.sh <config-name>
#   ./slime/run.sh qwen-4b-1replica

set -e

CONFIG="${1:?Usage: $0 <config-name>}"

# cd to script directory (slime/)
cd "$(dirname "$0")"

# Extract app name from config
APP_NAME=$(python -c "from configs import get_config; print(get_config('$CONFIG').app_name)")

echo "Config: $CONFIG"
echo "App:    $APP_NAME"
echo ""

SLIME_APP_NAME="$APP_NAME" modal run -d modal_train.py::train_multi_node --config "$CONFIG"
