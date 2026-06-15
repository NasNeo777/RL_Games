#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PLUGIN_DIR="$ROOT/.venv/lib/python3.13/site-packages/PyQt5/Qt5/plugins/platforms"
LABEL_DIR="$ROOT/datasets/jump_labelme/raw"

export QT_QPA_PLATFORM_PLUGIN_PATH="$PLUGIN_DIR"
export QT_PLUGIN_PATH="${QT_PLUGIN_PATH:-$ROOT/.venv/lib/python3.13/site-packages/PyQt5/Qt5/plugins}"

if [ "$#" -eq 0 ]; then
  exec "$ROOT/.venv/bin/labelme" "$LABEL_DIR"
else
  exec "$ROOT/.venv/bin/labelme" "$@"
fi
