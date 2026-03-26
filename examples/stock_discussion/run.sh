#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
uv run python examples/stock_discussion/main.py
