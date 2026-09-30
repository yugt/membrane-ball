#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/4] Python syntax check"
uv run python -m py_compile \
  game_python/game.py \
  game_python/physics.py \
  game_python/verify_physics.py \
  verify_energy.py \
  verify_physics_stage5.py \
  record_and_analyze.py

echo "[2/4] Energy conservation verification"
uv run python verify_energy.py

echo "[3/4] Stage 5 physics verification"
uv run python verify_physics_stage5.py

echo "[4/4] membrane-rl test suite"
(cd membrane-rl && uv run --project .. python -m pytest -q tests/)

echo "Verification complete."
