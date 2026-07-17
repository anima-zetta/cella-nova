#!/usr/bin/env bash
# Integration test: compare Python vs fl-rs GPU across all grid sizes.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GRID_SIZES=(64 128 256 512)
PASS=0
FAIL=0

for GRID in "${GRID_SIZES[@]}"; do
    echo ""
    echo "=============================================="
    echo "  Grid size: ${GRID}x${GRID}"
    echo "=============================================="

    rm -f pngs/*.png

    echo "  [Python] Generating frames..."
    .venv/bin/python3 train/save_frames_png.py --grid-size "$GRID" 2>&1 | tail -1

    echo "  [fl-rs] Generating frames..."
    cargo run --release --bin ml-rs-save-pngs -- --grid-size "$GRID" 2>&1 | tail -1

    echo "  [Compare] Python vs fl-rs..."
    .venv/bin/python3 train/compare_pngs.py 2>&1

    if .venv/bin/python3 train/compare_pngs.py 2>&1 | grep -q "✅ Match"; then
        PASS=$((PASS + 1))
        echo "  >>> ${GRID}x${GRID}: PASS"
    else
        FAIL=$((FAIL + 1))
        echo "  >>> ${GRID}x${GRID}: FAIL"
    fi
done

echo ""
echo "=============================================="
echo "  Results: $PASS passed, $FAIL failed"
echo "=============================================="
exit $FAIL
