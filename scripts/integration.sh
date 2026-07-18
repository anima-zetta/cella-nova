#!/usr/bin/env bash
# Integration test: compare Python vs ml-rs GPU across all grid sizes.
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

    # Generate config and kernels for this grid size
    echo "  [Config] Generating creature for ${GRID}x${GRID}..."
    .venv/bin/python3 train/generate_kernel_json.py --grid-size "$GRID" --name mcl_creature 2>&1 | tail -2

    echo "  [Python] Generating frames..."
    .venv/bin/python3 train/save_frames_png.py --creature mcl_creature 2>&1 | tail -1

    echo "  [ml-rs] Generating frames..."
    cargo run --release --bin ml-rs-save-pngs -- --creature mcl_creature 2>&1 | tail -1

    echo "  [Compare] Python vs ml-rs..."
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
