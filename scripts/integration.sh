#!/usr/bin/env bash
# Integration test: compare all 3 implementations across all grid sizes.
# Runs from the repo root (cella-nova/).
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

    # Clean previous PNGs
    rm -f pngs/*.png

    # --- 1. Python ---
    echo "  [Python] Generating frames..."
    if .venv/bin/python3 train/save_frames_png.py --grid-size "$GRID" 2>&1 | tail -1; then
        echo "  [Python] OK"
    else
        echo "  [Python] FAILED"
        exit 1
    fi

    # --- 2. Reference Rust ---
    echo "  [Reference] Generating frames..."
    if cargo run --release --bin save_pngs -- --grid-size "$GRID" 2>&1 | tail -1; then
        echo "  [Reference] OK"
    else
        echo "  [Reference] FAILED"
        exit 1
    fi

    # --- 3. fl-rs GPU ---
    echo "  [fl-rs] Generating frames..."
    if cargo run --release --bin fl-rs-save-pngs -- --grid-size "$GRID" 2>&1 | tail -1; then
        echo "  [fl-rs] OK"
    else
        echo "  [fl-rs] FAILED"
        exit 1
    fi

    # --- Compare ---
    echo "  [Compare] Running comparison..."
    if .venv/bin/python3 train/compare_pngs.py 2>&1; then
        echo "  [Compare] OK"
    else
        echo "  [Compare] FAILED"
        exit 1
    fi

    # Check if all three pairs report 100% or near-100% match
    # (grep for ✅ or "Match" in the output)
    if .venv/bin/python3 train/compare_pngs.py 2>&1 | grep -q "✅ Match"; then
        PASS=$((PASS + 1))
        echo "  >>> ${GRID}x${GRID}: PASS"
    else
        FAIL=$((FAIL + 1))
        echo "  >>> ${GRID}x${GRID}: FAIL (see above)"
    fi
done

echo ""
echo "=============================================="
echo "  Results: $PASS passed, $FAIL failed"
echo "=============================================="
exit $FAIL
