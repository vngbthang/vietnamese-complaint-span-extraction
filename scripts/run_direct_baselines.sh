#!/bin/bash
# Run direct baselines experiment

set -e

echo "========================================"
echo "Running Direct Baselines"
echo "========================================"

echo ""
echo "[1/2] Checking data integrity..."
python scripts/check_data_integrity.py

echo ""
echo "[2/2] Training direct baseline models..."
python scripts/train_direct_baselines.py --config configs/direct_baselines.yaml

echo ""
echo "========================================"
echo "Direct baselines complete!"
echo "Results: outputs/experiments/direct_baselines/"
echo "========================================"
