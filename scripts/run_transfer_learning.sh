#!/bin/bash
# Run transfer learning experiment

set -e

echo "========================================"
echo "Running Transfer Learning"
echo "========================================"

echo ""
echo "[1/2] Checking data integrity..."
python scripts/check_data_integrity.py

echo ""
echo "[2/2] Training transfer learning models..."
python scripts/train_transfer_learning.py --config configs/transfer_learning.yaml

echo ""
echo "========================================"
echo "Transfer learning complete!"
echo "Results: outputs/experiments/transfer_learning/"
echo "========================================"
