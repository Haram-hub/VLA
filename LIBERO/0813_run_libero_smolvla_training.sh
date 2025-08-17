#!/bin/bash

# LIBERO to SmolVLA Training Script
# This script runs the complete training pipeline

echo "=== LIBERO to SmolVLA Training Pipeline ==="

# Set environment
export PYTHONPATH="/home/lambs1031/SmolVLA_0812/lerobot/src:$PYTHONPATH"
export PYTHONPATH="/home/lambs1031/SmolVLA_0812/LIBERO:$PYTHONPATH"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate SmolVLA_py39

# Set output directory
OUTPUT_DIR="./smolvla_libero_results_$(date +%Y%m%d_%H%M%S)"

echo "Output directory: $OUTPUT_DIR"

# Run training pipeline
python libero_to_smolvla_training.py \
    --benchmark libero_object \
    --task_order 0 \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 32 \
    --steps 10000

echo "Training completed! Results saved to: $OUTPUT_DIR"

# Show results
echo ""
echo "=== Training Results ==="
echo "Check the following files:"
echo "- $OUTPUT_DIR/training_visualization.png (Training plots)"
echo "- $OUTPUT_DIR/loss_data.json (Loss statistics)"
echo "- $OUTPUT_DIR/smolvla_libero_model/checkpoint.pt (Trained model)"
echo "- $OUTPUT_DIR/libero_libero_object/dataset.json (Converted dataset)"
