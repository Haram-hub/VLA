#!/bin/bash

# Complete LeRobot Pipeline: Training + Evaluation + Visualization
# This script runs the full pipeline using LeRobot standard framework

echo "=== Complete LeRobot Pipeline with LIBERO Dataset ==="

# Set environment
export PYTHONPATH="/home/lambs1031/SmolVLA_0812/lerobot/src:$PYTHONPATH"
export PYTHONPATH="/home/lambs1031/SmolVLA_0812/LIBERO:$PYTHONPATH"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate SmolVLA_py39

# Set directories
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./lerobot_libero_pipeline_${TIMESTAMP}"
DATASET_DIR="${OUTPUT_DIR}/dataset"
TRAINING_DIR="${OUTPUT_DIR}/training"
EVALUATION_DIR="${OUTPUT_DIR}/evaluation"

echo "Output directory: $OUTPUT_DIR"

# Step 1: Convert LIBERO to LeRobot format
echo ""
echo "Step 1: Converting LIBERO dataset to LeRobot format..."
cd /home/lambs1031/SmolVLA_0812/LIBERO

python libero_to_lerobot_dataset.py \
    --benchmark libero_object \
    --task_order 0 \
    --output_dir "$DATASET_DIR"

# Step 2: Run LeRobot training
echo ""
echo "Step 2: Running LeRobot training..."
cd /home/lambs1031/SmolVLA_0812/lerobot

python lerobot/scripts/train.py \
    --policy.type=smolvla \
    --dataset.repo_id=local:"$DATASET_DIR" \
    --batch_size=32 \
    --steps=10000 \
    --output_dir="$TRAINING_DIR"

# Step 3: Run evaluation and visualization
echo ""
echo "Step 3: Running evaluation and visualization..."
cd /home/lambs1031/SmolVLA_0812/LIBERO

# Find the latest checkpoint
LATEST_CHECKPOINT=$(find "$TRAINING_DIR" -name "*.pt" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" ")

if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "Found checkpoint: $LATEST_CHECKPOINT"
    
    python lerobot_evaluation_visualization.py \
        --model_path "$LATEST_CHECKPOINT" \
        --output_dir "$EVALUATION_DIR"
else
    echo "No checkpoint found in $TRAINING_DIR"
fi

# Step 4: Show results summary
echo ""
echo "=== Pipeline Completed! ==="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "- Dataset: $DATASET_DIR"
echo "- Training: $TRAINING_DIR"
echo "- Evaluation: $EVALUATION_DIR"
echo ""
echo "Key files to check:"
echo "- Training logs: $TRAINING_DIR/logs/"
echo "- Model checkpoint: $LATEST_CHECKPOINT"
echo "- Evaluation plots: $EVALUATION_DIR/evaluation_results.png"
echo "- Evaluation videos: $EVALUATION_DIR/episode_*_video.mp4"
echo "- Results data: $EVALUATION_DIR/evaluation_results.json"


