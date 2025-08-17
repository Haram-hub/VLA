#!/bin/bash

# Standard LeRobot Training with LIBERO Dataset
# This uses the standard lerobot/scripts/train.py

echo "=== Standard LeRobot Training with LIBERO Dataset ==="

# Set environment
export PYTHONPATH="/home/lambs1031/SmolVLA_0812/lerobot/src:$PYTHONPATH"
export PYTHONPATH="/home/lambs1031/SmolVLA_0812/LIBERO:$PYTHONPATH"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate SmolVLA_py39

# Step 1: Convert LIBERO to LeRobot format
echo "Step 1: Converting LIBERO dataset to LeRobot format..."
python libero_to_lerobot_dataset.py \
    --benchmark libero_object \
    --task_order 0 \
    --output_dir ./lerobot_libero_dataset

# Step 2: Run standard LeRobot training
echo "Step 2: Running standard LeRobot training..."
cd /home/lambs1031/SmolVLA_0812/lerobot

python lerobot/scripts/train.py \
    --policy.type=smolvla \
    --dataset.repo_id=local:/home/lambs1031/SmolVLA_0812/LIBERO/lerobot_libero_dataset \
    --batch_size=32 \
    --steps=10000 \
    --output_dir=./libero_training_output

echo "Training completed!"
echo "Results saved to: ./libero_training_output"
