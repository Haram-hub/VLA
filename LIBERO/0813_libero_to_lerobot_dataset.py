"""
Convert LIBERO dataset to LeRobot standard format
This allows using the standard lerobot/scripts/train.py
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

# Add paths
sys.path.append('/home/lambs1031/SmolVLA_0812/LIBERO')

# LIBERO imports
from libero.libero import get_libero_path, get_benchmark
from libero.lifelong.datasets import get_dataset, SequenceVLDataset
from libero.lifelong.utils import get_task_embs


def convert_libero_to_lerobot_format(libero_dataset, output_dir, task_name):
    """
    Convert LIBERO dataset to LeRobot standard format
    """
    print(f"Converting LIBERO dataset to LeRobot format...")
    
    # Create output directory structure
    output_path = Path(output_dir) / f"libero_{task_name}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create LeRobot dataset structure
    episodes = []
    tasks = []
    
    for i, sample in enumerate(tqdm(libero_dataset, desc="Converting samples")):
        # Extract data from LIBERO format
        obs = sample.get('obs', {})
        actions = sample.get('actions', [])
        task_desc = sample.get('task', 'robot task')
        
        # Convert observations
        images = []
        states = []
        
        for key, value in obs.items():
            if 'rgb' in key:
                # Convert to LeRobot image format
                if isinstance(value, torch.Tensor):
                    img = value.numpy()
                else:
                    img = value
                
                # Ensure correct format (H, W, C)
                if img.ndim == 4:  # (B, T, C, H, W)
                    img = img[0, -1]  # Take last timestep
                elif img.ndim == 3:  # (C, H, W)
                    img = img.transpose(1, 2, 0)
                
                images.append(img)
            
            elif 'joint' in key or 'gripper' in key:
                # Convert state information
                if isinstance(value, torch.Tensor):
                    state = value.numpy()
                else:
                    state = value
                
                if state.ndim == 2:  # (T, dim)
                    state = state[-1]  # Take last timestep
                
                states.extend(state.flatten())
        
        # Create episode data in LeRobot format
        episode_data = {
            "episode_index": i,
            "timestamp": np.arange(len(actions)) * 0.1,  # 10Hz sampling
            "observation": {
                "images": {
                    "camera": images[0] if images else np.zeros((224, 224, 3))
                },
                "state": np.array(states) if states else np.zeros(32)
            },
            "action": np.array(actions),
            "task": task_desc
        }
        
        episodes.append(episode_data)
        tasks.append(task_desc)
    
    # Save dataset in LeRobot format
    dataset_info = {
        "episodes": episodes,
        "tasks": list(set(tasks)),
        "num_episodes": len(episodes),
        "action_dim": len(actions[0]) if actions else 7,
        "state_dim": len(states) if states else 32
    }
    
    # Save as JSON (LeRobot format)
    with open(output_path / "dataset.json", "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"Converted {len(episodes)} episodes to {output_path}")
    return str(output_path)


def create_lerobot_config(output_dir, action_dim=7, state_dim=32):
    """
    Create LeRobot configuration file
    """
    config = {
        "policy": {
            "type": "smolvla",
            "config": {
                "max_state_dim": max(32, state_dim),
                "max_action_dim": max(32, action_dim),
                "freeze_vision_encoder": True,
                "train_expert_only": True,
                "train_state_proj": True,
                "vlm_model_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
                "load_vlm_weights": True
            }
        },
        "dataset": {
            "repo_id": f"local:{output_dir}",
            "type": "lerobot"
        },
        "training": {
            "batch_size": 32,
            "steps": 10000,
            "lr": 1e-4,
            "device": "cuda"
        }
    }
    
    config_path = Path(output_dir) / "config.yaml"
    import yaml
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return str(config_path)


def main():
    """
    Main function to convert LIBERO to LeRobot format
    """
    parser = argparse.ArgumentParser(description="Convert LIBERO to LeRobot format")
    parser.add_argument("--benchmark", default="libero_object", 
                       choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    parser.add_argument("--task_order", type=int, default=0)
    parser.add_argument("--output_dir", default="./lerobot_libero_dataset")
    
    args = parser.parse_args()
    
    print("=== LIBERO to LeRobot Dataset Conversion ===")
    print(f"Benchmark: {args.benchmark}")
    print(f"Task Order: {args.task_order}")
    print(f"Output Directory: {args.output_dir}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load LIBERO dataset
    print("\n1. Loading LIBERO dataset...")
    folder = get_libero_path("datasets")
    benchmark = get_benchmark(args.benchmark)(args.task_order)
    
    # Load first task for simplicity
    task_dataset, shape_meta = get_dataset(
        dataset_path=os.path.join(folder, benchmark.get_task_demonstration(0)),
        obs_modality={
            "rgb": ["rgb_static", "rgb_gripper"],
            "state": ["joint_states", "gripper_states"]
        },
        initialize_obs_utils=True,
        seq_len=1,
    )
    
    # Create vision-language dataset
    task_description = benchmark.get_task(0).language
    task_emb = get_task_embs(None, [task_description])[0]
    vl_dataset = SequenceVLDataset(task_dataset, task_emb)
    
    print(f"Loaded {len(vl_dataset)} samples from {args.benchmark}")
    
    # 2. Convert to LeRobot format
    print("\n2. Converting to LeRobot format...")
    dataset_path = convert_libero_to_lerobot_format(
        vl_dataset, 
        output_dir, 
        args.benchmark
    )
    
    # 3. Create LeRobot config
    print("\n3. Creating LeRobot configuration...")
    config_path = create_lerobot_config(output_dir)
    
    print("\n=== Conversion Completed! ===")
    print(f"Dataset saved to: {dataset_path}")
    print(f"Config saved to: {config_path}")
    print("\nNow you can use the standard LeRobot training script:")
    print(f"python lerobot/scripts/train.py --config {config_path}")


if __name__ == "__main__":
    main()
