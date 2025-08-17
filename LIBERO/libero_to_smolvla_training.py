"""
LIBERO to SmolVLA Training Script

This script converts LIBERO dataset to SmolVLA format and trains the model.
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm

# Add paths
sys.path.append('/home/lambs1031/SmolVLA_0812/lerobot/src')
sys.path.append('/home/lambs1031/SmolVLA_0812/LIBERO')

# LIBERO imports
from libero.libero import get_libero_path, get_benchmark
from libero.lifelong.datasets import get_dataset, SequenceVLDataset
from libero.lifelong.utils import get_task_embs

# SmolVLA imports
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.train import main as train_main


def convert_libero_to_smolvla_format(libero_dataset, output_dir, task_name):
    """
    Convert LIBERO dataset to SmolVLA format
    """
    print(f"Converting LIBERO dataset to SmolVLA format...")
    
    # Create output directory
    output_path = Path(output_dir) / f"libero_{task_name}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # SmolVLA dataset structure
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
                # Convert to SmolVLA image format
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
        
        # Create episode data
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
    
    # Save dataset
    dataset_info = {
        "episodes": episodes,
        "tasks": list(set(tasks)),
        "num_episodes": len(episodes),
        "action_dim": len(actions[0]) if actions else 7,
        "state_dim": len(states) if states else 32
    }
    
    # Save as JSON
    with open(output_path / "dataset.json", "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"Converted {len(episodes)} episodes to {output_path}")
    return str(output_path)


def create_smolvla_config(shape_meta, action_dim=7, state_dim=32):
    """
    Create SmolVLA configuration for LIBERO data
    """
    config = SmolVLAConfig()
    
    # Configure input features
    config.input_features = {
        "observation.images.camera": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 224, 224),
        ),
        "observation.state": PolicyFeature(
            type=FeatureType.STATE,
            shape=(state_dim,),
        )
    }
    
    # Configure output features
    config.output_features = {
        "action": PolicyFeature(
            type=FeatureType.ACTION,
            shape=(action_dim,),
        )
    }
    
    # Configure normalization
    config.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.MEAN_STD,
        "ACTION": NormalizationMode.MEAN_STD,
    }
    
    # Set dimensions
    config.max_state_dim = max(32, state_dim)
    config.max_action_dim = max(32, action_dim)
    
    # Training settings
    config.freeze_vision_encoder = True
    config.train_expert_only = True
    config.train_state_proj = True
    
    # Model settings
    config.vlm_model_name = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    config.load_vlm_weights = True
    
    return config


def train_smolvla_on_libero(dataset_path, output_dir, batch_size=32, steps=10000):
    """
    Train SmolVLA on converted LIBERO dataset
    """
    print(f"Training SmolVLA on LIBERO dataset...")
    
    # Load dataset info
    with open(Path(dataset_path) / "dataset.json", "r") as f:
        dataset_info = json.load(f)
    
    # Create config
    config = create_smolvla_config(
        shape_meta=None,
        action_dim=dataset_info["action_dim"],
        state_dim=dataset_info["state_dim"]
    )
    
    # Create policy
    policy = SmolVLAPolicy(config)
    
    # Training setup
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    
    # Training loop
    policy.train()
    losses = []
    
    for step in tqdm(range(steps), desc="Training"):
        # Sample random episode
        episode_idx = np.random.randint(0, len(dataset_info["episodes"]))
        episode = dataset_info["episodes"][episode_idx]
        
        # Create batch
        batch = {
            "observation.images.camera": torch.tensor(episode["observation"]["images"]["camera"]).unsqueeze(0),
            "observation.state": torch.tensor(episode["observation"]["state"]).unsqueeze(0),
            "action": torch.tensor(episode["action"]).unsqueeze(0),
            "task": episode["task"]
        }
        
        # Forward pass
        optimizer.zero_grad()
        loss, loss_dict = policy.forward(batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        
        # Log progress
        if step % 100 == 0:
            avg_loss = np.mean(losses[-100:])
            print(f"Step {step}, Loss: {avg_loss:.6f}")
    
    # Save model
    model_path = Path(output_dir) / "smolvla_libero_model"
    model_path.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': policy.state_dict(),
        'config': config,
        'losses': losses
    }, model_path / "checkpoint.pt")
    
    print(f"Training completed. Model saved to {model_path}")
    return losses


def visualize_training_results(losses, output_dir):
    """
    Visualize training results
    """
    print("Creating training visualizations...")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curve
    axes[0, 0].plot(losses)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # Moving average loss
    window = 100
    moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
    axes[0, 1].plot(moving_avg)
    axes[0, 1].set_title(f'Moving Average Loss (window={window})')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True)
    
    # Loss distribution
    axes[1, 0].hist(losses, bins=50, alpha=0.7)
    axes[1, 0].set_title('Loss Distribution')
    axes[1, 0].set_xlabel('Loss')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True)
    
    # Loss statistics
    loss_stats = {
        'Mean': np.mean(losses),
        'Std': np.std(losses),
        'Min': np.min(losses),
        'Max': np.max(losses),
        'Final': losses[-1]
    }
    
    stats_text = '\n'.join([f'{k}: {v:.6f}' for k, v in loss_stats.items()])
    axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                    fontsize=12, verticalalignment='center')
    axes[1, 1].set_title('Loss Statistics')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save plots
    output_path = Path(output_dir) / "training_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training visualization saved to {output_path}")
    
    # Save loss data
    loss_data = {
        'losses': losses,
        'statistics': loss_stats
    }
    
    with open(Path(output_dir) / "loss_data.json", "w") as f:
        json.dump(loss_data, f, indent=2)
    
    plt.show()


def main():
    """
    Main function to run LIBERO to SmolVLA training pipeline
    """
    parser = argparse.ArgumentParser(description="Train SmolVLA on LIBERO dataset")
    parser.add_argument("--benchmark", default="libero_object", 
                       choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    parser.add_argument("--task_order", type=int, default=0)
    parser.add_argument("--output_dir", default="./smolvla_libero_output")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--convert_only", action="store_true", 
                       help="Only convert dataset, don't train")
    
    args = parser.parse_args()
    
    print("=== LIBERO to SmolVLA Training Pipeline ===")
    print(f"Benchmark: {args.benchmark}")
    print(f"Task Order: {args.task_order}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Training Steps: {args.steps}")
    
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
    
    # 2. Convert to SmolVLA format
    print("\n2. Converting to SmolVLA format...")
    dataset_path = convert_libero_to_smolvla_format(
        vl_dataset, 
        output_dir, 
        args.benchmark
    )
    
    if args.convert_only:
        print("Dataset conversion completed. Use --convert_only=False to train.")
        return
    
    # 3. Train SmolVLA
    print("\n3. Training SmolVLA...")
    losses = train_smolvla_on_libero(
        dataset_path,
        output_dir,
        batch_size=args.batch_size,
        steps=args.steps
    )
    
    # 4. Visualize results
    print("\n4. Visualizing results...")
    visualize_training_results(losses, output_dir)
    
    print("\n=== Training Pipeline Completed! ===")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()


