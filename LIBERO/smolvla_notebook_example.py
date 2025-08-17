"""
SmolVLA Notebook Example for LIBERO

This file contains code snippets that can be used in Jupyter notebooks
to integrate SmolVLA with LIBERO dataset.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Add LeRobot to path
sys.path.append('/home/lambs1031/SmolVLA_0812/lerobot/src')

# LIBERO imports
from libero.libero import get_libero_path, get_benchmark
from libero.lifelong.datasets import get_dataset, SequenceVLDataset
from libero.lifelong.utils import get_task_embs
from libero.lifelong.models.base_policy import BasePolicy

# SmolVLA imports
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature


# ============================================================================
# 2. SmolVLA Policy for LIBERO
# ============================================================================

class SmolVLALiberoPolicy(BasePolicy):
    """
    SmolVLA policy adapted for LIBERO dataset
    """
    
    def __init__(self, cfg, shape_meta):
        super().__init__(cfg, shape_meta)
        
        # Create SmolVLA configuration
        smolvla_config = SmolVLAConfig()
        
        # Configure input features based on LIBERO shape_meta
        smolvla_config.input_features = {}
        smolvla_config.output_features = {}
        
        # Add image features
        for name, shape in shape_meta["all_shapes"].items():
            if "rgb" in name:
                smolvla_config.input_features[name] = PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=shape,
                )
        
        # Add state features (joint states, gripper, etc.)
        state_features = []
        for name, shape in shape_meta["all_shapes"].items():
            if "rgb" not in name and "depth" not in name:
                state_features.append(shape[0])  # Assuming shape is (dim, H, W) or (dim,)
        
        if state_features:
            total_state_dim = sum(state_features)
            smolvla_config.input_features["observation.state"] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(total_state_dim,),
            )
        else:
            total_state_dim = 32  # Default state dimension
        
        # Add action features
        action_dim = shape_meta["action"]["shape"][0]
        smolvla_config.output_features["action"] = PolicyFeature(
            type=FeatureType.ACTION,
            shape=(action_dim,),
        )
        
        # Configure normalization
        smolvla_config.normalization_mapping = {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
        
        # Set dimensions
        smolvla_config.max_state_dim = max(32, total_state_dim)
        smolvla_config.max_action_dim = max(32, action_dim)
        
        # Training settings
        smolvla_config.freeze_vision_encoder = True
        smolvla_config.train_expert_only = True
        smolvla_config.train_state_proj = True
        
        # Create the SmolVLA policy
        self.smolvla_policy = SmolVLAPolicy(smolvla_config)
        
        # Store shape metadata for data preprocessing
        self.shape_meta = shape_meta
        
    def forward(self, batch):
        """
        Forward pass for training
        """
        # Preprocess batch to match SmolVLA format
        processed_batch = self._preprocess_batch(batch)
        
        # Forward through SmolVLA
        loss, loss_dict = self.smolvla_policy.forward(processed_batch)
        
        return loss, loss_dict
    
    def get_action(self, obs_dict, task_emb=None):
        """
        Get action for inference
        """
        # Preprocess observation to match SmolVLA format
        processed_batch = self._preprocess_obs(obs_dict, task_emb)
        
        # Get action from SmolVLA
        action = self.smolvla_policy.select_action(processed_batch)
        
        return action
    
    def _preprocess_batch(self, batch):
        """
        Convert LIBERO batch format to SmolVLA format
        """
        processed_batch = {}
        
        # Process images
        for name, shape in self.shape_meta["all_shapes"].items():
            if "rgb" in name and name in batch:
                # LIBERO format: (B, T, C, H, W) -> SmolVLA format: (B, C, H, W)
                img = batch[name]
                if img.ndim == 5:  # (B, T, C, H, W)
                    img = img[:, -1]  # Take last timestep
                processed_batch[name] = img
        
        # Process state
        state_features = []
        for name, shape in self.shape_meta["all_shapes"].items():
            if "rgb" not in name and "depth" not in name and name in batch:
                state = batch[name]
                if state.ndim == 3:  # (B, T, dim)
                    state = state[:, -1]  # Take last timestep
                state_features.append(state)
        
        if state_features:
            processed_batch["observation.state"] = torch.cat(state_features, dim=-1)
        
        # Process actions
        if "actions" in batch:
            processed_batch["action"] = batch["actions"]
        
        # Process task description
        if "task" in batch:
            processed_batch["task"] = batch["task"]
        
        return processed_batch
    
    def _preprocess_obs(self, obs_dict, task_emb=None):
        """
        Convert LIBERO observation format to SmolVLA format
        """
        processed_batch = {}
        
        # Process images
        for name, shape in self.shape_meta["all_shapes"].items():
            if "rgb" in name and name in obs_dict:
                img = obs_dict[name]
                if img.ndim == 4:  # (B, C, H, W)
                    processed_batch[name] = img
                elif img.ndim == 5:  # (B, T, C, H, W)
                    processed_batch[name] = img[:, -1]  # Take last timestep
        
        # Process state
        state_features = []
        for name, shape in self.shape_meta["all_shapes"].items():
            if "rgb" not in name and "depth" not in name and name in obs_dict:
                state = obs_dict[name]
                if state.ndim == 2:  # (B, dim)
                    state_features.append(state)
                elif state.ndim == 3:  # (B, T, dim)
                    state_features.append(state[:, -1])  # Take last timestep
        
        if state_features:
            processed_batch["observation.state"] = torch.cat(state_features, dim=-1)
        
        # Process task description
        if task_emb is not None:
            # Convert task embedding back to text if needed
            # This is a simplified version - you might need to implement proper conversion
            processed_batch["task"] = "robot task"  # Placeholder
        
        return processed_batch


# ============================================================================
# 3. Training Loop Example
# ============================================================================

def create_training_loop(policy, datasets, num_epochs=10, batch_size=32, lr=1e-4):
    """
    Create a simple training loop for SmolVLA with LIBERO data
    """
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)
    
    # Combine all datasets
    all_data = []
    for dataset in datasets:
        for i in range(len(dataset)):
            all_data.append(dataset[i])
    
    print(f"Total training samples: {len(all_data)}")
    
    # Training loop
    for epoch in range(num_epochs):
        policy.train()
        total_loss = 0.0
        num_batches = 0
        
        # Shuffle data
        np.random.shuffle(all_data)
        
        # Process in batches
        for i in range(0, len(all_data), batch_size):
            batch_data = all_data[i:i+batch_size]
            
            # Create batch
            batch = {}
            for key in batch_data[0].keys():
                batch[key] = torch.stack([sample[key] for sample in batch_data])
            
            # Forward pass
            optimizer.zero_grad()
            loss, loss_dict = policy.forward(batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}")
    
    return policy


# ============================================================================
# 4. Evaluation Example
# ============================================================================

def evaluate_policy(policy, test_dataset, num_samples=100):
    """
    Evaluate the trained SmolVLA policy
    """
    policy.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for i in range(0, min(len(test_dataset), num_samples)):
            sample = test_dataset[i]
            
            # Create batch with single sample
            batch = {key: value.unsqueeze(0) for key, value in sample.items()}
            
            # Forward pass
            loss, loss_dict = policy.forward(batch)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    print(f"Evaluation - Average Loss: {avg_loss:.6f}")
    
    return avg_loss


# ============================================================================
# Usage Example
# ============================================================================

def example_usage():
    """
    Example of how to use SmolVLA with LIBERO
    """
    print("=== SmolVLA + LIBERO Integration Example ===")
    
    # 1. Load LIBERO dataset
    print("1. Loading LIBERO dataset...")
    folder = get_libero_path("datasets")
    benchmark = get_benchmark("libero_object")(0)  # Use first task order
    
    # Load a single task for simplicity
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
    task_emb = get_task_embs(None, [task_description])[0]  # Simplified
    vl_dataset = SequenceVLDataset(task_dataset, task_emb)
    
    print(f"Dataset loaded: {len(vl_dataset)} samples")
    print(f"Shape metadata: {shape_meta}")
    
    # 2. Create SmolVLA policy
    print("\n2. Creating SmolVLA policy...")
    cfg = None  # You'll need to create proper config
    policy = SmolVLALiberoPolicy(cfg, shape_meta)
    
    print("Policy created successfully!")
    
    # 3. Test with sample data
    print("\n3. Testing with sample data...")
    if len(vl_dataset) > 0:
        sample = vl_dataset[0]
        print(f"Sample keys: {sample.keys()}")
        
        # Test preprocessing
        processed_batch = policy._preprocess_batch({key: value.unsqueeze(0) for key, value in sample.items()})
        print(f"Processed batch keys: {processed_batch.keys()}")
    
    print("\n=== Example completed! ===")
    print("Next steps:")
    print("- Implement proper training loop")
    print("- Add data augmentation")
    print("- Configure hyperparameters")
    print("- Add evaluation metrics")


if __name__ == "__main__":
    example_usage()
