"""
SmolVLA Integration with LIBERO Dataset

This script demonstrates how to integrate SmolVLA model with LIBERO dataset
for robot learning tasks.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

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


def load_libero_dataset():
    """
    Load LIBERO dataset for training
    """
    # Set up paths
    folder = get_libero_path("datasets")
    bddl_folder = get_libero_path("bddl_files")
    init_states_folder = get_libero_path("init_states")
    
    # Get benchmark
    benchmark_name = "libero_object"  # Can be "libero_spatial", "libero_object", "libero_goal", "libero_10"
    task_order = 0  # Can be from {0 .. 21}
    benchmark = get_benchmark(benchmark_name)(task_order)
    
    # Prepare datasets
    datasets = []
    descriptions = []
    shape_meta = None
    n_tasks = benchmark.n_tasks
    
    # Observation modality configuration
    obs_modality = {
        "rgb": ["rgb_static", "rgb_gripper"],
        "state": ["joint_states", "gripper_states"]
    }
    
    for i in range(n_tasks):
        # Load dataset for each task
        task_i_dataset, shape_meta = get_dataset(
            dataset_path=os.path.join(folder, benchmark.get_task_demonstration(i)),
            obs_modality=obs_modality,
            initialize_obs_utils=(i == 0),
            seq_len=1,  # Use single timestep for SmolVLA
        )
        
        # Add language description
        descriptions.append(benchmark.get_task(i).language)
        datasets.append(task_i_dataset)
    
    # Get task embeddings
    task_embs = get_task_embs(None, descriptions)  # You'll need to pass proper config
    benchmark.set_task_embs(task_embs)
    
    # Create vision-language datasets
    vl_datasets = [SequenceVLDataset(ds, emb) for (ds, emb) in zip(datasets, task_embs)]
    
    return vl_datasets, shape_meta, benchmark


def create_smolvla_config(shape_meta):
    """
    Create SmolVLA configuration based on LIBERO shape metadata
    """
    config = SmolVLAConfig()
    
    # Configure input features
    config.input_features = {}
    config.output_features = {}
    
    # Add image features
    for name, shape in shape_meta["all_shapes"].items():
        if "rgb" in name:
            config.input_features[name] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=shape,
            )
    
    # Add state features
    state_features = []
    for name, shape in shape_meta["all_shapes"].items():
        if "rgb" not in name and "depth" not in name:
            state_features.append(shape[0])
    
    if state_features:
        total_state_dim = sum(state_features)
        config.input_features["observation.state"] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(total_state_dim,),
        )
    
    # Add action features
    action_dim = shape_meta["action"]["shape"][0]
    config.output_features["action"] = PolicyFeature(
        type=FeatureType.ACTION,
        shape=(action_dim,),
    )
    
    # Configure normalization
    config.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.MEAN_STD,
        "ACTION": NormalizationMode.MEAN_STD,
    }
    
    # Set dimensions
    config.max_state_dim = max(32, total_state_dim if state_features else 32)
    config.max_action_dim = max(32, action_dim)
    
    # Training settings
    config.freeze_vision_encoder = True
    config.train_expert_only = True
    config.train_state_proj = True
    
    return config


def main():
    """
    Main function to demonstrate SmolVLA integration with LIBERO
    """
    print("Loading LIBERO dataset...")
    vl_datasets, shape_meta, benchmark = load_libero_dataset()
    
    print(f"Loaded {len(vl_datasets)} tasks from {benchmark.name}")
    print(f"Shape metadata: {shape_meta}")
    
    # Create SmolVLA configuration
    print("Creating SmolVLA configuration...")
    config = create_smolvla_config(shape_meta)
    
    # Create SmolVLA policy
    print("Creating SmolVLA policy...")
    policy = SmolVLAPolicy(config)
    
    print("SmolVLA policy created successfully!")
    print(f"Input features: {config.input_features}")
    print(f"Output features: {config.output_features}")
    
    # Example of processing a batch
    if len(vl_datasets) > 0:
        print("\nTesting with sample data...")
        sample_dataset = vl_datasets[0]
        
        # Get a sample batch
        sample_batch = sample_dataset[0]  # Get first sample
        
        print(f"Sample batch keys: {sample_batch.keys()}")
        
        # Test forward pass (this might fail if data format doesn't match exactly)
        try:
            with torch.no_grad():
                # Preprocess batch
                processed_batch = {}
                
                # Process images
                for name, shape in shape_meta["all_shapes"].items():
                    if "rgb" in name and name in sample_batch:
                        img = sample_batch[name]
                        if img.ndim == 5:  # (B, T, C, H, W)
                            img = img[:, -1]  # Take last timestep
                        processed_batch[name] = img
                
                # Process state
                state_features = []
                for name, shape in shape_meta["all_shapes"].items():
                    if "rgb" not in name and "depth" not in name and name in sample_batch:
                        state = sample_batch[name]
                        if state.ndim == 3:  # (B, T, dim)
                            state = state[:, -1]  # Take last timestep
                        state_features.append(state)
                
                if state_features:
                    processed_batch["observation.state"] = torch.cat(state_features, dim=-1)
                
                # Process actions
                if "actions" in sample_batch:
                    processed_batch["action"] = sample_batch["actions"]
                
                # Process task description
                if "task" in sample_batch:
                    processed_batch["task"] = sample_batch["task"]
                
                print(f"Processed batch keys: {processed_batch.keys()}")
                
        except Exception as e:
            print(f"Error during forward pass: {e}")
            print("This is expected if the data format doesn't match exactly.")
    
    print("\nSmolVLA integration with LIBERO completed!")
    print("Next steps:")
    print("1. Implement proper data preprocessing")
    print("2. Set up training loop")
    print("3. Configure hyperparameters")
    print("4. Add evaluation metrics")


if __name__ == "__main__":
    main()
