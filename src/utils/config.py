"""
Configuration loading and seed setting utilities.
Loads YAML configs and exposes CFG dict with sensible defaults.
"""

import os
import random
import yaml
import numpy as np
import torch

# Global CFG dict, populated by load_config
CFG = {}


def load_config(path: str) -> dict:
    """
    Load a YAML configuration file and return as a Python dict.
    Sets sensible defaults for missing keys.
    Also sets global CFG and seeds.
    
    Args:
        path: Path to the YAML config file
        
    Returns:
        dict: Configuration dictionary with defaults applied
    """
    global CFG
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "r") as f:
        config = yaml.safe_load(f) or {}
    
    # Apply defaults for main config keys
    defaults = {
        "data_root": "data",
        "embedding_root": "embeddings",
        "artifact_root": "artifacts",
        "rating_threshold": 4.5,
        "k_values": [5, 10, 20],
        "seed": 34,
        "n_test": 5,
        "n_val": 5,
        "min_pos": 1,
    }
    
    for key, default_val in defaults.items():
        if key not in config:
            config[key] = default_val
    
    # Model-specific defaults
    mf_defaults = {
        "embedding_dim": 32,
        "batch_size": 4096,
        "learning_rate": 1e-3,
        "lambda_item": 1e-4,
        "lambda_bias": 1e-6,
        "max_epochs": 30,
        "patience": 3,
        "min_delta": 1e-4,
        "alpha": 0.75,
    }
    if "MF" not in config:
        config["MF"] = {}
    for key, val in mf_defaults.items():
        if key not in config["MF"]:
            config["MF"][key] = val
    
    implicit_defaults = {
        "factors": 64,
        "learning_rate": 0.01,
        "regularization": 1e-4,
        "max_epochs": 30,
        "patience": 3,
        "min_delta": 1e-4,
        "alpha": 0.75,
    }
    if "implicitBPR" not in config:
        config["implicitBPR"] = {}
    for key, val in implicit_defaults.items():
        if key not in config["implicitBPR"]:
            config["implicitBPR"][key] = val
    
    two_tower_defaults = {
        "hidden_dims": [256, 128],
        "emb_dim": 64,
        "dropout": 0.2,
        "batch_size": 4096,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "temperature": 0.07,
        "max_epochs": 100,
        "patience": 30,
        "min_delta": 1e-4,
    }
    if "twoTower" not in config:
        config["twoTower"] = {}
    for key, val in two_tower_defaults.items():
        if key not in config["twoTower"]:
            config["twoTower"][key] = val
    
    CFG = config
    
    # Set seeds from config
    set_seeds(config.get("seed", 34))
    
    return config


def set_seeds(seed: int):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> str:
    """Get the appropriate device (cuda or cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"
