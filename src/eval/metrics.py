"""
Evaluation metrics module.
Contains test_eval function to run final evaluation on test set and save results.
"""

import os
import json
import pandas as pd
from typing import Dict, List, Any, Callable, Tuple, Optional
from .helper import evaluate_model
from ..data.splits import load_splits, load_item_support_table


def test_eval(cfg: dict, 
              predict_fn: Callable[[int, int], List[int]], 
              model_name: str,
              seed: int = 34) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
    """
    Run final evaluation on test set and save metrics/slices.
    
    Args:
        cfg: Configuration dictionary
        predict_fn: Function taking (user_id, k) and returning ordered list of item IDs
        model_name: Name of the model (artifacts directory name)
        seed: Random seed
        
    Returns:
        Tuple of (metrics dict, slices dict)
    """
    print(f"\nStarting final test evaluation for {model_name}...")
    
    # Load test data
    _, _, test = load_splits(cfg)
    item_support_table = load_item_support_table(cfg)
    
    # Construct ground truth
    ground_truth = {}
    test_users = test.userId.unique()
    for uid, g in test.groupby("userId"):
        ground_truth[int(uid)] = {int(r.movieId): float(r.rating) for _, r in g.iterrows()}
    
    # Generate predictions
    predictions = {}
    k_max = max(cfg["k_values"])
    
    # We only predict for users in the test set
    for uid in ground_truth.keys():
        try:
            # predict_fn should handle if user is unseen (return empty list)
            preds = predict_fn(int(uid), k_max)
            predictions[int(uid)] = preds
        except Exception as e:
            # Fallback for errors
            predictions[int(uid)] = []
            
    # Compute metrics
    metrics, slices = evaluate_model(
        predictions, 
        ground_truth, 
        item_support_table, 
        ks=cfg["k_values"], 
        seed=seed
    )
    
    # Save results
    artifact_dir = os.path.join(cfg["artifact_root"], model_name)
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Save metrics
    metrics_path = os.path.join(artifact_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    # Save slices
    slices_path = os.path.join(artifact_dir, "slices_long.csv")
    slices_long = pd.concat(
        [df.assign(slice_type=name) for name, df in slices.items()],
        ignore_index=True
    )
    slices_long.to_csv(slices_path, index=False)
    
    print(f"Evaluation complete. Metrics saved to {metrics_path}")
    print(json.dumps(metrics, indent=2))
    
    return metrics, slices
