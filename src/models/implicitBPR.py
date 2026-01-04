"""
Implicit BPR model using the implicit library.
Contains training wrapper and prediction helpers.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import scipy.sparse as sp
from implicit.bpr import BayesianPersonalizedRanking

from ..data.splits import load_splits, load_item_support_table
from ..eval.helper import evaluate_model, set_item2idx


def check_implicit_installed():
    """Check if implicit library is installed."""
    try:
        import implicit
        return True
    except ImportError:
        return False


def train_implicit_bpr(cfg: dict) -> None:
    """
    Train implicit BPR model and save artifacts.
    
    Args:
        cfg: Configuration dictionary
        
    Raises:
        ImportError: If implicit library is not installed
    """
    if not check_implicit_installed():
        raise ImportError(
            "The 'implicit' library is not installed. "
            "Please install it with: pip install implicit"
        )
    
    seed = cfg["seed"]
    rating_threshold = cfg["rating_threshold"]
    implicit_cfg = cfg.get("implicitBPR", {})
    
    np.random.seed(seed)
    
    # Load splits
    train, val, test = load_splits(cfg)
    item_support_table = load_item_support_table(cfg)
    
    # Prepare training data
    train_cf = train[train.rating >= rating_threshold][["userId", "movieId"]].copy()
    
    # Load movie links for catalog restriction
    links = pd.read_parquet(os.path.join(cfg["data_root"], "movieLinks.parquet"))
    cf_movie_set = set(links.movieId.astype(int))
    
    train_cf = train_cf[train_cf.movieId.isin(cf_movie_set)]
    
    user_ids = train_cf.userId.unique()
    item_ids = train_cf.movieId.unique()
    
    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {m: i for i, m in enumerate(item_ids)}
    idx2item = {i: m for m, i in item2idx.items()}
    
    # Set global item2idx for evaluation
    set_item2idx(item2idx)
    
    train_cf["u"] = train_cf.userId.map(user2idx)
    train_cf["i"] = train_cf.movieId.map(item2idx)
    
    n_users = len(user2idx)
    n_items = len(item2idx)
    
    # Build sparse matrix
    rows = train_cf.u.values
    cols = train_cf.i.values
    data = np.ones(len(train_cf), dtype=np.float32)
    
    user_item = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(n_users, n_items)
    ).tocsr()
    
    # Initialize model
    model = BayesianPersonalizedRanking(
        factors=implicit_cfg.get("factors", 64),
        learning_rate=implicit_cfg.get("learning_rate", 0.01),
        regularization=implicit_cfg.get("regularization", 1e-4),
        iterations=1,
        random_state=seed,
        verify_negative_samples=True
    )
    
    # Validation subset
    rng = np.random.RandomState(seed)
    val_users_all = val.userId.unique()
    val_users_small = set(
        rng.choice(val_users_all, size=min(10000, len(val_users_all)), replace=False)
    )
    
    def predict_topk_implicit(model, user_id, k=10):
        if user_id not in user2idx:
            return []
        uidx = user2idx[user_id]
        ids, scores = model.recommend(
            uidx,
            user_item,
            N=k,
            filter_already_liked_items=False
        )
        return [idx2item[i] for i in ids]
    
    max_epochs = implicit_cfg.get("max_epochs", 30)
    patience = implicit_cfg.get("patience", 3)
    min_delta = implicit_cfg.get("min_delta", 1e-4)
    
    best_metric = -np.inf
    epochs_no_improve = 0
    best_state = None
    
    for epoch in range(max_epochs):
        model.fit(user_item, show_progress=False)
        
        # Validation
        val_predictions = {}
        for uid, g in val.groupby("userId"):
            if uid in val_users_small and uid in user2idx:
                val_predictions[int(uid)] = predict_topk_implicit(model, int(uid), k=10)
        
        val_ground_truth = {}
        for uid, g in val.groupby("userId"):
            if uid in val_users_small:
                val_ground_truth[int(uid)] = {int(r.movieId): float(r.rating) for _, r in g.iterrows()}
        
        val_metrics, _ = evaluate_model(val_predictions, val_ground_truth, item_support_table, ks=[10], seed=seed)
        current_metric = val_metrics["ndcg@10"]
        ci_low = val_metrics.get("ndcg@10_ci_low")
        ci_high = val_metrics.get("ndcg@10_ci_high")
        
        print(f"epoch={epoch} val_ndcg@10={current_metric:.6f} ci=({ci_low:.6f},{ci_high:.6f})")
        
        if current_metric > best_metric + min_delta:
            best_metric = current_metric
            if hasattr(model, "to_cpu"):
                model = model.to_cpu()
            best_state = model
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print("early stopping triggered")
            break
    
    if best_state is not None:
        model = best_state
    
    # Save artifacts
    save_implicit_artifacts(model, user2idx, item2idx, cfg)


def save_implicit_artifacts(model, user2idx: Dict[int, int], item2idx: Dict[int, int], cfg: dict):
    """
    Save implicit BPR model artifacts.
    
    Args:
        model: Trained implicit BPR model
        user2idx: User to index mapping
        item2idx: Item to index mapping
        cfg: Configuration dictionary
    """
    implicit_dir = os.path.join(cfg["artifact_root"], "implicitBPR")
    os.makedirs(implicit_dir, exist_ok=True)
    
    # Ensure model is on CPU
    if hasattr(model, "to_cpu"):
        model = model.to_cpu()
    
    # Extract embeddings
    user_emb = model.user_factors
    item_emb = model.item_factors
    
    # Create index DataFrames
    user_index = pd.DataFrame({
        "userId": list(user2idx.keys()),
        "user_idx": list(user2idx.values())
    })
    
    item_index = pd.DataFrame({
        "movieId": list(item2idx.keys()),
        "item_idx": list(item2idx.values())
    })
    
    # Save embeddings
    np.save(os.path.join(implicit_dir, "user_embeddings.npy"), user_emb)
    np.save(os.path.join(implicit_dir, "item_embeddings.npy"), item_emb)
    
    # Save indices
    user_index.to_csv(os.path.join(implicit_dir, "user_index.csv"), index=False)
    item_index.to_csv(os.path.join(implicit_dir, "item_index.csv"), index=False)
    
    # Save model
    model.save(os.path.join(implicit_dir, "implicit_bpr_model.npz"))
    
    # Save metadata
    implicit_cfg = cfg.get("implicitBPR", {})
    implicit_metadata = {
        "model_type": "implicit_BPR",
        "embedding_dim": int(user_emb.shape[1]),
        "n_users": int(user_emb.shape[0]),
        "n_items": int(item_emb.shape[0]),
        "rating_threshold": cfg["rating_threshold"],
        "loss": "BPR",
        "negative_sampling": "implicit_internal",
        "regularization": "implicit_default",
        "early_stopping": {"patience": implicit_cfg.get("patience", 3), "min_delta": implicit_cfg.get("min_delta", 1e-4)},
        "seed": cfg["seed"]
    }
    
    with open(os.path.join(implicit_dir, "implicit_metadata.json"), "w") as f:
        json.dump(implicit_metadata, f, indent=2)
    
    print(f"Implicit BPR artifacts saved to {implicit_dir}")
