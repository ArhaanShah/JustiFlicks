"""
Matrix Factorization with BPR loss.
Contains MF_BPR class, training wrapper, and prediction helpers.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple, Any

from ..data.splits import load_splits, load_item_support_table
from ..eval.helper import evaluate_model, set_item2idx
from ..utils.config import get_device


class MF_BPR(nn.Module):
    """Matrix Factorization with BPR loss."""
    
    def __init__(self, n_users: int, n_items: int, k: int = 32):
        """
        Initialize MF_BPR model.
        
        Args:
            n_users: Number of users
            n_items: Number of items
            k: Embedding dimension
        """
        super().__init__()
        self.user_emb = nn.Embedding(n_users, k)
        self.item_emb = nn.Embedding(n_items, k)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.constant_(self.user_bias.weight, 0.0)
        nn.init.constant_(self.item_bias.weight, 0.0)
    
    def score(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """Compute scores for user-item pairs."""
        ue = self.user_emb(u)
        ie = self.item_emb(i)
        ub = self.user_bias(u).squeeze(-1)
        ib = self.item_bias(i).squeeze(-1)
        return (ue * ie).sum(dim=1) + ub + ib
    
    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """Forward pass returns scores."""
        return self.score(u, i)


class PosDataset(Dataset):
    """Dataset of positive user-item pairs."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize from DataFrame with 'u' and 'i' columns.
        
        Args:
            df: DataFrame with user and item indices
        """
        self.u = torch.tensor(df.u.values, dtype=torch.long)
        self.i = torch.tensor(df.i.values, dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.u)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.u[idx], self.i[idx]


def predict_topk(model: MF_BPR, user_id: int, user2idx: Dict[int, int], 
                 idx2item: Dict[int, int], device: str, k: int = 20) -> List[int]:
    """
    Predict top-k items for a user.
    
    Args:
        model: MF_BPR model
        user_id: User ID
        user2idx: User to index mapping
        idx2item: Index to item mapping
        device: Device to use
        k: Number of items to return
        
    Returns:
        List of top-k item IDs
    """
    if user_id not in user2idx:
        return []
    
    u = torch.tensor([user2idx[user_id]], device=device)
    i = torch.arange(len(idx2item), device=device)
    
    with torch.no_grad():
        scores = model(u.repeat(len(i)), i).cpu().numpy()
    
    topk_idx = np.argsort(-scores)[:k]
    return [idx2item[j] for j in topk_idx]


def train_mf(cfg: dict) -> None:
    """
    Train MF_BPR model and save artifacts.
    
    Args:
        cfg: Configuration dictionary
    """
    device = get_device()
    seed = cfg["seed"]
    rating_threshold = cfg["rating_threshold"]
    mf_cfg = cfg.get("MF", {})
    
    # Load splits
    train, val, test = load_splits(cfg)
    item_support_table = load_item_support_table(cfg)
    
    # Prepare training data
    train_cf = train[train.rating >= rating_threshold][["userId", "movieId"]].copy()
    
    # Load movie links for catalog restriction
    links = pd.read_parquet(os.path.join(cfg["data_root"], "movieLinks.parquet"))
    cf_movie_set = set(links.movieId.astype(int))
    
    user_ids = train_cf.userId.unique()
    item_ids = train_cf.movieId.unique()
    item_ids = [m for m in item_ids if m in cf_movie_set]
    
    item2idx = {m: i for i, m in enumerate(item_ids)}
    idx2item = {i: m for m, i in item2idx.items()}
    user2idx = {u: i for i, u in enumerate(user_ids)}
    
    # Set global item2idx for evaluation
    set_item2idx(item2idx)
    
    train_cf["u"] = train_cf.userId.map(user2idx)
    train_cf["i"] = train_cf.movieId.map(item2idx)
    train_cf = train_cf.dropna(subset=["u", "i"])
    train_cf["u"] = train_cf["u"].astype(int)
    train_cf["i"] = train_cf["i"].astype(int)
    
    # Popularity-weighted negative sampling
    item_counts = pd.Series(train_cf.i).value_counts().reindex(range(len(item2idx))).fillna(0).astype(float).values
    alpha = mf_cfg.get("alpha", 0.75)
    probs = (item_counts ** alpha)
    probs = probs / probs.sum()
    
    # Validation subset
    val_users_all = val.userId.unique()
    rng = np.random.RandomState(seed)
    val_users_small = set(rng.choice(val_users_all, size=min(10000, len(val_users_all)), replace=False))
    
    # Create dataloader
    ds = PosDataset(train_cf)
    batch_size = mf_cfg.get("batch_size", 4096)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Initialize model
    model = MF_BPR(len(user2idx), len(item2idx), k=mf_cfg.get("embedding_dim", 32)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=mf_cfg.get("learning_rate", 1e-3))
    
    lambda_item = mf_cfg.get("lambda_item", 1e-4)
    lambda_bias = mf_cfg.get("lambda_bias", 1e-6)
    max_epochs = mf_cfg.get("max_epochs", 30)
    patience = mf_cfg.get("patience", 3)
    min_delta = mf_cfg.get("min_delta", 1e-4)
    
    best_metric = -np.inf
    epochs_no_improve = 0
    best_state = None
    
    for epoch in range(max_epochs):
        model.train()
        losses = []
        
        for u_batch, i_batch in loader:
            bs = len(u_batch)
            neg_idx = rng.choice(len(item2idx), size=bs, p=probs)
            u = u_batch.to(device)
            pos = i_batch.to(device)
            neg = torch.tensor(neg_idx, dtype=torch.long, device=device)
            
            pos_score = model.score(u, pos)
            neg_score = model.score(u, neg)
            
            loss_bpr = -F.logsigmoid(pos_score - neg_score).mean()
            
            # Regularization
            pos_ie = model.item_emb(pos)
            neg_ie = model.item_emb(neg)
            reg_item = (pos_ie.pow(2).sum() + neg_ie.pow(2).sum()) / bs
            
            pos_b = model.item_bias(pos).squeeze(-1)
            neg_b = model.item_bias(neg).squeeze(-1)
            reg_bias = (pos_b.pow(2).sum() + neg_b.pow(2).sum()) / bs
            
            loss = loss_bpr + lambda_item * reg_item + lambda_bias * reg_bias
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions = {}
            for uid, g in val.groupby("userId"):
                if uid in val_users_small and uid in user2idx:
                    val_predictions[int(uid)] = predict_topk(model, uid, user2idx, idx2item, device, k=10)
            
            val_ground_truth = {}
            for uid, g in val.groupby("userId"):
                if uid in val_users_small:
                    val_ground_truth[int(uid)] = {int(r.movieId): float(r.rating) for _, r in g.iterrows()}
            
            val_metrics, _ = evaluate_model(val_predictions, val_ground_truth, item_support_table, ks=[10], seed=seed)
            current_metric = val_metrics["ndcg@10"]
            ci_low = val_metrics.get("ndcg@10_ci_low")
            ci_high = val_metrics.get("ndcg@10_ci_high")
        
        print(f"epoch={epoch} loss={np.mean(losses):.4f} val_ndcg@10={current_metric:.6f} ci=({ci_low:.6f},{ci_high:.6f})")
        
        if current_metric > best_metric + min_delta:
            best_metric = current_metric
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print("early stopping triggered")
            break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    model.eval()
    
    # Save artifacts (excluding metrics/slices)
    save_mf_artifacts(model, user2idx, item2idx, cfg)


def save_mf_artifacts(model: MF_BPR, user2idx: Dict[int, int], item2idx: Dict[int, int], cfg: dict):
    """
    Save MF model artifacts.
    
    Args:
        model: Trained MF_BPR model
        user2idx: User to index mapping
        item2idx: Item to index mapping
        cfg: Configuration dictionary
    """
    mf_dir = os.path.join(cfg["artifact_root"], "MF")
    os.makedirs(mf_dir, exist_ok=True)
    
    # Extract embeddings and biases
    user_emb = model.user_emb.weight.detach().cpu().numpy()
    item_emb = model.item_emb.weight.detach().cpu().numpy()
    user_bias = model.user_bias.weight.detach().cpu().numpy().reshape(-1)
    item_bias = model.item_bias.weight.detach().cpu().numpy().reshape(-1)
    
    # Create index DataFrames
    user_index = pd.DataFrame({
        "userId": list(user2idx.keys()),
        "user_idx": list(user2idx.values()),
        "user_bias": user_bias
    })
    
    item_index = pd.DataFrame({
        "movieId": list(item2idx.keys()),
        "item_idx": list(item2idx.values()),
        "item_bias": item_bias
    })
    
    # Save embeddings
    np.save(os.path.join(mf_dir, "user_embeddings.npy"), user_emb)
    np.save(os.path.join(mf_dir, "item_embeddings.npy"), item_emb)
    np.save(os.path.join(mf_dir, "user_bias.npy"), user_bias)
    np.save(os.path.join(mf_dir, "item_bias.npy"), item_bias)
    
    # Save indices
    user_index.to_csv(os.path.join(mf_dir, "user_index.csv"), index=False)
    item_index.to_csv(os.path.join(mf_dir, "item_index.csv"), index=False)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(mf_dir, "mf_bpr_model.pt"))
    
    # Save metadata
    mf_cfg = cfg.get("MF", {})
    mf_metadata = {
        "model_type": "MF_BPR",
        "embedding_dim": int(user_emb.shape[1]),
        "n_users": int(user_emb.shape[0]),
        "n_items": int(item_emb.shape[0]),
        "rating_threshold": cfg["rating_threshold"],
        "loss": "BPR",
        "negative_sampling": "popularity_aware",
        "regularization": {"item": mf_cfg.get("lambda_item", 1e-4), "bias": mf_cfg.get("lambda_bias", 1e-6)},
        "early_stopping": {"patience": mf_cfg.get("patience", 3), "min_delta": mf_cfg.get("min_delta", 1e-4)},
        "seed": cfg["seed"]
    }
    
    with open(os.path.join(mf_dir, "mf_metadata.json"), "w") as f:
        json.dump(mf_metadata, f, indent=2)
    
    print(f"MF artifacts saved to {mf_dir}")
