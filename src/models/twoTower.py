"""
Two-Tower content-based model with InfoNCE loss.
Contains TwoTower model class, training wrapper, and prediction helpers.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
from collections import defaultdict
from typing import Dict, List, Any, Tuple

from ..data.splits import load_splits, load_item_support_table
from ..eval.helper import evaluate_model, set_item2idx
from ..utils.config import get_device


class TwoTower(nn.Module):
    """Two-Tower neural network for user-item matching."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128], 
                 emb_dim: int = 64, drop: float = 0.2):
        """
        Initialize Two-Tower model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            emb_dim: Output embedding dimension
            drop: Dropout probability
        """
        super().__init__()
        
        def make_mlp(in_dim, layers):
            seq = []
            d = in_dim
            for h in layers:
                seq.append(nn.Linear(d, h))
                seq.append(nn.ReLU())
                seq.append(nn.Dropout(drop))
                d = h
            seq.append(nn.Linear(d, emb_dim))
            return nn.Sequential(*seq)
        
        self.item_net = make_mlp(input_dim, hidden_dims)
        self.user_net = make_mlp(input_dim, hidden_dims)
    
    def forward_user(self, user_feat: torch.Tensor) -> torch.Tensor:
        """Encode user features to embeddings."""
        u = self.user_net(user_feat)
        return F.normalize(u, p=2, dim=1)
    
    def forward_item(self, item_feat: torch.Tensor) -> torch.Tensor:
        """Encode item features to embeddings."""
        i = self.item_net(item_feat)
        return F.normalize(i, p=2, dim=1)


class PosPairDataset(data.Dataset):
    """Dataset of positive user-item pairs."""
    
    def __init__(self, pairs: np.ndarray):
        """
        Initialize from pairs array.
        
        Args:
            pairs: Nx2 array of (user_idx, item_idx) pairs
        """
        self.pairs = pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[int, int]:
        return int(self.pairs[idx, 0]), int(self.pairs[idx, 1])


def info_nce_loss(u_emb: torch.Tensor, pos_i_emb: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
    """
    Compute InfoNCE loss.
    
    Args:
        u_emb: User embeddings (B x D)
        pos_i_emb: Positive item embeddings (B x D)
        temp: Temperature parameter
        
    Returns:
        InfoNCE loss
    """
    logits = torch.matmul(u_emb, pos_i_emb.T) / temp
    labels = torch.arange(logits.size(0), device=logits.device)
    loss = F.cross_entropy(logits, labels)
    return loss

    
def train_two_tower(cfg: dict) -> None:
    """
    Train Two-Tower model and save artifacts.
    
    Args:
        cfg: Configuration dictionary
    """
    device=get_device()
    seed=cfg["seed"]
    rating_threshold=cfg["rating_threshold"]
    tt_cfg=cfg.get("twoTower",{})
    np.random.seed(seed)
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    train,val,test=load_splits(cfg)
    item_support_table=load_item_support_table(cfg)
    item_features_path=os.path.join(cfg["embedding_root"],"item","item_features.parquet")
    if not os.path.exists(item_features_path):
        raise FileNotFoundError(f"Item features not found at {item_features_path}. Please run build_embeddings first.")
    item_features=pd.read_parquet(item_features_path)
    train_cf=train[train.rating>=rating_threshold][["userId","movieId"]].copy()
    links=pd.read_parquet(os.path.join(cfg["data_root"],"movieLinks.parquet"))
    cf_movie_set=set(links.movieId.astype(int))
    user_ids=train_cf.userId.unique()
    item_ids=train_cf.movieId.unique()
    item_ids=[int(m) for m in item_ids if int(m) in cf_movie_set]
    item2idx={m:i for i,m in enumerate(item_ids)}
    idx2item={i:m for m,i in item2idx.items()}
    set_item2idx(item2idx)
    items_df=item_features[item_features.movieId.isin(item2idx.keys())].copy()
    items_df.movieId=items_df.movieId.astype(int)
    items_df=items_df.merge(item_support_table[["movieId","support","num_votes_imdb","release_year"]],on="movieId",how="left")
    numeric_cols=["support","num_votes_imdb","release_year"]
    for c in numeric_cols:
        items_df[c]=items_df[c].fillna(0).astype(np.float32)
        if c in ("support","num_votes_imdb"):
            items_df[c]=np.log1p(items_df[c])
        denom=items_df[c].max()-items_df[c].min()
        if denom>0:
            items_df[c]=(items_df[c]-items_df[c].min())/denom
    exclude={"movieId","imdbId","support","num_votes_imdb","release_year"}
    onehot_cols=[c for c in items_df.columns if c not in exclude]
    onehot_cols=sorted(onehot_cols)
    feat_dim=len(onehot_cols)+len(numeric_cols)
    n_items=len(item2idx)
    X_item=np.zeros((n_items,feat_dim),dtype=np.float32)
    if len(items_df)>0:
        onehot_arr=items_df[onehot_cols].astype(np.float32).to_numpy() if len(onehot_cols)>0 else np.zeros((len(items_df),0),dtype=np.float32)
        numeric_arr=items_df[numeric_cols].astype(np.float32).to_numpy()
        for i,row_idx in enumerate(items_df.index):
            mid=int(items_df.at[row_idx,"movieId"])
            if mid in item2idx:
                idx=item2idx[mid]
                if onehot_arr.size>0:
                    X_item[idx,:len(onehot_cols)]=onehot_arr[i]
                X_item[idx,len(onehot_cols):]=numeric_arr[i]
    train_pos=train[train.rating>=rating_threshold][["userId","movieId","datetime"]].copy()
    train_pos=train_pos[train_pos.movieId.isin(item2idx.keys())].copy()
    train_pos["item_idx"]=train_pos.movieId.map(item2idx).astype(int)
    user_ids_list=train_pos.userId.unique().tolist()
    user2idx={u:i for i,u in enumerate(user_ids_list)}
    idx2user={i:u for u,i in user2idx.items()}
    train_pos["u"]=train_pos.userId.map(user2idx).astype(int)
    n_users=len(user2idx)
    user_profiles=np.zeros((n_users,X_item.shape[1]),dtype=np.float32)
    user_counts=np.zeros(n_users,dtype=np.int32)
    if len(train_pos)>0:
        us=train_pos.u.values.astype(np.int64)
        its=train_pos.item_idx.values.astype(np.int64)
        np.add.at(user_counts,us,1)
        sums=np.zeros((n_users,X_item.shape[1]),dtype=np.float32)
        np.add.at(sums,us,X_item[its])
        nonzero_mask=user_counts>0
        user_profiles[nonzero_mask]=sums[nonzero_mask]/user_counts[nonzero_mask,None]
    nonzero_mask=(user_counts>0)
    if nonzero_mask.sum()>0:
        global_mean=user_profiles[nonzero_mask].mean(axis=0)
    else:
        global_mean=np.zeros(X_item.shape[1],dtype=np.float32)
    user_profiles[~nonzero_mask]=global_mean
    if len(train_pos)>0:
        pairs=np.stack([train_pos.u.values.astype(np.int64),train_pos.item_idx.values.astype(np.int64)],axis=1)
    else:
        pairs=np.zeros((0,2),dtype=np.int64)
    pairs_u=torch.from_numpy(pairs[:,0]).long()
    pairs_i=torch.from_numpy(pairs[:,1]).long()
    batch_size=tt_cfg.get("batch_size",512)
    num_workers=tt_cfg.get("num_workers",max(0, min(4, (os.cpu_count() or 1)-1)))
    dataset=torch.utils.data.TensorDataset(pairs_u,pairs_i)
    g=torch.Generator()
    g.manual_seed(seed)
    loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True,drop_last=True,generator=g,num_workers=num_workers,pin_memory=False)
    input_dim=X_item.shape[1]
    model=TwoTower(input_dim=input_dim,hidden_dims=tt_cfg.get("hidden_dims",[256,128]),emb_dim=tt_cfg.get("emb_dim",64),drop=tt_cfg.get("dropout",0.2))
    model.to(device)
    opt=torch.optim.Adam(model.parameters(),lr=tt_cfg.get("learning_rate",1e-3),weight_decay=tt_cfg.get("weight_decay",1e-5))
    item_feat_tensor=torch.tensor(X_item,dtype=torch.float32,device=device)
    user_profiles_tensor=torch.tensor(user_profiles,dtype=torch.float32,device=device)
    rng=np.random.RandomState(seed)
    val_users_all=val.userId.unique()
    val_users_small=set(rng.choice(val_users_all,size=min(10000,len(val_users_all)),replace=False))
    val_user_list=[u for u in val_users_small if u in user2idx]
    val_user_idx=[user2idx[u] for u in val_user_list]
    val_ground_truth={}
    for uid,gdf in val.groupby("userId"):
        if uid in val_users_small:
            val_ground_truth[int(uid)]={int(r.movieId):float(r.rating) for _,r in gdf.iterrows()}
    max_epochs=tt_cfg.get("max_epochs",10)
    patience=tt_cfg.get("patience",3)
    min_delta=tt_cfg.get("min_delta",1e-4)
    temperature=tt_cfg.get("temperature",0.07)
    best_metric=-np.inf
    epochs_no_improve=0
    best_state=None
    for epoch in range(max_epochs):
        model.train()
        losses=[]
        for batch in loader:
            u_batch=batch[0].to(device)
            i_batch=batch[1].to(device)
            batch_user_feats=user_profiles_tensor[u_batch]
            batch_item_feats=item_feat_tensor[i_batch]
            user_embs=model.forward_user(batch_user_feats)
            pos_item_embs=model.forward_item(batch_item_feats)
            loss=info_nce_loss(user_embs,pos_item_embs,temp=temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            item_embs=model.forward_item(item_feat_tensor).cpu().numpy()
            if len(val_user_idx)>0:
                val_user_embs=model.forward_user(user_profiles_tensor[val_user_idx]).cpu().numpy()
            else:
                val_user_embs=np.zeros((0,item_embs.shape[1]),dtype=np.float32)
            val_predictions={}
            k_eval=max(cfg["k_values"])
            for idx,uid in enumerate(val_user_list):
                scores=val_user_embs[idx].dot(item_embs.T)
                topk=np.argpartition(-scores,k_eval-1)[:k_eval]
                topk=topk[np.argsort(-scores[topk])]
                val_predictions[int(uid)]=[idx2item[int(t)] for t in topk]
            val_metrics,_=evaluate_model(val_predictions,val_ground_truth,item_support_table,ks=[10],seed=seed)
            current_metric=val_metrics["ndcg@10"]
            ci_low=val_metrics.get("ndcg@10_ci_low")
            ci_high=val_metrics.get("ndcg@10_ci_high")
        print(f"epoch={epoch} loss={np.mean(losses):.4f} val_ndcg@10={current_metric:.6f} ci=({ci_low:.6f},{ci_high:.6f})")
        if current_metric>best_metric+min_delta:
            best_metric=current_metric
            best_state={"model":model.cpu().state_dict()}
            model.to(device)
            epochs_no_improve=0
        else:
            epochs_no_improve+=1
        if epochs_no_improve>=patience:
            print("early stopping triggered")
            break
    if best_state is not None:
        model.load_state_dict(best_state["model"])
    model.eval()
    save_two_tower_artifacts(model,user2idx,item2idx,user_profiles,X_item,cfg,patience,min_delta)
    

def save_two_tower_artifacts(model: TwoTower, user2idx: Dict[int, int], item2idx: Dict[int, int],
                             user_profiles: np.ndarray, X_item: np.ndarray,
                             cfg: dict, patience: int, min_delta: float):
    """
    Save Two-Tower model artifacts.
    
    Args:
        model: Trained TwoTower model
        user2idx: User to index mapping
        item2idx: Item to index mapping
        user_profiles: User profile features
        X_item: Item feature matrix
        cfg: Configuration dictionary
        patience: Early stopping patience
        min_delta: Early stopping min delta
    """
    two_dir = os.path.join(cfg["artifact_root"], "twoTower")
    os.makedirs(two_dir, exist_ok=True)
    
    # Move model to CPU for saving
    model_cpu = model.cpu()
    
    # Compute embeddings
    with torch.no_grad():
        user_embs = model_cpu.forward_user(torch.tensor(user_profiles, dtype=torch.float32)).detach().cpu().numpy()
        item_embs = model_cpu.forward_item(torch.tensor(X_item, dtype=torch.float32)).detach().cpu().numpy()
    
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
    np.save(os.path.join(two_dir, "user_embeddings.npy"), user_embs)
    np.save(os.path.join(two_dir, "item_embeddings.npy"), item_embs)
    
    # Save indices
    user_index.to_csv(os.path.join(two_dir, "user_index.csv"), index=False)
    item_index.to_csv(os.path.join(two_dir, "item_index.csv"), index=False)
    
    # Save model
    torch.save(model_cpu.state_dict(), os.path.join(two_dir, "two_tower_model.pt"))
    
    # Save metadata
    two_tower_metadata = {
        "model_type": "two_tower_content",
        "embedding_dim": int(user_embs.shape[1]),
        "n_users": int(user_embs.shape[0]),
        "n_items": int(item_embs.shape[0]),
        "rating_threshold": cfg["rating_threshold"],
        "loss": "InfoNCE",
        "negative_sampling": "in_batch",
        "regularization": "adam_weight_decay",
        "early_stopping": {"patience": patience, "min_delta": min_delta},
        "seed": cfg["seed"]
    }
    
    with open(os.path.join(two_dir, "two_tower_metadata.json"), "w") as f:
        json.dump(two_tower_metadata, f, indent=2)
    
    print(f"Two-Tower artifacts saved to {two_dir}")
