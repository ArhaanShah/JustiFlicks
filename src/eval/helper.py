"""
Evaluation helper module.
Contains evaluate_model and all helper functions for NDCG, Recall, MAP,
bootstrap CI, and slice-based evaluation.
"""

import os
from math import log2
from typing import Dict, List, Tuple, Any, Optional, Set, Union
import numpy as np
import pandas as pd


# Global item2idx for catalog coverage calculation
# Can be set externally before calling evaluate_model
item2idx: Dict[int, int] = {}


def _relevance_from_rating(r: Optional[float]) -> float:
    """Convert a rating to relevance score."""
    if r is None:
        return 0.0
    try:
        r = float(r)
    except Exception:
        return 0.0
    if r >= 5.0:
        return 1
    if r >= 4.5:
        return 1
    if r >= 4.0:
        return 0
    return 0.0


def _item_relevance(item: int, gt: Union[Dict[int, float], Set[int]]) -> float:
    """Get relevance of an item given ground truth."""
    if isinstance(gt, dict):
        return _relevance_from_rating(gt.get(item, None))
    if isinstance(gt, set):
        return 1.0 if item in gt else 0.0
    return 0.0


def ndcg_at_k_graded(pred: List[int], gt: Union[Dict[int, float], Set[int]], k: int) -> Optional[float]:
    """
    Compute graded NDCG@k.
    
    Args:
        pred: List of predicted item IDs
        gt: Ground truth as dict {item_id: rating} or set of relevant items
        k: Cutoff for evaluation
        
    Returns:
        NDCG@k score or None if no ground truth
    """
    if (isinstance(gt, set) and len(gt) == 0) or (isinstance(gt, dict) and len(gt) == 0):
        return None
    
    dcg = 0.0
    for i, item in enumerate(pred[:k], start=1):
        rel = _item_relevance(item, gt)
        if rel > 0:
            dcg += rel / log2(i + 1)
    
    if isinstance(gt, dict):
        rels = [_relevance_from_rating(r) for r in gt.values()]
    else:
        rels = [1.0 for _ in gt]
    
    rels_sorted = sorted(rels, reverse=True)
    idcg = sum(v / log2(i + 1) for i, v in enumerate(rels_sorted[:k], start=1))
    
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k_binary(pred: List[int], gt: Union[Dict[int, float], Set[int]], 
                       k: int, threshold: float = 5.0) -> Optional[float]:
    """
    Compute binary recall@k with rating threshold.
    
    Args:
        pred: List of predicted item IDs
        gt: Ground truth as dict {item_id: rating} or set of relevant items
        k: Cutoff for evaluation
        threshold: Rating threshold for relevance
        
    Returns:
        Recall@k score or None if no relevant items
    """
    if isinstance(gt, dict):
        gt_set = {i for i, r in gt.items() if r >= threshold}
    elif isinstance(gt, set):
        gt_set = set(gt)
    else:
        gt_set = set()
    
    if len(gt_set) == 0:
        return None
    
    hits = sum(1 for i in pred[:k] if i in gt_set)
    return hits / len(gt_set)


def map_at_k_binary(pred: List[int], gt: Union[Dict[int, float], Set[int]], 
                    k: int, threshold: float = 5.0) -> Optional[float]:
    """
    Compute binary MAP@k with rating threshold.
    
    Args:
        pred: List of predicted item IDs
        gt: Ground truth as dict {item_id: rating} or set of relevant items
        k: Cutoff for evaluation
        threshold: Rating threshold for relevance
        
    Returns:
        MAP@k score or None if no relevant items
    """
    if isinstance(gt, dict):
        gt_set = {i for i, r in gt.items() if r >= threshold}
    elif isinstance(gt, set):
        gt_set = set(gt)
    else:
        gt_set = set()
    
    if len(gt_set) == 0:
        return None
    
    hits = 0
    s = 0.0
    for i, item in enumerate(pred[:k], start=1):
        if item in gt_set:
            hits += 1
            s += hits / i
    
    return s / min(len(gt_set), k)


def _build_popularity_map(item_support_df: pd.DataFrame, data_root: Optional[str] = None) -> Dict[int, float]:
    """
    Build a popularity map from item support DataFrame.
    
    Args:
        item_support_df: DataFrame with movieId and num_votes_imdb columns
        data_root: Optional data root for loading movieLinks if needed
        
    Returns:
        Dict mapping movieId to popularity (num_votes_imdb)
    """
    pop_map = {}
    
    if "movieId" in item_support_df.columns and "num_votes_imdb" in item_support_df.columns:
        pop_map = dict(zip(
            item_support_df["movieId"].tolist(),
            item_support_df["num_votes_imdb"].fillna(0).astype(float).tolist()
        ))
        return pop_map
    
    if "imdbId" in item_support_df.columns and "num_votes_imdb" in item_support_df.columns and "movieId" in item_support_df.columns:
        pop_map = dict(zip(
            item_support_df["movieId"].tolist(),
            item_support_df["num_votes_imdb"].fillna(0).astype(float).tolist()
        ))
        return pop_map
    
    if data_root and "imdbId" in item_support_df.columns and "num_votes_imdb" in item_support_df.columns:
        try:
            links = pd.read_parquet(os.path.join(data_root, "movieLinks.parquet"))
            merged = links.merge(item_support_df[["imdbId", "num_votes_imdb"]], on="imdbId", how="left")
            pop_map = dict(zip(
                merged["movieId"].tolist(),
                merged["num_votes_imdb"].fillna(0).astype(float).tolist()
            ))
            return pop_map
        except Exception:
            return {}
    
    return {}


def _bootstrap_ci(values: List[float], boot_iters: int = 1000, 
                  alpha: float = 0.05, random_state: int = 34) -> Tuple[Optional[float], Tuple[Optional[float], Optional[float]]]:
    """
    Compute bootstrap confidence interval.
    
    Args:
        values: List of metric values
        boot_iters: Number of bootstrap iterations
        alpha: Significance level (0.05 for 95% CI)
        random_state: Random seed
        
    Returns:
        Tuple of (mean, (ci_low, ci_high))
    """
    vals = np.array([v for v in values if v is not None])
    if len(vals) == 0:
        return None, (None, None)
    
    rng = np.random.RandomState(random_state)
    n = len(vals)
    idxs = rng.randint(0, n, size=(boot_iters, n))
    samp_means = np.mean(vals[idxs], axis=1)
    
    mean = float(np.mean(vals))
    lo = float(np.percentile(samp_means, 100 * (alpha / 2)))
    hi = float(np.percentile(samp_means, 100 * (1 - alpha / 2)))
    
    return mean, (lo, hi)


def _compute_slice_recall(predictions: Dict[int, List[int]], 
                          ground_truth: Dict[int, Union[Dict[int, float], Set[int]]],
                          item_support_df: pd.DataFrame, 
                          bin_col: str, 
                          ks: List[int], 
                          min_users: int = 50) -> pd.DataFrame:
    """
    Compute recall sliced by a categorical column.
    
    Args:
        predictions: Dict of user_id -> list of predicted item IDs
        ground_truth: Dict of user_id -> ground truth (dict or set)
        item_support_df: DataFrame with slicing columns
        bin_col: Column name to slice by
        ks: List of k values
        min_users: Minimum users required to report a slice
        
    Returns:
        DataFrame with slice recall results
    """
    bin_map = item_support_df.set_index("movieId")[bin_col].to_dict()
    rows = []
    
    for b in item_support_df[bin_col].dropna().unique():
        for k in ks:
            vals = []
            n_users = 0
            
            for u, gt in ground_truth.items():
                if isinstance(gt, dict):
                    gt_items = set(gt.keys())
                else:
                    gt_items = set(gt)
                
                gt_b = {i for i in gt_items if bin_map.get(i) == b}
                if not gt_b:
                    continue
                
                preds = predictions.get(u, [])
                vals.append(len([i for i in preds[:k] if i in gt_b]) / len(gt_b))
                n_users += 1
            
            rows.append({
                "bin": b,
                "k": k,
                "recall": float(np.mean(vals)) if len(vals) >= min_users else None,
                "n_users_with_gt": n_users
            })
    
    return pd.DataFrame(rows)


def evaluate_model(predictions: Dict[int, List[int]], 
                   ground_truth: Dict[int, Union[Dict[int, float], Set[int]]],
                   item_support_df: pd.DataFrame,
                   ks: Optional[List[int]] = None,
                   boot_iters: int = 1000,
                   seed: int = 34,
                   data_root: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
    """
    Evaluate a recommendation model.
    
    Args:
        predictions: Dict of user_id -> list of predicted item IDs (ranked)
        ground_truth: Dict of user_id -> ground truth as dict {item_id: rating} or set
        item_support_df: DataFrame with item metadata for slicing
        ks: List of k values for evaluation (default: [5, 10, 20])
        boot_iters: Number of bootstrap iterations for CI
        seed: Random seed for bootstrap
        data_root: Optional data root for popularity map
        
    Returns:
        Tuple of (metrics dict, slices dict)
    """
    global item2idx
    
    if ks is None:
        ks = [5, 10, 20]
    
    users = list(ground_truth.keys())
    eligible_users = []
    skipped_users = []
    
    for u in users:
        gt = ground_truth[u]
        has_pos = any(r >= 5.0 for r in gt.values()) if isinstance(gt, dict) else len(gt) > 0
        if has_pos:
            eligible_users.append(u)
        else:
            skipped_users.append(u)
    
    metrics = {}
    slices = {}
    
    pop_map = _build_popularity_map(item_support_df, data_root)
    pop_vals = [v for v in pop_map.values() if v is not None]
    pop_90 = np.percentile(pop_vals, 90) if pop_vals else None
    
    for k in ks:
        ndcgs = []
        recall5s = []
        map5s = []
        mean_pops = []
        prop_top = []
        
        for u in eligible_users:
            preds = predictions.get(u, [])
            gt = ground_truth[u]
            
            n = ndcg_at_k_graded(preds, gt, k)
            r5 = recall_at_k_binary(preds, gt, k, threshold=5.0)
            m5 = map_at_k_binary(preds, gt, k, threshold=5.0)
            
            if n is not None:
                ndcgs.append(n)
            if r5 is not None:
                recall5s.append(r5)
            if m5 is not None:
                map5s.append(m5)
            
            pops = [pop_map.get(i, 0.0) for i in preds[:k]]
            mean_pops.append(np.mean(pops) if pops else 0.0)
            
            if pop_90 is not None and preds:
                prop_top.append(sum(1 for i in preds[:k] if pop_map.get(i, 0.0) >= pop_90) / k)
        
        mean_ndcg, ci = _bootstrap_ci(ndcgs, boot_iters=boot_iters, random_state=seed)
        
        metrics[f"ndcg@{k}"] = mean_ndcg
        metrics[f"ndcg@{k}_ci_low"] = ci[0]
        metrics[f"ndcg@{k}_ci_high"] = ci[1]
        metrics[f"recall5@{k}"] = float(np.mean(recall5s)) if recall5s else None
        metrics[f"map5@{k}"] = float(np.mean(map5s)) if map5s else None
        metrics[f"mean_popularity_num_votes@{k}"] = float(np.mean(mean_pops)) if mean_pops else None
        metrics[f"prop_top10pct_popularity@{k}"] = float(np.mean(prop_top)) if prop_top else None
    
    # Coverage metrics
    all_pred_items = set()
    for u in eligible_users:
        all_pred_items.update(predictions.get(u, [])[:max(ks)])
    
    metrics["users_total"] = len(users)
    metrics["users_evaluated"] = len(eligible_users)
    metrics["users_skipped"] = len(skipped_users)
    metrics["items_covered"] = len(all_pred_items)
    
    # Catalog coverage (requires item2idx to be set externally)
    if item2idx:
        metrics["evaluation_catalog_items"] = len(item2idx)
        metrics["catalog_coverage_fraction"] = len(all_pred_items) / len(item2idx) if item2idx else 0.0
    else:
        metrics["evaluation_catalog_items"] = len(all_pred_items)
        metrics["catalog_coverage_fraction"] = 1.0
    
    # Slice-based evaluation
    slices["by_support"] = _compute_slice_recall(predictions, ground_truth, item_support_df, "support_bin", ks)
    slices["by_imdb_votes"] = _compute_slice_recall(predictions, ground_truth, item_support_df, "imdb_vote_bin", ks)
    slices["by_era"] = _compute_slice_recall(predictions, ground_truth, item_support_df, "era_bin", ks)
    
    return metrics, slices


def set_item2idx(mapping: Dict[int, int]):
    """Set the global item2idx mapping for catalog coverage calculation."""
    global item2idx
    item2idx = mapping
