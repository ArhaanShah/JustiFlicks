"""
Data splits module.
Creates per-user chronological train/val/test splits, small sample splits,
item support table, and manifest file.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


def support_bin(x: int) -> str:
    """Bin support counts into categories."""
    if x == 0:
        return "0"
    if x < 5:
        return "1-4"
    if x < 20:
        return "5-19"
    return "20+"


def vote_bin(x: float) -> str:
    """Bin IMDb vote counts into categories."""
    if x <= 0:
        return "0"
    if x < 10:
        return "1-9"
    if x < 100:
        return "10-99"
    if x < 1000:
        return "100-999"
    return "1000+"


def era_bin(y: float) -> str:
    """Bin release years into eras."""
    if pd.isna(y):
        return "unknown"
    y = int(y)
    if y < 1970:
        return "1900-1969"
    if y < 1980:
        return "1970-1979"
    if y < 1990:
        return "1980-1989"
    if y < 2000:
        return "1990-1999"
    if y < 2010:
        return "2000-2009"
    if y < 2020:
        return "2010-2019"
    return "2020-2029"


def create_splits(cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create per-user chronological train/val/test splits.
    
    Args:
        cfg: Configuration dictionary with data_root, artifact_root, n_test, n_val, etc.
        
    Returns:
        Tuple of (train, val, test) DataFrames
    """
    data_root = cfg["data_root"]
    artifact_root = cfg["artifact_root"]
    n_test = cfg["n_test"]
    n_val = cfg["n_val"]
    seed = cfg["seed"]
    rating_threshold = cfg["rating_threshold"]
    min_pos = cfg["min_pos"]
    
    splits_dir = os.path.join(artifact_root, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    
    # Load ratings
    ratings_path = os.path.join(data_root, "movieRatings.parquet")
    ratings = pd.read_parquet(ratings_path)
    ratings = ratings.sort_values(["userId", "datetime"])
    
    # Calculate per-user ranks and counts
    g = ratings.groupby("userId", group_keys=False)
    ratings["rank_from_end"] = g.cumcount(ascending=False)
    ratings["user_count"] = g["userId"].transform("size")
    
    # Create splits (only users with > 10 ratings)
    train = ratings[(ratings.user_count > 10) & (ratings.rank_from_end >= (n_val + n_test))].copy()
    val = ratings[(ratings.user_count > 10) & (ratings.rank_from_end < (n_val + n_test)) & (ratings.rank_from_end >= n_test)].copy()
    test = ratings[(ratings.user_count > 10) & (ratings.rank_from_end < n_test)].copy()
    
    # Drop temporary columns
    train = train.drop(columns=["rank_from_end", "user_count"])
    val = val.drop(columns=["rank_from_end", "user_count"])
    test = test.drop(columns=["rank_from_end", "user_count"])
    
    # Save main splits
    train.to_parquet(os.path.join(splits_dir, "train.parquet"), index=False)
    val.to_parquet(os.path.join(splits_dir, "val.parquet"), index=False)
    test.to_parquet(os.path.join(splits_dir, "test.parquet"), index=False)
    
    # Create sample splits (10k samples each)
    rng = np.random.RandomState(seed)
    train_sample = train.sample(n=min(10000, len(train)), random_state=seed)
    val_sample = val.sample(n=min(10000, len(val)), random_state=seed)
    test_sample = test.sample(n=min(10000, len(test)), random_state=seed)
    
    train_sample.to_csv(os.path.join(splits_dir, "train_sample.csv"), index=False)
    val_sample.to_csv(os.path.join(splits_dir, "val_sample.csv"), index=False)
    test_sample.to_csv(os.path.join(splits_dir, "test_sample.csv"), index=False)
    
    # Create item support table
    item_support_table = _create_item_support_table(train, data_root, rating_threshold)
    item_support_table.to_parquet(os.path.join(splits_dir, "item_support_table.parquet"), index=False)
    
    # Create small splits (1/5 of users)
    _create_small_splits(train, val, test, splits_dir, seed)
    
    # Create manifest
    manifest = _create_manifest(train, val, test, rating_threshold, min_pos)
    with open(os.path.join(splits_dir, "splits_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Splits created in {splits_dir}")
    print(f"  train: {len(train)} rows, {train.userId.nunique()} users")
    print(f"  val: {len(val)} rows, {val.userId.nunique()} users")
    print(f"  test: {len(test)} rows, {test.userId.nunique()} users")
    
    return train, val, test


def _create_item_support_table(train: pd.DataFrame, data_root: str, rating_threshold: float) -> pd.DataFrame:
    """Create the item support table with slicing columns."""
    train_pos = train[train.rating >= rating_threshold]
    support = train_pos.groupby("movieId").size().reset_index(name="support")
    
    links = pd.read_parquet(os.path.join(data_root, "movieLinks.parquet"))
    
    # Merge support with IMDb IDs
    support_imdb = support.merge(links[["movieId", "imdbId"]], on="movieId", how="left")
    support_imdb = support_imdb.groupby("imdbId", dropna=False)["support"].sum().reset_index()
    
    # Get first movieId per imdbId
    movie_ids_per_imdb = links.groupby("imdbId")["movieId"].first().reset_index()
    
    # Load movie data
    movie_data = pd.read_parquet(os.path.join(data_root, "movieData.parquet"))
    
    # Build item support table
    item_support_table = movie_data[["imdbId", "release_year", "original_language", "num_votes_imdb"]]\
        .drop_duplicates("imdbId")\
        .merge(movie_ids_per_imdb, on="imdbId", how="left")\
        .merge(support_imdb, on="imdbId", how="left")\
        .fillna({"support": 0})
    
    item_support_table["support"] = item_support_table["support"].astype(int)
    
    # Add bin columns
    item_support_table["support_bin"] = item_support_table.support.apply(support_bin)
    item_support_table["imdb_vote_bin"] = item_support_table.num_votes_imdb.apply(vote_bin)
    item_support_table["era_bin"] = item_support_table.release_year.apply(era_bin)
    
    return item_support_table


def _create_small_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, 
                         splits_dir: str, seed: int):
    """Create small sample splits (1/5 of users)."""
    rng = np.random.RandomState(seed)
    
    users = train.userId.unique()
    n_small = len(users) // 5
    small_users = set(rng.choice(users, size=n_small, replace=False))
    
    train_small = train[train.userId.isin(small_users)].copy()
    val_small = val[val.userId.isin(small_users)].copy()
    test_small = test[test.userId.isin(small_users)].copy()
    
    # Sort by user and datetime
    train_small = train_small.sort_values(["userId", "datetime"])
    val_small = val_small.sort_values(["userId", "datetime"])
    test_small = test_small.sort_values(["userId", "datetime"])
    
    train_small.to_parquet(os.path.join(splits_dir, "train_small.parquet"), index=False)
    val_small.to_parquet(os.path.join(splits_dir, "val_small.parquet"), index=False)
    test_small.to_parquet(os.path.join(splits_dir, "test_small.parquet"), index=False)
    
    print(f"Small splits: {len(train_small)} train, {len(val_small)} val, {len(test_small)} test")
    print(f"Unique users: {train_small.userId.nunique()}")


def _create_manifest(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                     rating_threshold: float, min_pos: int) -> Dict[str, Any]:
    """Create the splits manifest with metadata."""
    train_pos = train[train.rating >= rating_threshold]
    users_before = int(train.userId.nunique())
    user_pos_counts = train_pos.groupby("userId").size()
    eligible_users = user_pos_counts[user_pos_counts >= min_pos].index
    users_after = int(len(eligible_users))
    users_removed = users_before - users_after
    
    manifest = {
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "unique_items_train": int(train.movieId.nunique()),
        "rating_threshold": rating_threshold,
        "train_eligibility": {
            "n_items": int(train.movieId.nunique()),
            "full_catalog": int(train.movieId.nunique())
        },
        "cf_eligibility": {
            "criterion": "min_positive_in_train",
            "min_pos": min_pos,
            "users_before": users_before,
            "users_after": users_after,
            "users_removed": users_removed
        }
    }
    
    return manifest


def load_splits(cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load existing splits from disk.
    
    Args:
        cfg: Configuration dictionary with artifact_root
        
    Returns:
        Tuple of (train, val, test) DataFrames
    """
    splits_dir = os.path.join(cfg["artifact_root"], "splits")
    
    train = pd.read_parquet(os.path.join(splits_dir, "train.parquet"))
    val = pd.read_parquet(os.path.join(splits_dir, "val.parquet"))
    test = pd.read_parquet(os.path.join(splits_dir, "test.parquet"))
    
    return train, val, test


def load_item_support_table(cfg: dict) -> pd.DataFrame:
    """Load the item support table from disk."""
    splits_dir = os.path.join(cfg["artifact_root"], "splits")
    return pd.read_parquet(os.path.join(splits_dir, "item_support_table.parquet"))
