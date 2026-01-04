"""
CLI entrypoints for JustiFlicks baseline CF pipeline.

Commands:
    python -m src.cli.entrypoints create_splits --config configs/default.yaml
    python -m src.cli.entrypoints build_embeddings --config configs/default.yaml
    python -m src.cli.entrypoints train_mf --config configs/default.yaml
    python -m src.cli.entrypoints train_implicit_bpr --config configs/default.yaml
    python -m src.cli.entrypoints train_two_tower --config configs/default.yaml
    python -m src.cli.entrypoints eval_all --config configs/default.yaml
    python -m src.cli.entrypoints reproducibility_eval --config configs/reproducibility.yaml
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import torch


def load_cfg(config_path: str) -> dict:
    """Load configuration from yaml file."""
    from src.utils.config import load_config
    return load_config(config_path)


def create_splits_cmd(args):
    """Create train/val/test splits."""
    cfg = load_cfg(args.config)
    from src.data.splits import create_splits
    create_splits(cfg)


def build_embeddings_cmd(args):
    """Build item feature embeddings."""
    cfg = load_cfg(args.config)
    from src.embeddings.langGenre import build_item_features
    build_item_features(cfg)


def train_mf_cmd(args):
    """Train MF_BPR model."""
    cfg = load_cfg(args.config)
    from src.models.MF import train_mf
    train_mf(cfg)
    print("MF training complete. Artifacts saved.")


def train_implicit_bpr_cmd(args):
    """Train implicit BPR model."""
    cfg = load_cfg(args.config)
    from src.models.implicitBPR import train_implicit_bpr
    train_implicit_bpr(cfg)
    print("Implicit BPR training complete. Artifacts saved.")


def train_two_tower_cmd(args):
    """Train Two-Tower model."""
    cfg = load_cfg(args.config)
    from src.models.twoTower import train_two_tower
    train_two_tower(cfg)
    print("Two-Tower training complete. Artifacts saved.")


def eval_all_cmd(args):
    """Evaluate all saved models using the standardized test_eval."""
    cfg = load_cfg(args.config)
    from src.eval.metrics import test_eval
    from src.eval.helper import set_item2idx
    
    # Check for wandb (optional logging) - preserved from original intent if needed
    
    models = ["MF", "implicitBPR", "twoTower"]
    
    for model_name in models:
        model_dir = os.path.join(cfg["artifact_root"], model_name)
        
        if not os.path.exists(model_dir):
            print(f"[SKIP] {model_name}: artifacts not found at {model_dir}")
            continue
        
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name}")
        print('='*50)
        
        # Load necessary artifacts to build a predict_fn
        # Note: We load embeddings directly for prediction as it's faster/standard for this pipeline
        try:
            user_emb = np.load(os.path.join(model_dir, "user_embeddings.npy"))
            item_emb = np.load(os.path.join(model_dir, "item_embeddings.npy"))
            
            user_index = pd.read_csv(os.path.join(model_dir, "user_index.csv"))
            item_index = pd.read_csv(os.path.join(model_dir, "item_index.csv"))
            
            user2idx = dict(zip(user_index.userId, user_index.user_idx))
            item2idx = dict(zip(item_index.movieId, item_index.item_idx))
            idx2item = dict(zip(item_index.item_idx, item_index.movieId))
            
            set_item2idx(item2idx)
            
            # Define predict function closure
            def predict_fn_closure(user_id, k):
                if user_id not in user2idx:
                    return []
                uidx = user2idx[user_id]
                # Dot product scoring
                # metrics.py expects this signature
                scores = user_emb[uidx] @ item_emb.T
                topk_idx = np.argsort(-scores)[:k]
                return [idx2item[int(i)] for i in topk_idx]
            
            # Run test_eval
            # This handles getting ground truth, computing metrics, and saving them
            test_eval(cfg, predict_fn_closure, model_name=model_name, seed=cfg["seed"])
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()


def reproducibility_eval_cmd(args):
    """Run reproducibility evaluation across models and seeds."""
    import yaml
    from src.eval.helper import evaluate_model, set_item2idx
    from src.data.splits import load_splits, load_item_support_table
    
    with open(args.config, "r") as f:
        repro_cfg = yaml.safe_load(f)
    
    # Load default config for paths
    default_config_path = repro_cfg.get("default_config", "configs/default.yaml")
    cfg = load_cfg(default_config_path)
    
    # Merge reproducibility params
    seeds = repro_cfg.get("seeds", [34, 35, 36])
    sample_frac = repro_cfg.get("sample_frac", 0.1)
    min_users = repro_cfg.get("min_users", 10000)
    max_users = repro_cfg.get("max_users", 50000)
    k_val = repro_cfg.get("k", 10)
    
    _, _, test = load_splits(cfg)
    item_support_table = load_item_support_table(cfg)
    
    # Check for wandb
    try:
        import wandb
        has_wandb = True
    except ImportError:
        has_wandb = False
        print("[WARN] wandb not installed, skipping W&B logging")
    
    def predict_topk_from_embeddings(user_emb, item_emb, user2idx, idx2item, user_id, k):
        if user_id not in user2idx:
            return []
        uidx = user2idx[user_id]
        scores = user_emb[uidx] @ item_emb.T
        topk_idx = np.argsort(-scores)[:k]
        return [idx2item[int(i)] for i in topk_idx]
    
    def run_repro_eval(model_name, artifact_root):
        model_dir = os.path.join(artifact_root, model_name)
        
        if not os.path.exists(model_dir):
            print(f"[SKIP] {model_name}: artifacts not found")
            return
        
        # Load embeddings and indices
        user_emb = np.load(os.path.join(model_dir, "user_embeddings.npy"))
        item_emb = np.load(os.path.join(model_dir, "item_embeddings.npy"))
        
        user_index = pd.read_csv(os.path.join(model_dir, "user_index.csv"))
        item_index = pd.read_csv(os.path.join(model_dir, "item_index.csv"))
        
        user2idx = dict(zip(user_index.userId, user_index.user_idx))
        item2idx = dict(zip(item_index.movieId, item_index.item_idx))
        idx2item = dict(zip(item_index.item_idx, item_index.movieId))
        
        set_item2idx(item2idx)
        
        all_users = np.array(list(user2idx.keys()))
        
        for seed in seeds:
            rng = np.random.RandomState(seed)
            n_sample = min(max_users, max(min_users, int(len(all_users) * sample_frac)))
            sampled_users = rng.choice(all_users, size=n_sample, replace=False)
            
            test_sub = test[test.userId.isin(sampled_users)]
            
            gt = {}
            preds = {}
            
            for uid, g in test_sub.groupby("userId"):
                gt[int(uid)] = {int(r.movieId): float(r.rating) for _, r in g.iterrows()}
            
            for uid in gt.keys():
                preds[int(uid)] = predict_topk_from_embeddings(
                    user_emb, item_emb, user2idx, idx2item, uid, max(cfg["k_values"])
                )
            
            metrics, _ = evaluate_model(
                preds, gt, item_support_table, ks=cfg["k_values"], seed=seed
            )
            
            ndcg10 = metrics["ndcg@10"]
            print(f"[REPRO] model={model_name} seed={seed} ndcg@10={ndcg10:.6f}")
            
            # Log to W&B quietly
            if has_wandb:
                try:
                    run = wandb.init(
                        project="JustiFlicks",
                        job_type="compare",
                        name="01_baseline_ndcg",
                        reinit=True,
                        settings=wandb.Settings(silent=True),
                        config={
                            "model": model_name,
                            "seed": seed,
                            "evaluation_type": "reproducibility",
                            "sample_frac": sample_frac
                        }
                    )
                    run.log({"ndcg@10": ndcg10})
                    run.log(metrics)
                    run.finish()
                except Exception as e:
                    print(f"[WARN] W&B logging failed: {e}")
    
    # Run for all models
    for model_name in ["MF", "implicitBPR", "twoTower"]:
        run_repro_eval(model_name, cfg["artifact_root"])


def main():
    parser = argparse.ArgumentParser(
        description="JustiFlicks Baseline CF Pipeline CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # create_splits command
    splits_parser = subparsers.add_parser("create_splits", help="Create train/val/test splits")
    splits_parser.add_argument("--config", type=str, required=True, help="Path to config file")
    splits_parser.set_defaults(func=create_splits_cmd)
    
    # build_embeddings command
    emb_parser = subparsers.add_parser("build_embeddings", help="Build item feature embeddings")
    emb_parser.add_argument("--config", type=str, required=True, help="Path to config file")
    emb_parser.set_defaults(func=build_embeddings_cmd)
    
    # train_mf command
    mf_parser = subparsers.add_parser("train_mf", help="Train MF_BPR model")
    mf_parser.add_argument("--config", type=str, required=True, help="Path to config file")
    mf_parser.set_defaults(func=train_mf_cmd)
    
    # train_implicit_bpr command
    implicit_parser = subparsers.add_parser("train_implicit_bpr", help="Train implicit BPR model")
    implicit_parser.add_argument("--config", type=str, required=True, help="Path to config file")
    implicit_parser.set_defaults(func=train_implicit_bpr_cmd)
    
    # train_two_tower command
    tt_parser = subparsers.add_parser("train_two_tower", help="Train Two-Tower model")
    tt_parser.add_argument("--config", type=str, required=True, help="Path to config file")
    tt_parser.set_defaults(func=train_two_tower_cmd)
    
    # eval_all command
    eval_parser = subparsers.add_parser("eval_all", help="Evaluate all saved models")
    eval_parser.add_argument("--config", type=str, required=True, help="Path to config file")
    eval_parser.set_defaults(func=eval_all_cmd)
    
    # reproducibility_eval command
    repro_parser = subparsers.add_parser("reproducibility_eval", help="Run reproducibility evaluation")
    repro_parser.add_argument("--config", type=str, required=True, help="Path to reproducibility config file")
    repro_parser.set_defaults(func=reproducibility_eval_cmd)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
