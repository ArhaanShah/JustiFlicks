"""
Smoke tests for collaborative filtering models.
Tests basic functionality without running full training.
"""

import pytest
import numpy as np
import pandas as pd
import torch
import tempfile
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.MF import MF_BPR, PosDataset, predict_topk
from src.models.implicitBPR import train_implicit_bpr
from src.models.twoTower import TwoTower


class TestMFBPR:
    """Test MF_BPR model basic functionality."""
    
    def test_model_initialization(self):
        """Test that MF_BPR model can be initialized."""
        model = MF_BPR(n_users=100, n_items=50, k=16)
        assert model.user_emb.num_embeddings == 100
        assert model.item_emb.num_embeddings == 50
        assert model.user_emb.embedding_dim == 16
        assert model.item_emb.embedding_dim == 16
    
    def test_model_forward(self):
        """Test forward pass."""
        model = MF_BPR(n_users=100, n_items=50, k=16)
        u = torch.tensor([0, 1, 2])
        i = torch.tensor([5, 10, 15])
        scores = model(u, i)
        assert scores.shape == (3,)
        assert scores.dtype == torch.float32
    
    def test_model_score_function(self):
        """Test score computation."""
        model = MF_BPR(n_users=100, n_items=50, k=16)
        u = torch.tensor([0])
        i = torch.tensor([5])
        score = model.score(u, i)
        assert score.shape == (1,)
        assert isinstance(score.item(), float)


class TestPosDataset:
    """Test PosDataset functionality."""
    
    def test_dataset_creation(self):
        """Test dataset can be created from DataFrame."""
        df = pd.DataFrame({
            'u': [0, 1, 2, 3],
            'i': [10, 20, 30, 40]
        })
        dataset = PosDataset(df)
        assert len(dataset) == 4
    
    def test_dataset_getitem(self):
        """Test dataset indexing."""
        df = pd.DataFrame({
            'u': [0, 1, 2],
            'i': [10, 20, 30]
        })
        dataset = PosDataset(df)
        u, i = dataset[0]
        assert u.item() == 0
        assert i.item() == 10


class TestPredictTopK:
    """Test prediction functionality."""
    
    def test_predict_topk_basic(self):
        """Test top-k prediction."""
        model = MF_BPR(n_users=10, n_items=20, k=8)
        model.eval()
        
        user2idx = {5: 0, 6: 1, 7: 2}
        idx2item = {i: i+100 for i in range(20)}
        
        result = predict_topk(model, 5, user2idx, idx2item, "cpu", k=5)
        assert len(result) <= 5
        assert all(isinstance(x, (int, np.integer)) for x in result)
    
    def test_predict_topk_unseen_user(self):
        """Test prediction for unseen user returns empty list."""
        model = MF_BPR(n_users=10, n_items=20, k=8)
        model.eval()
        
        user2idx = {5: 0, 6: 1}
        idx2item = {i: i+100 for i in range(20)}
        
        result = predict_topk(model, 999, user2idx, idx2item, "cpu", k=5)
        assert result == []


class TestTwoTowerModel:
    """Test Two-Tower model basic functionality."""
    
    def test_model_initialization(self):
        """Test that TwoTower can be initialized."""
        model = TwoTower(
            input_dim=30,
            hidden_dims=[64, 32],
            emb_dim=16,
            drop=0.2
        )
        assert model.item_net is not None
        assert model.user_net is not None
    
    def test_model_forward(self):
        """Test forward pass with features."""
        model = TwoTower(
            input_dim=30,
            hidden_dims=[64, 32],
            emb_dim=16,
            drop=0.2
        )
        user_feat = torch.randn(3, 30)
        item_feat = torch.randn(3, 30)
        
        user_emb = model.user_net(user_feat)
        item_emb = model.item_net(item_feat)
        
        assert user_emb.shape == (3, 16)
        assert item_emb.shape == (3, 16)
        assert user_emb.dtype == torch.float32


class TestImplicitBPR:
    """Test implicit BPR training can be called."""
    
    def test_imports(self):
        """Test that implicit BPR module can be imported."""
        from src.models.implicitBPR import train_implicit_bpr
        assert callable(train_implicit_bpr)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
