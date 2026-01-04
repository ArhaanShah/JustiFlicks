"""
Smoke tests for embeddings and feature building.
Tests basic functionality without processing full datasets.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestItemFeatureBuilding:
    """Test item feature building logic."""
    
    def test_genre_one_hot_encoding_logic(self):
        """Test that genre one-hot encoding works as expected."""
        # Create synthetic movie data with genres
        df = pd.DataFrame({
            'movieId': [1, 2, 3],
            'genres': [
                ['Action', 'Drama'],
                ['Comedy'],
                ['Action', 'Comedy']
            ]
        })
        
        # Explode genres and lowercase
        exploded = df['genres'].explode().str.lower()
        
        # Create dummies
        genre_dummies = pd.get_dummies(exploded).groupby(level=0).sum()
        
        # Verify shape and values
        assert len(genre_dummies) == 3
        assert 'action' in genre_dummies.columns
        assert 'comedy' in genre_dummies.columns
        assert 'drama' in genre_dummies.columns
        
        # Verify movie 1 has action and drama
        assert genre_dummies.loc[0, 'action'] == 1
        assert genre_dummies.loc[0, 'drama'] == 1
        assert genre_dummies.loc[0, 'comedy'] == 0
        
        # Verify movie 3 has action and comedy
        assert genre_dummies.loc[2, 'action'] == 1
        assert genre_dummies.loc[2, 'comedy'] == 1
    
    def test_language_one_hot_encoding_logic(self):
        """Test that language one-hot encoding works."""
        # Create synthetic language data
        df = pd.DataFrame({
            'movieId': [1, 2, 3, 4],
            'original_language': ['en', 'fr', 'en', 'es']
        })
        
        # Create language dummies
        lang_dummies = pd.get_dummies(df['original_language'], prefix='lang')
        
        # Verify shape
        assert len(lang_dummies) == 4
        assert 'lang_en' in lang_dummies.columns
        assert 'lang_fr' in lang_dummies.columns
        assert 'lang_es' in lang_dummies.columns
        
        # Verify encoding
        assert lang_dummies.loc[0, 'lang_en'] == 1
        assert lang_dummies.loc[0, 'lang_fr'] == 0
        assert lang_dummies.loc[1, 'lang_fr'] == 1
        assert lang_dummies.loc[1, 'lang_en'] == 0
    
    def test_feature_concatenation(self):
        """Test that features can be concatenated correctly."""
        # Create synthetic data
        base = pd.DataFrame({
            'movieId': [1, 2, 3],
            'imdbId': [100, 200, 300]
        })
        
        lang_features = pd.DataFrame({
            'lang_en': [1, 0, 1],
            'lang_fr': [0, 1, 0]
        })
        
        genre_features = pd.DataFrame({
            'action': [1, 0, 1],
            'comedy': [0, 1, 1]
        })
        
        # Concatenate
        combined = pd.concat([base, lang_features, genre_features], axis=1)
        
        # Verify
        assert len(combined) == 3
        assert 'movieId' in combined.columns
        assert 'lang_en' in combined.columns
        assert 'action' in combined.columns
        assert combined.shape[1] == 6  # 2 base + 2 lang + 2 genre


class TestEmbeddingModules:
    """Test embedding model components."""
    
    def test_embedding_initialization(self):
        """Test that PyTorch embeddings can be initialized."""
        import torch
        from torch import nn
        
        # Create simple embedding
        emb = nn.Embedding(num_embeddings=100, embedding_dim=32)
        
        # Test forward pass
        indices = torch.tensor([0, 5, 10])
        output = emb(indices)
        
        assert output.shape == (3, 32)
        assert output.dtype == torch.float32
    
    def test_embedding_lookup(self):
        """Test embedding lookup functionality."""
        import torch
        from torch import nn
        
        emb = nn.Embedding(10, 8)
        nn.init.constant_(emb.weight, 1.0)
        
        # Lookup
        idx = torch.tensor([0, 1, 2])
        result = emb(idx)
        
        assert result.shape == (3, 8)
        assert torch.allclose(result, torch.ones(3, 8))
    
    def test_dot_product_similarity(self):
        """Test dot product similarity computation."""
        import torch
        
        # Create two embedding vectors
        user_emb = torch.tensor([[1.0, 2.0, 3.0]])  # 1x3
        item_emb = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0]
        ])  # 3x3
        
        # Compute similarities (matrix multiplication)
        scores = user_emb @ item_emb.T  # 1x3
        
        assert scores.shape == (1, 3)
        assert torch.isclose(scores[0, 0], torch.tensor(1.0))  # 1*1 + 2*0 + 3*0 = 1
        assert torch.isclose(scores[0, 1], torch.tensor(2.0))  # 1*0 + 2*1 + 3*0 = 2
        assert torch.isclose(scores[0, 2], torch.tensor(6.0))  # 1*1 + 2*1 + 3*1 = 6


class TestNumpyOperations:
    """Test numpy operations used in embeddings."""
    
    def test_numpy_dot_product(self):
        """Test numpy dot product for scoring."""
        user_emb = np.array([[1.0, 2.0, 3.0]])
        item_emb = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0]
        ])
        
        # Compute scores
        scores = user_emb @ item_emb.T
        
        assert scores.shape == (1, 3)
        assert np.isclose(scores[0, 0], 1.0)
        assert np.isclose(scores[0, 1], 2.0)
        assert np.isclose(scores[0, 2], 6.0)
    
    def test_top_k_selection(self):
        """Test top-k selection with numpy."""
        scores = np.array([0.5, 0.9, 0.1, 0.8, 0.3])
        
        k = 3
        top_k_idx = np.argsort(-scores)[:k]
        
        assert len(top_k_idx) == 3
        assert top_k_idx[0] == 1  # 0.9 is highest
        assert top_k_idx[1] == 3  # 0.8 is second
        assert top_k_idx[2] == 0  # 0.5 is third


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
