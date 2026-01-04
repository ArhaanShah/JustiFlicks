"""
Smoke tests for evaluation metrics.
Tests basic functionality without running full evaluations.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestMetricsCalculation:
    """Test evaluation metrics computation."""
    
    def test_ndcg_simple_case(self):
        """Test NDCG calculation on simple case."""
        from src.eval.helper import ndcg_at_k_graded
        
        # Perfect ranking: relevant items at top
        predictions = [1, 2, 3, 4, 5]
        ground_truth = {1: 5.0, 2: 4.0, 3: 3.0}  # First 3 are relevant
        
        ndcg = ndcg_at_k_graded(predictions, ground_truth, k=5)
        
        # Should be close to 1.0 since relevant items are at top
        assert ndcg is not None and 0.8 < ndcg <= 1.0
    
    def test_ndcg_worst_case(self):
        """Test NDCG for worst ranking."""
        from src.eval.helper import ndcg_at_k_graded
        
        # Worst ranking: irrelevant items at top
        predictions = [10, 11, 12, 1, 2]
        ground_truth = {1: 5.0, 2: 4.0}  # Relevant items at bottom
        
        ndcg = ndcg_at_k_graded(predictions, ground_truth, k=5)
        
        # Should be low since relevant items are at bottom
        assert ndcg is not None and ndcg < 0.8
    
    def test_recall_calculation(self):
        """Test recall calculation."""
        from src.eval.helper import recall_at_k_binary
        
        # Predictions contain 2 out of 3 relevant items
        predictions = [1, 2, 10, 11, 12]
        ground_truth = {1: 5.0, 2: 4.5, 3: 4.5}  # All 3 are relevant (>= 4.5)
        
        recall = recall_at_k_binary(predictions, ground_truth, k=5)
        
        # Recall should be positive since we found some relevant items
        assert recall is not None and recall > 0 and recall <= 1.0
    
    def test_precision_calculation(self):
        """Test precision calculation with MAP."""
        from src.eval.helper import map_at_k_binary
        
        # 2 relevant out of 5 predictions
        predictions = [1, 2, 10, 11, 12]
        ground_truth = {1: 5.0, 2: 4.5, 3: 4.5}  # First 2 in predictions are relevant
        
        map_score = map_at_k_binary(predictions, ground_truth, k=5)
        
        # MAP should be positive when relevant items are found
        assert map_score is not None and map_score > 0
    
    def test_empty_predictions(self):
        """Test metrics with empty predictions."""
        from src.eval.helper import ndcg_at_k_graded, recall_at_k_binary, map_at_k_binary
        
        predictions = []
        ground_truth = {1: 5.0, 2: 4.5}
        
        ndcg = ndcg_at_k_graded(predictions, ground_truth, k=10)
        recall = recall_at_k_binary(predictions, ground_truth, k=10)
        map_score = map_at_k_binary(predictions, ground_truth, k=10)
        
        # All should be 0 or None
        assert ndcg == 0.0 or ndcg is None
        assert recall == 0.0 or recall is None
        assert map_score == 0.0 or map_score is None
    
    def test_no_relevant_items(self):
        """Test metrics when no relevant items in predictions."""
        from src.eval.helper import ndcg_at_k_graded, recall_at_k_binary, map_at_k_binary
        
        predictions = [10, 11, 12, 13, 14]
        ground_truth = {1: 5.0, 2: 4.5}
        
        ndcg = ndcg_at_k_graded(predictions, ground_truth, k=5)
        recall = recall_at_k_binary(predictions, ground_truth, k=5)
        map_score = map_at_k_binary(predictions, ground_truth, k=5)
        
        # All should be 0
        assert ndcg == 0.0 or ndcg is None
        assert recall == 0.0 or recall is None
        assert map_score == 0.0 or map_score is None


class TestEvaluationHelper:
    """Test evaluation helper functions."""
    
    def test_item2idx_setting(self):
        """Test global item2idx setting."""
        from src.eval.helper import set_item2idx
        import src.eval.helper as helper
        
        item2idx_dict = {1: 0, 2: 1, 3: 2}
        set_item2idx(item2idx_dict)
        
        # Access the global variable directly
        retrieved = helper.item2idx
        assert retrieved == item2idx_dict
    
    def test_evaluate_model_structure(self):
        """Test that evaluate_model returns correct structure."""
        from src.eval.helper import evaluate_model
        
        # Create minimal synthetic data
        predictions = {
            1: [10, 20, 30],
            2: [40, 50, 60]
        }
        
        ground_truth = {
            1: {10: 5.0, 20: 4.0},
            2: {40: 4.5, 50: 3.5}
        }
        
        item_support = pd.DataFrame({
            'movieId': [10, 20, 30, 40, 50, 60],
            'support_bin': ['20+'] * 6,
            'era_bin': ['2000-2009'] * 6,
            'imdb_vote_bin': ['1000+'] * 6
        })
        
        metrics, slices = evaluate_model(
            predictions, 
            ground_truth, 
            item_support, 
            ks=[5, 10], 
            seed=42
        )
        
        # Check structure
        assert isinstance(metrics, dict)
        assert isinstance(slices, dict)
        
        # Check metrics keys (actual implementation uses recall5@k format)
        assert 'ndcg@5' in metrics
        assert 'ndcg@10' in metrics
        assert 'recall5@5' in metrics  # Note: uses recall5@k format
        assert 'map5@5' in metrics


class TestSliceEvaluation:
    """Test slice-based evaluation."""
    
    def test_slice_filtering(self):
        """Test that slices can be filtered correctly."""
        # Create item support table
        item_support = pd.DataFrame({
            'movieId': [1, 2, 3, 4, 5],
            'support_bin': ['20+', '5-19', '20+', '1-4', '20+'],
            'era_bin': ['2000-2009', '2010-2019', '2000-2009', '1990-1999', '2010-2019'],
            'vote_bin': ['1000+', '100-999', '1000+', '10-99', '1000+']
        })
        
        # Filter by support bin
        high_support = item_support[item_support.support_bin == '20+']
        assert len(high_support) == 3
        
        # Filter by era
        modern = item_support[item_support.era_bin == '2010-2019']
        assert len(modern) == 2
        
        # Filter by vote bin
        popular = item_support[item_support.vote_bin == '1000+']
        assert len(popular) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
