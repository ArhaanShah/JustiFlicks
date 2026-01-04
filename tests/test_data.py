"""
Smoke tests for data processing modules.
Tests basic functionality without processing full datasets.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import sys
import yaml

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.splits import support_bin, vote_bin, era_bin


class TestBinningFunctions:
    """Test data binning utility functions."""
    
    def test_support_bin(self):
        """Test support count binning."""
        assert support_bin(0) == "0"
        assert support_bin(1) == "1-4"
        assert support_bin(4) == "1-4"
        assert support_bin(5) == "5-19"
        assert support_bin(19) == "5-19"
        assert support_bin(20) == "20+"
        assert support_bin(100) == "20+"
    
    def test_vote_bin(self):
        """Test vote count binning."""
        assert vote_bin(0) == "0"
        assert vote_bin(-1) == "0"
        assert vote_bin(1) == "1-9"
        assert vote_bin(9) == "1-9"
        assert vote_bin(10) == "10-99"
        assert vote_bin(99) == "10-99"
        assert vote_bin(100) == "100-999"
        assert vote_bin(999) == "100-999"
        assert vote_bin(1000) == "1000+"
        assert vote_bin(10000) == "1000+"
    
    def test_era_bin(self):
        """Test release year binning."""
        assert era_bin(1950) == "1900-1969"
        assert era_bin(1969) == "1900-1969"
        assert era_bin(1970) == "1970-1979"
        assert era_bin(1979) == "1970-1979"
        assert era_bin(1980) == "1980-1989"
        assert era_bin(1990) == "1990-1999"
        assert era_bin(2000) == "2000-2009"
        assert era_bin(2010) == "2010-2019"
        assert era_bin(2020) == "2020-2029"
        assert era_bin(np.nan) == "unknown"
        assert era_bin(None) == "unknown"


class TestSplitsCreation:
    """Test splits creation logic without running on full data."""
    
    def test_split_logic_simulation(self):
        """Test split assignment logic with small synthetic data."""
        # Create synthetic ratings data
        ratings = pd.DataFrame({
            'userId': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            'movieId': range(24),
            'rating': [4.0] * 24,
            'datetime': pd.date_range('2020-01-01', periods=24, freq='D').tolist() * 1
        })
        
        # Sort by user and datetime
        ratings = ratings.sort_values(['userId', 'datetime'])
        
        # Apply the same logic as create_splits
        g = ratings.groupby('userId', group_keys=False)
        ratings['rank_from_end'] = g.cumcount(ascending=False)
        ratings['user_count'] = g['userId'].transform('size')
        
        n_test = 5
        n_val = 5
        
        # Create splits
        train = ratings[(ratings.user_count > 10) & (ratings.rank_from_end >= (n_val + n_test))].copy()
        val = ratings[(ratings.user_count > 10) & (ratings.rank_from_end < (n_val + n_test)) & (ratings.rank_from_end >= n_test)].copy()
        test = ratings[(ratings.user_count > 10) & (ratings.rank_from_end < n_test)].copy()
        
        # Verify splits
        assert len(train) > 0
        assert len(val) == 10  # 2 users * 5 val items
        assert len(test) == 10  # 2 users * 5 test items
        
        # Verify chronological ordering: max datetime in train < min datetime in val < min datetime in test
        for uid in [1, 2]:
            user_ratings = ratings[ratings.userId == uid].sort_values('datetime')
            user_train = train[train.userId == uid]
            user_val = val[val.userId == uid]
            user_test = test[test.userId == uid]
            
            if len(user_train) > 0 and len(user_val) > 0:
                assert user_train.datetime.max() < user_val.datetime.min()
            if len(user_val) > 0 and len(user_test) > 0:
                assert user_val.datetime.max() < user_test.datetime.min()


class TestConfigLoading:
    """Test configuration loading."""
    
    def test_config_defaults(self):
        """Test that config module provides defaults."""
        from src.utils.config import load_config
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({'seed': 42}, f)
            config_path = f.name
        
        try:
            cfg = load_config(config_path)
            # Check defaults are applied
            assert 'seed' in cfg
            assert cfg['seed'] == 42
            assert 'data_root' in cfg  # Should have default
            assert 'k_values' in cfg  # Should have default
        finally:
            os.unlink(config_path)


class TestDataLoading:
    """Test data loading functionality."""
    
    def test_synthetic_dataframe_operations(self):
        """Test basic DataFrame operations used in data processing."""
        # Simulate movie data
        movie_data = pd.DataFrame({
            'imdbId': [1, 2, 3],
            'original_language': ['en', 'fr', 'es'],
            'genres': [['Action', 'Drama'], ['Comedy'], ['Horror', 'Thriller']]
        })
        
        assert len(movie_data) == 3
        assert 'original_language' in movie_data.columns
        
        # Test exploding genres
        exploded = movie_data.explode('genres')
        assert len(exploded) > len(movie_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
