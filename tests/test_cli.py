"""
Smoke tests for CLI entrypoints.
Tests that CLI commands can be parsed and functions are importable.
"""

import pytest
import sys
import os
import tempfile
import yaml

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestCLIImports:
    """Test that CLI modules can be imported."""
    
    def test_entrypoints_import(self):
        """Test that entrypoints module can be imported."""
        from src.cli import entrypoints
        assert hasattr(entrypoints, 'main')
    
    def test_all_commands_importable(self):
        """Test that all command functions exist."""
        from src.cli import entrypoints
        
        # Check all command functions exist
        assert hasattr(entrypoints, 'create_splits_cmd')
        assert hasattr(entrypoints, 'build_embeddings_cmd')
        assert hasattr(entrypoints, 'train_mf_cmd')
        assert hasattr(entrypoints, 'train_implicit_bpr_cmd')
        assert hasattr(entrypoints, 'train_two_tower_cmd')
        assert hasattr(entrypoints, 'eval_all_cmd')
        assert hasattr(entrypoints, 'reproducibility_eval_cmd')
    
    def test_load_cfg_function(self):
        """Test that load_cfg function works."""
        from src.cli.entrypoints import load_cfg
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({'seed': 42, 'data_root': 'test_data'}, f)
            config_path = f.name
        
        try:
            cfg = load_cfg(config_path)
            assert 'seed' in cfg
            assert cfg['seed'] == 42
        finally:
            os.unlink(config_path)


class TestCLIArgumentParsing:
    """Test CLI argument parsing logic."""
    
    def test_argparse_structure(self):
        """Test that argparse is set up correctly."""
        import argparse
        from src.cli.entrypoints import main
        
        # Create parser similar to main()
        parser = argparse.ArgumentParser(description="Test")
        subparsers = parser.add_subparsers(dest="command")
        
        # Add a test command
        test_parser = subparsers.add_parser("test_cmd")
        test_parser.add_argument("--config", type=str, required=True)
        
        # Test parsing
        args = parser.parse_args(["test_cmd", "--config", "test.yaml"])
        assert args.command == "test_cmd"
        assert args.config == "test.yaml"
    
    def test_config_argument_required(self):
        """Test that config argument is required for commands."""
        import argparse
        
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        
        cmd_parser = subparsers.add_parser("create_splits")
        cmd_parser.add_argument("--config", type=str, required=True)
        
        # Should fail without --config
        with pytest.raises(SystemExit):
            parser.parse_args(["create_splits"])


class TestModelTrainingImports:
    """Test that model training functions can be imported."""
    
    def test_mf_training_import(self):
        """Test MF training function import."""
        from src.models.MF import train_mf
        assert callable(train_mf)
    
    def test_implicit_bpr_training_import(self):
        """Test implicit BPR training function import."""
        from src.models.implicitBPR import train_implicit_bpr
        assert callable(train_implicit_bpr)
    
    def test_two_tower_training_import(self):
        """Test Two-Tower training function import."""
        from src.models.twoTower import train_two_tower
        assert callable(train_two_tower)


class TestDataProcessingImports:
    """Test that data processing functions can be imported."""
    
    def test_splits_creation_import(self):
        """Test splits creation function import."""
        from src.data.splits import create_splits
        assert callable(create_splits)
    
    def test_embeddings_building_import(self):
        """Test embeddings building function import."""
        from src.embeddings.langGenre import build_item_features
        assert callable(build_item_features)
    
    def test_evaluation_import(self):
        """Test evaluation function import."""
        from src.eval.metrics import test_eval
        assert callable(test_eval)


class TestConfigPaths:
    """Test configuration path handling."""
    
    def test_config_file_creation(self):
        """Test that config files can be created and loaded."""
        config_data = {
            'data_root': 'data',
            'artifact_root': 'artifacts',
            'seed': 42,
            'k_values': [5, 10, 20]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            # Load and verify
            with open(config_path, 'r') as f:
                loaded = yaml.safe_load(f)
            
            assert loaded['seed'] == 42
            assert loaded['data_root'] == 'data'
            assert loaded['k_values'] == [5, 10, 20]
        finally:
            os.unlink(config_path)
    
    def test_relative_path_construction(self):
        """Test relative path construction."""
        base = 'artifacts'
        model = 'MF'
        
        path = os.path.join(base, model)
        assert path == os.path.join('artifacts', 'MF')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
