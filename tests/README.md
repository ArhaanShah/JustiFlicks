# JustiFlicks Test Suite

This directory contains smoke tests for the JustiFlicks recommendation system. These tests verify that all components work correctly without running expensive model training or data processing operations.

## Test Files

- **test_cf.py** - Tests for collaborative filtering models (MF_BPR, Two-Tower, implicit BPR)
- **test_data.py** - Tests for data processing and splitting logic
- **test_embeddings.py** - Tests for feature engineering and embedding operations
- **test_eval.py** - Tests for evaluation metrics and slice-based analysis
- **test_cli.py** - Tests for CLI entrypoints and command parsing

## Running Tests

### Using pytest directly:
```bash
pytest tests/ -v
```

### Using the smoke test scripts:

**Linux/Mac:**
```bash
bash tests/smoke_test.sh
```

**Windows (PowerShell):**
```powershell
powershell tests/smoke_test.ps1
```

## Test Categories

Tests are organized into classes that test specific components:

1. **Model Tests** - Verify model initialization, forward passes, and basic functionality
2. **Data Processing Tests** - Test data transformations, binning, and split logic
3. **Embedding Tests** - Verify feature encoding and embedding operations
4. **Evaluation Tests** - Test metric calculations (NDCG, recall, precision)
5. **CLI Tests** - Verify command parsing and module imports

## Design Philosophy

These are **smoke tests** designed to:
- Run quickly (< 1 minute total)
- Require no external data files
- Use synthetic/minimal data
- Verify basic functionality without heavy computation
- Catch import errors and basic logic bugs

They do NOT:
- Train actual models
- Process full datasets
- Run end-to-end pipelines
- Test model accuracy

## Requirements

Install test dependencies:
```bash
pip install pytest numpy pandas torch pyyaml
```

## Adding New Tests

When adding new functionality:
1. Create corresponding test class in appropriate test file
2. Use synthetic data (small DataFrames, tensors)
3. Test basic operations, not full workflows
4. Follow naming convention: `test_<functionality>`
5. Add docstrings explaining what is tested

Example:
```python
def test_new_feature(self):
    """Test that new feature works with minimal data."""
    # Create minimal test case
    data = pd.DataFrame({'col': [1, 2, 3]})
    result = process_feature(data)
    # Assert expected behavior
    assert len(result) == 3
```
