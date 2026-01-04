#!/usr/bin/env bash
set -euo pipefail

echo "========================================"
echo "Running JustiFlicks Smoke Tests"
echo "========================================"
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "ERROR: pytest not found. Please install with: pip install pytest"
    exit 1
fi

# Check if required directories exist
if [ ! -d "src" ]; then
    echo "ERROR: src directory not found"
    exit 2
fi

if [ ! -d "tests" ]; then
    echo "ERROR: tests directory not found"
    exit 2
fi

echo "Step 1: Testing basic imports..."
python - <<'PY'
import sys
try:
    import src
    from src.models import embeddings
    from src.cli import entrypoints
    from src.data import splits
    from src.eval import metrics
    print("✓ All basic imports successful")
except Exception as e:
    print("✗ Import failed:", e)
    sys.exit(2)
PY

if [ $? -ne 0 ]; then
    echo "Basic import tests failed"
    exit 2
fi

echo ""
echo "Step 2: Running pytest test suite..."
echo ""

# Run pytest with verbose output
pytest tests/ -v --tb=short

TEST_EXIT_CODE=$?

echo ""
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "========================================"
    echo "✓ All smoke tests passed successfully!"
    echo "========================================"
    exit 0
else
    echo "========================================"
    echo "✗ Some tests failed"
    echo "========================================"
    exit $TEST_EXIT_CODE
fi
