########################################
# PowerShell Smoke Test Script
# Windows version of smoke_test.sh
########################################

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Running JustiFlicks Smoke Tests" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if pytest is installed
$pytestExists = Get-Command pytest -ErrorAction SilentlyContinue
if (-not $pytestExists) {
    Write-Host "ERROR: pytest not found. Please install with: pip install pytest" -ForegroundColor Red
    exit 1
}

# Check if required directories exist
if (-not (Test-Path "src")) {
    Write-Host "ERROR: src directory not found" -ForegroundColor Red
    exit 2
}

if (-not (Test-Path "tests")) {
    Write-Host "ERROR: tests directory not found" -ForegroundColor Red
    exit 2
}

Write-Host "Step 1: Testing basic imports..." -ForegroundColor Yellow

# Test basic imports
$importTest = @"
import sys
try:
    import src
    from src.models import embeddings
    from src.cli import entrypoints
    from src.data import splits
    from src.eval import metrics
    print('[OK] All basic imports successful')
    sys.exit(0)
except Exception as e:
    print('[ERROR] Import failed:', e)
    sys.exit(2)
"@

$result = python -c $importTest
$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
    Write-Host "Basic import tests failed" -ForegroundColor Red
    exit 2
}

Write-Host $result -ForegroundColor Green
Write-Host ""
Write-Host "Step 2: Running pytest test suite..." -ForegroundColor Yellow
Write-Host ""

# Run pytest with verbose output
pytest tests/ -v --tb=short

$testExitCode = $LASTEXITCODE

Write-Host ""
if ($testExitCode -eq 0) {
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "[PASS] All smoke tests passed successfully!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    exit 0
} else {
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "[FAIL] Some tests failed" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    exit $testExitCode
}
