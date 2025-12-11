#!/usr/bin/env bash
set -euo pipefail
echo "Running smoke tests..."

python - <<'PY'
import sys
try:
    import src
    from src.models import embeddings
    print("Imports ok")
except Exception as e:
    print("Import failed:", e)
    sys.exit(2)
# quick CLI check for required files
import os
if not os.path.exists("src"):
    print("src not found")
    sys.exit(2)

print("Smoke tests passed")
PY

exit 0
