"""Make `import sparkprof` work when scripts are run directly (python scripts/0X_*.py)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
