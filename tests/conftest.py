import sys
from pathlib import Path

# Ensure the project root is on sys.path when running pytest in importlib mode.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
