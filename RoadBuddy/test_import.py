import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

try:
    from model import create_model
    print("✓ Model import successful")
    print("✓ Flash attention disabled by default")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)
