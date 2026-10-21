import os
import sys

# The codebase uses top-level imports (`from agents import ...`), so tests run with flowrl/ on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
