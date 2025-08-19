"""
cellforge: Open-Ended Autonomous Design of Computational Methods for Single-Cell Omics via Multi-Agent Collaboration

This is the main package entry point for cellforge.
"""

# Import the main cellforge package
from cellforge import *

# Version and metadata
__version__ = "0.1.0"
__author__ = "cellforge Team"

# Re-export main functions for convenience
__all__ = [
    "run_end_to_end_workflow",
    "cellforge"
]
