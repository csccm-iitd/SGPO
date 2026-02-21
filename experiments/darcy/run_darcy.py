#!/usr/bin/env python
"""Darcy flow experiment.

Runs the full pipeline for 2-D Darcy flow:
    a(x,y) → u(x,y)

Usage (from project root):
    python experiments/darcy/run_darcy.py
    python experiments/darcy/run_darcy.py --device cuda:0 --optimizer ngd
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from experiments.run import main as run_main

DEFAULT_CONFIG = "configs/darcy.yaml"

if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", DEFAULT_CONFIG])
    run_main()
