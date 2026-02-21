#!/usr/bin/env python
"""Burgers' equation experiment.

Runs the full pipeline for the 1-D Burgers equation:
    a(x) → u(x, t=1)

Usage (from project root):
    python experiments/burger/run_burger.py
    python experiments/burger/run_burger.py --device cuda:0 --optimizer ngd
    python experiments/burger/run_burger.py --epochs 50 --seed 42
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from experiments.run import main as run_main

# Default config for this experiment
DEFAULT_CONFIG = "configs/burger.yaml"

if __name__ == "__main__":
    # Inject default config if not provided
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", DEFAULT_CONFIG])
    run_main()
