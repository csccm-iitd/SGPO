#!/usr/bin/env python
"""Wave-equation experiment.

Runs the full pipeline for 1-D wave equation:
    u(x, 0) → u(x, t_final)

Usage (from project root):
    python experiments/wave/run_wave.py
    python experiments/wave/run_wave.py --device cuda:0 --optimizer ngd
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from experiments.run import main as run_main

DEFAULT_CONFIG = "configs/wave.yaml"

if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", DEFAULT_CONFIG])
    run_main()
