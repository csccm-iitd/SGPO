#!/usr/bin/env python
"""Darcy triangular-notch experiment.

Runs the full pipeline for 2-D Darcy flow with triangular notch:
    boundCoeff(x,y) → sol(x,y)

Usage (from project root):
    python experiments/darcy_notch/run_darcy_notch.py
    python experiments/darcy_notch/run_darcy_notch.py --device cuda:0
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from experiments.run import main as run_main

DEFAULT_CONFIG = "configs/darcy_notch.yaml"

if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", DEFAULT_CONFIG])
    run_main()
