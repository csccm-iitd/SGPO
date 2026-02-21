#!/usr/bin/env python
"""Navier-Stokes experiment.

Runs the full pipeline for 2-D Navier-Stokes vorticity equation:
    ω_in(x,y) → ω_out(x,y)

Usage (from project root):
    python experiments/navier/run_navier.py
    python experiments/navier/run_navier.py --device cuda:0 --optimizer muon
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from experiments.run import main as run_main

DEFAULT_CONFIG = "configs/navier.yaml"

if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", DEFAULT_CONFIG])
    run_main()
