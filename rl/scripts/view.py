#!/usr/bin/env python3
"""Watch a trained agent play one episode at a time.

Usage:
  python scripts/view.py                        # random policy
  python scripts/view.py --model runs/models/X  # trained SB3 model

Controls:
  Space       pause / resume
  ← / →       slower / faster  (0.5× … max)
  R           restart episode
  Q / Esc     quit
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from viewer.runner import run


def main() -> None:
    p = argparse.ArgumentParser(description="ATB agent viewer")
    p.add_argument("--model", default=None,
                   help="Path to trained SB3 PPO model (omit for random policy)")
    args = p.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
