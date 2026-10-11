#!/usr/bin/env python3
"""
Stand-alone script to compile visualizations and markdown research report.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from statarb.cli import run_full_pipeline
from statarb.config import AppConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate research report and figures")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    args = parser.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    code = run_full_pipeline(cfg)
    sys.exit(code)


if __name__ == "__main__":
    main()
