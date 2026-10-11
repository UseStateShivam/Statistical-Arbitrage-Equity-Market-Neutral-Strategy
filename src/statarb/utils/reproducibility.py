"""
Reproducibility and experiment provenance tracking.
"""

import hashlib
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across standard Python and NumPy for deterministic runs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def compute_file_hash(filepath: Path) -> str:
    """Computes SHA-256 hash of a file for integrity provenance."""
    if not filepath.exists():
        return "NOT_FOUND"
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_commit() -> str:
    """Attempts to retrieve the current git commit hash."""
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("ascii")
            .strip()
        )
        return commit
    except Exception:
        return "NO_GIT_REPO"


def create_manifest(
    experiment_id: str,
    config_dict: dict[str, Any],
    dataset_info: dict[str, Any],
    success: bool = True,
) -> dict[str, Any]:
    """
    Creates an institutional provenance manifest documenting environment, packages, and hashes.
    """
    manifest = {
        "experiment_id": experiment_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": None,
            "scipy": None,
            "statsmodels": None,
        },
        "random_seed": config_dict.get("experiment", {}).get("random_seed", 42),
        "dataset": dataset_info,
        "config_summary": {
            "slippage_bps": config_dict.get("execution", {}).get("slippage_bps", 5.0),
            "fill_convention": config_dict.get("execution", {}).get("fill_convention", "next_open"),
            "formation_days": config_dict.get("backtest", {}).get("formation_window_days", 252),
            "trading_days": config_dict.get("backtest", {}).get("trading_window_days", 63),
        },
        "success": success,
    }

    try:
        import pandas as pd

        manifest["packages"]["pandas"] = pd.__version__
    except ImportError:
        pass

    try:
        import scipy

        manifest["packages"]["scipy"] = scipy.__version__
    except ImportError:
        pass

    try:
        import statsmodels

        manifest["packages"]["statsmodels"] = statsmodels.__version__
    except ImportError:
        pass

    return manifest
