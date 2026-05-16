"""Utility functions: config loading, logging, seed control, bounds checking.

All modules import from here. No upstream project dependencies.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed YAML contents.

    Raises
    ------
    FileNotFoundError
        If the file does not exist (with a helpful message).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Config file not found: {p.resolve()}\n"
            f"Check that you are running from the transformer_lol_sim/ directory."
        )
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_project_root() -> Path:
    """Return the project root directory (parent of src/).

    Works regardless of which subdirectory the script is called from.
    """
    this_file = Path(__file__).resolve()
    # src/transformer_lol/utils.py -> go up 3 levels to project root
    return this_file.parent.parent.parent


def get_paths(paths_yaml: str | Path | None = None) -> dict[str, Path]:
    """Load paths from config/paths.yaml and return as absolute Path objects.

    Parameters
    ----------
    paths_yaml : str or Path, optional
        Override path to paths.yaml. Defaults to config/paths.yaml
        relative to the project root.

    Returns
    -------
    dict[str, Path]
        Flat dict mapping key names to absolute Path objects.
        Keys: data.raw, data.processed, data.synthetic, results.figures,
        results.tables, results.metrics, results.logs, results.mc_results,
        results.surrogate_model, results.surrogate_scaler.
    """
    root = get_project_root()
    if paths_yaml is None:
        paths_yaml = root / "config" / "paths.yaml"
    raw = load_yaml(paths_yaml)

    flat: dict[str, Path] = {}
    for section, entries in raw.items():
        if isinstance(entries, dict):
            for key, rel_path in entries.items():
                flat[f"{section}.{key}"] = root / rel_path
    return flat


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(
    log_dir: str | Path | None = None,
    level: str = "INFO",
    name: str = "transformer_lol",
) -> logging.Logger:
    """Configure and return a logger that writes to console and optionally a file.

    Parameters
    ----------
    log_dir : str or Path, optional
        Directory to write ``transformer_lol.log``. If None, file handler is skipped.
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR).
    name : str
        Logger name.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured in this process

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_dir is not None:
        log_path = Path(log_dir) / "transformer_lol.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Seed control
# ---------------------------------------------------------------------------

def set_global_seeds(seed: int) -> None:
    """Set seeds for numpy, random, and xgboost for full reproducibility.

    Parameters
    ----------
    seed : int
        Master seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import xgboost as xgb  # noqa: F401
        # XGBoost uses its own internal RNG controlled via random_state param;
        # there is no global XGBoost seed API, so we seed numpy only.
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_bounds(
    value: float,
    name: str,
    lo: float,
    hi: float,
) -> None:
    """Raise ValueError if value is outside [lo, hi].

    Never silently clips — all validation failures are hard errors.

    Parameters
    ----------
    value : float
    name : str
        Human-readable parameter name for the error message.
    lo, hi : float
        Inclusive lower and upper bounds.

    Raises
    ------
    ValueError
    """
    if not (lo <= value <= hi):
        raise ValueError(
            f"Parameter '{name}' = {value} is outside valid range [{lo}, {hi}]."
        )


def validate_array_bounds(
    arr: np.ndarray,
    name: str,
    lo: float,
    hi: float,
) -> None:
    """Raise ValueError if any element of arr is outside [lo, hi].

    Parameters
    ----------
    arr : np.ndarray
    name : str
    lo, hi : float

    Raises
    ------
    ValueError
    """
    if np.any(arr < lo) or np.any(arr > hi):
        raise ValueError(
            f"Array '{name}' contains values outside [{lo}, {hi}]. "
            f"Min={arr.min():.3f}, Max={arr.max():.3f}."
        )
