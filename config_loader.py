"""Canonical configuration loader for EECE 499 PINN-RL project.

This module loads ``config_frozen_v1.yaml`` and provides:
  - get_config(): returns the full config as a dict
  - get_method_config(method_name): returns method-specific defaults
  - get_tier_config(tier_name): returns tier-specific defaults
  - check_config_lock(): verifies config_lock.json matches the current YAML hash
  - generate_config_lock(): generates config_lock.json

Phase 1F: lives at the repo root so any module can import via
    from config_loader import get_config

Loading is cached on first call; pass ``reload=True`` to force a re-read
(only used in verification tests).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError as e:  # pragma: no cover - PyYAML is a hard dep
    raise ImportError(
        "config_loader requires PyYAML. Install with: pip install pyyaml"
    ) from e


CONFIG_FILE = "config_frozen_v1.yaml"
LOCK_FILE = "config_lock.json"


def _find_config_root(start: Optional[Path] = None) -> Path:
    """Walk up from cwd (or ``start``) until ``config_frozen_v1.yaml`` is found.

    Raises FileNotFoundError if no parent directory contains the canonical
    config file. This is treated as a hard error: the project cannot run
    without a frozen config after Phase 1F.
    """
    cur = (start or Path(os.getcwd())).resolve()
    for parent in [cur, *cur.parents]:
        if (parent / CONFIG_FILE).exists():
            return parent
    # Fallback: also check next to this module (handles odd cwd cases)
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        if (parent / CONFIG_FILE).exists():
            return parent
    raise FileNotFoundError(
        f"Could not find {CONFIG_FILE} in cwd or any parent directory. "
        "Phase 1F establishes this as the canonical config; ensure it exists "
        "at repo root."
    )


_CONFIG_CACHE: Dict[str, Any] = {}


def get_config(reload: bool = False) -> Dict[str, Any]:
    """Load and return the full frozen config as a dict.

    Caches on first load; pass ``reload=True`` to force a re-read.
    """
    global _CONFIG_CACHE
    if not reload and _CONFIG_CACHE:
        return _CONFIG_CACHE

    root = _find_config_root()
    config_path = root / CONFIG_FILE
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(
            f"{CONFIG_FILE} did not parse to a dict; got {type(config)!r}."
        )

    _CONFIG_CACHE = config
    return config


def get_method_config(method_name: str) -> Dict[str, Any]:
    """Get method-specific defaults from frozen config."""
    config = get_config()
    methods = config.get("methods", {})
    if method_name not in methods:
        raise KeyError(
            f"Method '{method_name}' not in frozen config. "
            f"Known methods: {sorted(methods.keys())}"
        )
    return methods[method_name]


def get_tier_config(tier_name: str) -> Dict[str, Any]:
    """Get tier-specific configuration."""
    config = get_config()
    if tier_name not in config:
        raise KeyError(
            f"Tier '{tier_name}' not in frozen config. "
            "Known tiers: tier1, tier2, tier3, tier4, supplementary"
        )
    return config[tier_name]


def compute_config_hash() -> str:
    """SHA256 hash of the current ``config_frozen_v1.yaml`` (raw bytes)."""
    root = _find_config_root()
    config_path = root / CONFIG_FILE
    with open(config_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_config_lock(strict: bool = False) -> Dict[str, Any]:
    """Verify the config hasn't been modified since the lock was created.

    Args:
        strict: if True, raise an exception on hash mismatch instead of
                returning a warning dict.

    Returns:
        dict with keys: matches (bool), expected_hash (str|None),
        actual_hash (str), warning (str|None).
    """
    root = _find_config_root()
    lock_path = root / LOCK_FILE
    actual_hash = compute_config_hash()

    if not lock_path.exists():
        msg = (
            f"{LOCK_FILE} not found. Phase 1F should generate this; "
            "proceeding without lock check."
        )
        if strict:
            raise FileNotFoundError(msg)
        return {
            "matches": False,
            "expected_hash": None,
            "actual_hash": actual_hash,
            "warning": msg,
        }

    with open(lock_path) as f:
        lock_data = json.load(f)
    expected_hash = lock_data["config_hash"]

    matches = expected_hash == actual_hash
    if not matches:
        warning = (
            f"\n⚠️  CONFIG HASH MISMATCH: {CONFIG_FILE} has been modified "
            f"since lock was created.\n"
            f"   Expected: {expected_hash[:16]}...\n"
            f"   Actual:   {actual_hash[:16]}...\n"
            f"   Lock created: {lock_data.get('created_iso', 'unknown')}\n"
            f"   This may invalidate paper claims. If the change is "
            f"intentional, regenerate the lock with:\n"
            f"     python3 -c 'from config_loader import "
            f"generate_config_lock; generate_config_lock()'\n"
            f"   Or rename to config_frozen_v2.yaml if this is a major "
            f"version bump.\n"
        )
        if strict:
            raise RuntimeError(warning)
        return {
            "matches": False,
            "expected_hash": expected_hash,
            "actual_hash": actual_hash,
            "warning": warning,
        }
    return {
        "matches": True,
        "expected_hash": expected_hash,
        "actual_hash": actual_hash,
        "warning": None,
    }


def generate_config_lock() -> Path:
    """Generate ``config_lock.json`` from the current ``config_frozen_v1.yaml``.

    Returns the path to the written lock file.
    """
    root = _find_config_root()
    lock_path = root / LOCK_FILE
    config_hash = compute_config_hash()
    lock_data = {
        "config_file": CONFIG_FILE,
        "config_hash": config_hash,
        "created_iso": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "note": (
            f"This file was generated when {CONFIG_FILE} was frozen on "
            f"Phase 1F. Hash mismatch at runtime indicates the config has "
            f"been modified since freeze. If the change is intentional, "
            f"regenerate this lock or version-bump the config to v2."
        ),
    }
    with open(lock_path, "w") as f:
        json.dump(lock_data, f, indent=2)
    print(f"Generated {lock_path} with hash {config_hash[:16]}...")
    return lock_path


__all__ = [
    "CONFIG_FILE",
    "LOCK_FILE",
    "get_config",
    "get_method_config",
    "get_tier_config",
    "compute_config_hash",
    "check_config_lock",
    "generate_config_lock",
]
