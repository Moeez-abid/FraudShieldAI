"""Small shared helpers: config loading, hashing, run metadata."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(name: str) -> dict:
    """Load configs/<name>.yaml relative to the project root."""
    with (PROJECT_ROOT / "configs" / f"{name}.yaml").open() as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def frame_hash(df: pd.DataFrame, cols: list[str]) -> str:
    """Order-sensitive hash of selected columns, used to fingerprint splits."""
    h = hashlib.sha256()
    for row in df[cols].astype(str).itertuples(index=False):
        h.update("\x1f".join(row).encode())
        h.update(b"\x1e")
    return h.hexdigest()[:16]


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
