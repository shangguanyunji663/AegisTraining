"""Portable roots for the isolated training repository."""
from __future__ import annotations

import os
from pathlib import Path


def training_root() -> Path:
    """Return the configured training root, defaulting to this checkout."""
    configured = os.environ.get("AEGIS_TRAINING_ROOT")
    return Path(configured).expanduser().resolve() if configured else Path(__file__).resolve().parents[3]


def project_root() -> Path | None:
    """Return the optional production checkout used for committed fixtures."""
    configured = os.environ.get("AEGIS_PROJECT_ROOT")
    return Path(configured).expanduser().resolve() if configured else None


def configured_project_corpus() -> Path | None:
    """Return an explicitly configured production fixture, if present."""
    configured = os.environ.get("AEGIS_PROJECT_CORPUS")
    return Path(configured).expanduser().resolve() if configured else None


def under(path: Path, root: Path, role: str) -> Path:
    """Resolve a path and ensure it stays inside an allowed root."""
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{role} path escapes allowed root: {resolved}") from exc
    return resolved


def resolve_from(root: Path, value: str | Path) -> Path:
    """Resolve an absolute path or a path relative to ``root``."""
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else root / candidate
