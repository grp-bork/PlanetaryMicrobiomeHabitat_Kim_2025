"""Utilities for initializing notebooks in a reproducible environment."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

REPO_NAME = "Habitat_Clustering_Code_Availability"


def ensure_project_root(repo_name: str = REPO_NAME) -> Tuple[Path, Path]:
    """Return (PROJECT_ROOT, DATA_DIR) and guarantee the root is importable.

    Many IDEs launch kernels from the parent directory of this repository.
    This helper walks up one level, finds the repository root, and appends it
    to ``sys.path`` so that ``scripts.*`` modules can be imported reliably.
    """

    project_root = Path().resolve()
    if not (project_root / "scripts").exists():
        candidate = project_root / repo_name
        if (candidate / "scripts").exists():
            project_root = candidate

    if not (project_root / "scripts").exists():
        raise RuntimeError(
            f"Could not locate 'scripts' directory from {project_root}. "
            "Ensure the notebook is executed inside the repository."
        )

    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.append(project_root_str)

    data_dir = project_root / "data"
    return project_root, data_dir

