"""
Shared model path resolution for VIMU training scripts.

Models are organized by family and variant:
    <MODELS_DIR>/
        segmentation/
            <variant>/
                vimu_seg.pt
                vimu_seg.onnx
        pose/
            <variant>/
                best.pt
                vimu_pose.onnx

MODELS_DIR resolution order:
    1. --models-dir CLI argument (if the script supports it)
    2. MODELS_DIR environment variable (from .env)
    3. <repo_root>/models/  (auto-detected from git root)
"""

import os
import subprocess
from pathlib import Path


def _find_repo_root() -> Path:
    """Find the top-level workspace root by walking up from this file.

    Submodules have a .git *file* (pointer), real repos have a .git *directory*.
    We prefer the outermost real .git directory as the workspace root.
    """
    p = Path(__file__).resolve().parent
    best = None
    while p != p.parent:
        git = p / ".git"
        if git.is_dir():
            # Real repo — this is the best candidate so far
            best = p
        p = p.parent
    return best or Path(__file__).resolve().parent


def get_models_dir(cli_override: str | None = None) -> Path:
    """Resolve the models directory.

    Priority: cli_override > MODELS_DIR env var > <repo_root>/models/
    """
    if cli_override:
        return Path(cli_override)
    env = os.environ.get("MODELS_DIR")
    if env:
        return Path(env)
    return _find_repo_root() / "models"


def get_model_path(
    family: str,
    variant: str,
    filename: str,
    models_dir: str | None = None,
) -> Path:
    """Get the full path for a model file.

    Example: get_model_path("segmentation", "large_v1", "vimu_seg.pt")
    -> <models_dir>/segmentation/large_v1/vimu_seg.pt
    """
    return get_models_dir(models_dir) / family / variant / filename


def get_variant_dir(
    family: str,
    variant: str,
    models_dir: str | None = None,
) -> Path:
    """Get the directory for a specific variant.

    Example: get_variant_dir("segmentation", "large_v1")
    -> <models_dir>/segmentation/large_v1/
    """
    return get_models_dir(models_dir) / family / variant


def list_variants(family: str, models_dir: str | None = None) -> list[str]:
    """List all variants for a model family."""
    family_dir = get_models_dir(models_dir) / family
    if not family_dir.exists():
        return []
    return sorted(
        d.name for d in family_dir.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    )
