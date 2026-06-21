#!/usr/bin/env python3
"""Run the ORACLE Mission Copilot FastAPI server."""

from __future__ import annotations

import sys
from pathlib import Path

# Add repository root to sys.path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Automatically create shared/__init__.py if missing
shared_init_path = repo_root / "shared" / "__init__.py"
if not shared_init_path.exists():
    shared_init_path.parent.mkdir(parents=True, exist_ok=True)
    shared_init_path.write_text(
        '"""Cross-model utilities shared by all LUNAR OS model packages."""\n\n'
        'from .dataset_utils import collate_tensor_dict, resolve_repo_paths\n'
        'from .geospatial_utils import crs_from_authority_code, pole_to_epsg, validate_crs\n'
        'from .lunar_constants import (\n'
        '    DEFAULT_NODATA,\n'
        '    DEFAULT_PIXEL_SIZE_M,\n'
        '    PATCH_SIZE,\n'
        '    POLAR_CRS,\n'
        '    Pole,\n'
        ')\n'
        'from .uncertainty_utils import dirichlet_entropy, normalize_uncertainty_map\n'
        'from .visualization import save_probability_png\n\n'
        '__all__ = [\n'
        '    "DEFAULT_NODATA",\n'
        '    "DEFAULT_PIXEL_SIZE_M",\n'
        '    "PATCH_SIZE",\n'
        '    "POLAR_CRS",\n'
        '    "Pole",\n'
        '    "collate_tensor_dict",\n'
        '    "crs_from_authority_code",\n'
        '    "dirichlet_entropy",\n'
        '    "normalize_uncertainty_map",\n'
        '    "pole_to_epsg",\n'
        '    "resolve_repo_paths",\n'
        '    "save_probability_png",\n'
        '    "validate_crs",\n'
        ']\n',
        encoding="utf-8"
    )

from shared.dataset_utils import ensure_import_paths
ensure_import_paths(Path(__file__).resolve().parent)


def main() -> None:
    import uvicorn
    # Import app explicitly to verify imports
    from Model_6_ORACLE_Mission_Copilot.src.api.app import app
    print("Starting ORACLE Mission Copilot API Server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
