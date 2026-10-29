"""Paths and configurations for project-owned assets."""

from pathlib import Path

CABLE_ASSEMBLY_SIM_ASSETS_DIR = Path(__file__).resolve().parent / "data"
"""Fetched and generated asset data; see scripts/build_iiwa14.py."""

from .iiwa14 import IIWA14_2F85_CFG  # noqa: E402
