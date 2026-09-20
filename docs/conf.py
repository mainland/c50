"""Sphinx configuration for the C5.0 GPL documentation."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOXYGEN_BUILD = ROOT / "build-docs-sphinx"
DOXYGEN_XML = DOXYGEN_BUILD / "docs" / "doxygen" / "xml"


def build_doxygen_xml() -> None:
    """Generate warning-checked Doxygen XML for Breathe."""
    subprocess.run(
        [
            "cmake",
            "-S",
            str(ROOT),
            "-B",
            str(DOXYGEN_BUILD),
            "-DC50_BUILD_DOCUMENTATION=ON",
            "-DBUILD_TESTING=OFF",
        ],
        check=True,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(DOXYGEN_BUILD),
            "--target",
            "c50-docs",
        ],
        check=True,
    )


build_doxygen_xml()

project = "C5.0 GPL"
author = "Geoffrey Mainland"
release = "2.07"

extensions = [
    "breathe",
    "myst_parser",
    "sphinx.ext.autodoc",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "furo"
html_static_path = ["_static"]
html_title = "C5.0 GPL"

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

myst_enable_extensions = [
    "colon_fence",
]

breathe_projects = {
    "C50": str(DOXYGEN_XML),
}
breathe_default_project = "C50"

autodoc_member_order = "bysource"
autodoc_typehints = "description"
