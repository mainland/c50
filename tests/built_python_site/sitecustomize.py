# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep an editable install from redirecting the c50 package.

``tests/run_built_python.py`` puts this directory on ``PYTHONPATH`` and names
the build root in ``C50_BUILT_PYTHON_ROOT``, so that every interpreter that a
test starts removes the import hooks of other copies of ``c50`` after the
startup ``.pth`` files installed them. The runner also loads this file for its
own process. ``sitecustomize`` runs after ``.pth`` processing, which is why the
removal happens here and not in an environment variable alone.
"""

from __future__ import annotations

import importlib.machinery
import os
import sys
from pathlib import Path
from typing import Any

ROOT_VARIABLE = "C50_BUILT_PYTHON_ROOT"  # Set by the runner.


def isolate(root: Path) -> None:
    """Drop the meta path finders that find ``c50`` outside ``root``."""

    def redirects_package(finder: Any) -> bool:
        if finder is importlib.machinery.PathFinder:
            return False
        try:
            spec = finder.find_spec("c50", None)
        except (AttributeError, ImportError, TypeError, ValueError):
            return False
        return (
            spec is not None
            and spec.origin is not None
            and not Path(spec.origin).resolve().is_relative_to(root)
        )

    sys.meta_path[:] = [f for f in sys.meta_path if not redirects_package(f)]


def _isolate_from_environment() -> None:
    root = os.environ.get(ROOT_VARIABLE)
    if root:
        isolate(Path(root).resolve())


_isolate_from_environment()
