# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run a Python check against the CMake build, also when editable."""

from __future__ import annotations

import importlib.util
import os
import runpy
import sys
from pathlib import Path
from types import ModuleType

SITE_DIRECTORY = Path(__file__).resolve().parent / "built_python_site"


def load_isolation() -> ModuleType:
    """Run the startup module that the interpreters of a test also run."""
    spec = importlib.util.spec_from_file_location(
        "c50_built_python_isolation", SITE_DIRECTORY / "sitecustomize.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SITE_DIRECTORY}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    target = sys.argv[2]
    sys.path.insert(0, str(root))
    # Descendant interpreters read the environment. Each puts the build first
    # on its path and removes the editable-install finders that its startup
    # files install. Loading the startup module does the same here.
    os.environ["C50_BUILT_PYTHON_ROOT"] = str(root)
    inherited = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(SITE_DIRECTORY), str(root), *([inherited] if inherited else [])]
    )
    load_isolation()
    import c50
    import c50._c50

    for module in (c50, c50._c50):
        if module.__file__ is None:
            raise RuntimeError(f"{module.__name__} has no file")
        loaded = Path(module.__file__).resolve()
        if not loaded.is_relative_to(root):
            raise RuntimeError(
                f"{module.__name__} loaded from {loaded}, expected {root}"
            )
    print(f"Testing native extension: {c50._c50.__file__}", flush=True)
    sys.argv = [target, *sys.argv[3:]]
    if target.endswith(".py"):
        runpy.run_path(target, run_name="__main__")
    else:
        runpy.run_module(target, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
