"""Run a Python check against the CMake build, including in editable environments."""

from __future__ import annotations

import importlib.machinery
from pathlib import Path
import runpy
import sys
from typing import Any


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    target = sys.argv[2]
    sys.path.insert(0, str(root))

    def redirects_package(finder: Any) -> bool:
        if finder is importlib.machinery.PathFinder:
            return False
        try:
            spec = finder.find_spec("c50", None)
        except (AttributeError, ImportError, TypeError, ValueError):
            return False
        return (
            spec is not None and spec.origin is not None
            and not Path(spec.origin).resolve().is_relative_to(root)
        )

    sys.meta_path[:] = [f for f in sys.meta_path if not redirects_package(f)]
    import c50
    import c50._c50

    for module in (c50, c50._c50):
        loaded = Path(module.__file__).resolve()
        if not loaded.is_relative_to(root):
            raise RuntimeError(f"{module.__name__} loaded from {loaded}, expected {root}")
    print(f"Testing native extension: {c50._c50.__file__}", flush=True)
    sys.argv = [target, *sys.argv[3:]]
    if target.endswith(".py"):
        runpy.run_path(target, run_name="__main__")
    else:
        runpy.run_module(target, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
