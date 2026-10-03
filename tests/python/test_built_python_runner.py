# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The build-tree runner isolates the interpreters that a test starts."""

from __future__ import annotations

import os
import subprocess
import venv
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "run_built_python.py"

# What an editable install does at startup: a finder that sends `c50` to
# another directory.
_STALE_FINDER = """\
import importlib.util
import os
import sys


class Finder:
    @staticmethod
    def find_spec(name, path=None, target=None):
        if name != "c50":
            return None
        return importlib.util.spec_from_file_location(
            name,
            os.path.join(os.environ["STALE_C50"], "c50", "__init__.py"),
            submodule_search_locations=[
                os.path.join(os.environ["STALE_C50"], "c50")
            ],
        )


sys.meta_path.insert(0, Finder)
"""

_PROBE = """\
import subprocess
import sys

import c50._c50

print("parent", c50._c50.__file__)
child = subprocess.run(
    [sys.executable, "-c", "import c50._c50; print(c50._c50.__file__)"],
    capture_output=True,
    text=True,
    check=False,
)
print("child", child.stdout.strip())
print("status", child.returncode, child.stderr.strip())
"""


def _package(root: Path) -> None:
    (root / "c50").mkdir(parents=True)
    (root / "c50" / "__init__.py").write_text("")
    (root / "c50" / "_c50.py").write_text("")


def test_child_interpreters_import_the_requested_build(
    tmp_path: Path,
) -> None:
    build = tmp_path / "build"
    stale = tmp_path / "stale"
    _package(build)
    _package(stale)
    environment = tmp_path / "environment"
    try:
        venv.create(environment, symlinks=os.name != "nt", with_pip=False)
        builder = venv.EnvBuilder()
        python = builder.ensure_directories(environment).env_exe
    except (OSError, subprocess.CalledProcessError) as error:
        pytest.skip(f"cannot create a virtual environment: {error}")
    site = next((environment / "lib").glob("python*/site-packages"), None)
    if site is None:
        site = environment / "Lib" / "site-packages"
    (site / "stale_finder.py").write_text(_STALE_FINDER)
    (site / "stale.pth").write_text("import stale_finder\n")
    probe = tmp_path / "probe.py"
    probe.write_text(_PROBE)

    result = subprocess.run(
        [python, str(RUNNER), str(build), str(probe)],
        env={
            **{k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
            "STALE_C50": str(stale),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    lines = dict(line.split(" ", 1) for line in result.stdout.splitlines())
    for name in ("parent", "child"):
        assert Path(lines[name]).resolve().is_relative_to(build.resolve()), (
            result.stdout
        )
    assert lines["status"].startswith("0 "), result.stdout
