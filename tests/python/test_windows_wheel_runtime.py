# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Windows wheel check recognizes a bundled Microsoft runtime DLL.

The container job that runs `tests/windows_clean_import.py` needs Windows. This
test runs the archive check of that script on any platform.
"""

from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "windows_clean_import.py"
SPEC = importlib.util.spec_from_file_location("windows_clean_import", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)

PACKAGE = ["c50/__init__.py", "c50/_c50.cp312-win_amd64.pyd"]


def wheel(path: Path, members: list[str]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name in members:
            archive.writestr(name, b"")
    return path


def test_wheel_without_a_runtime_dll_passes(tmp_path: Path) -> None:
    path = wheel(tmp_path / "clean.whl", PACKAGE)
    assert CHECK.bundled_runtime(path) == []


@pytest.mark.parametrize(
    "name",
    [
        "c50.libs/msvcp140-a4c2229bdc2a2a630acdc095b4d86008.dll",
        "c50.libs/MSVCP140.dll",
        "c50.libs/msvcp140_1.dll",
        "c50.libs/vcruntime140_1.dll",
        "c50/concrt140.dll",
    ],
)
def test_wheel_with_a_runtime_dll_is_reported(
    tmp_path: Path, name: str
) -> None:
    path = wheel(tmp_path / "bundled.whl", [*PACKAGE, name])
    assert CHECK.bundled_runtime(path) == [name]
