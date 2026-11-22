# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Windows wheel check recognizes a bundled Microsoft runtime DLL.

The container job that runs `tests/windows_clean_import.py` needs Windows. This
test runs the archive check of that script on any platform.
"""

from __future__ import annotations

import importlib.util
import struct
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


def pe_image(imports: dict[str, list[str]], *, wide: bool = True) -> bytes:
    """Build a minimal PE image with one section that holds the imports."""
    width = 8 if wide else 4
    base = 0x1000
    raw = 0x200
    blob = bytearray()

    def add(data: bytes) -> int:
        position = len(blob)
        blob.extend(data)
        return base + position

    count = len(imports)
    table = add(bytes(20 * (count + 1)))
    descriptors = bytearray()
    for dll, names in imports.items():
        hints = [add(b"\0\0" + name.encode() + b"\0") for name in names]
        thunks = b"".join(
            hint.to_bytes(width, "little") for hint in hints
        ) + bytes(width)
        lookup = add(thunks)
        dll_name = add(dll.encode() + b"\0")
        descriptors += struct.pack("<IIIII", lookup, 0, 0, dll_name, lookup)
    descriptors += bytes(20)
    blob[0 : len(descriptors)] = descriptors

    optional = bytearray(240 if wide else 224)
    struct.pack_into("<H", optional, 0, 0x20B if wide else 0x10B)
    struct.pack_into("<I", optional, 112 + 8 if wide else 96 + 8, table)
    coff = struct.pack("<HHIIIHH", 0x8664, 1, 0, 0, 0, len(optional), 0)
    section = struct.pack(
        "<8sIIIIIIHHI",
        b".idata",
        len(blob),
        base,
        len(blob),
        raw,
        0,
        0,
        0,
        0,
        0,
    )
    header = bytearray(raw)
    struct.pack_into("<2s", header, 0, b"MZ")
    struct.pack_into("<I", header, 0x3C, 0x40)
    header[0x40:0x44] = b"PE\0\0"
    body = coff + bytes(optional) + section
    header[0x44 : 0x44 + len(body)] = body
    return bytes(header) + bytes(blob)


MUTEX = {"MSVCP140.dll": ["_Mtx_lock", "_Mtx_init_in_situ"]}
NO_MUTEX = {"MSVCP140.dll": ["_Mtx_lock", "_Cnd_wait"]}


def extension_wheel(path: Path, image: bytes) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("c50/__init__.py", b"")
        archive.writestr("c50/_c50.cp312-win_amd64.pyd", image)
    return path


@pytest.mark.parametrize("wide", [True, False])
def test_pe_reader_lists_imports_by_name(wide: bool) -> None:
    imports = {**MUTEX, "KERNEL32.dll": ["Sleep"]}
    assert CHECK.pe_imports(pe_image(imports, wide=wide)) == {
        ("msvcp140.dll", "_Mtx_lock"),
        ("msvcp140.dll", "_Mtx_init_in_situ"),
        ("kernel32.dll", "Sleep"),
    }


def test_extension_that_imports_the_mutex_initializer_passes(
    tmp_path: Path,
) -> None:
    path = extension_wheel(tmp_path / "good.whl", pe_image(MUTEX))
    assert CHECK.run_time_mutex(path) == []


def test_extension_without_the_mutex_initializer_is_reported(
    tmp_path: Path,
) -> None:
    path = extension_wheel(tmp_path / "bad.whl", pe_image(NO_MUTEX))
    assert CHECK.run_time_mutex(path) == ["c50/_c50.cp312-win_amd64.pyd"]


def test_initializer_from_another_library_does_not_count(
    tmp_path: Path,
) -> None:
    image = pe_image({"OTHER.dll": ["_Mtx_init_in_situ"]})
    path = extension_wheel(tmp_path / "other.whl", image)
    assert CHECK.run_time_mutex(path) == ["c50/_c50.cp312-win_amd64.pyd"]


def test_wheel_check_fails_for_an_extension_without_the_initializer(
    tmp_path: Path,
) -> None:
    path = extension_wheel(tmp_path / "bad.whl", pe_image(NO_MUTEX))
    with pytest.raises(AssertionError, match="_Mtx_init_in_situ"):
        CHECK.check_wheel(path)


def test_wheel_without_an_extension_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no extension"):
        CHECK.run_time_mutex(wheel(tmp_path / "none.whl", ["c50/a.py"]))


def test_input_that_is_not_a_pe_image_is_an_error() -> None:
    with pytest.raises(ValueError, match="MZ"):
        CHECK.pe_imports(b"\x7fELF" + bytes(100))
    with pytest.raises(ValueError, match="malformed"):
        CHECK.pe_imports(b"MZ" + bytes(10))
