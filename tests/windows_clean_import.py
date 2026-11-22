# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check a Windows wheel against the Visual C++ Redistributable.

`tests/windows_clean_import.ps1` runs this script with an embeddable CPython in
a container, in four modes. The script needs only the standard library and the
wheel.

- `wheel WHEEL` checks that the wheel archive holds no Microsoft C++ runtime
  DLL, and that each extension module imports `_Mtx_init_in_situ` from
  `MSVCP140.dll`. The wheel relies on the Visual C++ Redistributable of the
  system.
- `absent` runs before the redistributable is installed. It checks that
  `import c50` fails with `ImportError`, so that the requirement is real.
- `present` runs after the installation. It checks that `c50` imports, that the
  process uses the system copy of `msvcp140.dll`, and that a text-input model
  of each kind trains and predicts.
- `stale CASE DLL` checks that a process that uses the older `msvcp140.dll` at
  the path DLL can lock a mutex. The case is `system`, `conda`, or
  `preloaded`, as described below.

An extension built without `_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR` leaves the
state of each `std::mutex` zero. An `msvcp140.dll` older than 14.38 reads a
function pointer from that state at the first lock, so the process dies with an
access violation. Three situations put such a copy in the process, and each is
a case of the `stale` mode:

- `system` plants the old copy next to `python.exe` before the redistributable
  is installed, the way a system with an old redistributable supplies it.
- `conda` plants it next to `python.exe` after the installation, the way a
  stale conda environment does. The application directory precedes `System32`
  in the search order.
- `preloaded` loads the old copy with `ctypes.WinDLL` before `import c50`, the
  way another package would. The loader reuses a loaded module of the same
  name.

Each case runs in a child process, so that a crash is a failure of that case
and the other cases still run. The child checks that the process loaded the
planted copy, so that a pass cannot come from the system copy. It then locks
the `Predictor` mutex with `prepare_predictor().predict_details` and runs the
multi-worker training path with `split_workers=2`, which starts `std::async`
tasks that construct a mutex and a condition variable.
"""

from __future__ import annotations

import ctypes
import os
import random
import re
import struct
import subprocess
import sys
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import c50

# The Microsoft C++ runtime libraries that a wheel must not carry. The name may
# have a hash suffix, as a repair tool gives a bundled copy.
RUNTIME_DLL = re.compile(
    r"(?:msvcp|vcruntime|concrt|vcomp|vccorlib)[\w-]*\.dll", re.IGNORECASE
)


def bundled_runtime(wheel: Path) -> list[str]:
    """Return the members of the wheel that are Microsoft runtime DLLs."""
    with zipfile.ZipFile(wheel) as archive:
        return sorted(
            name
            for name in archive.namelist()
            if RUNTIME_DLL.fullmatch(name.rsplit("/", 1)[-1])
        )


# The extension must construct its mutexes at run time. A `std::mutex` built by
# a constexpr constructor leaves the state that an `msvcp140.dll` older than
# the toolset needs unset, and the first lock then dereferences a null pointer.
# The run-time constructor imports `_Mtx_init_in_situ`.
MUTEX_INIT = ("msvcp140.dll", "_Mtx_init_in_situ")


def pe_imports(image: bytes) -> set[tuple[str, str]]:
    """Return the (lowercase DLL, function) pairs that a PE image imports.

    Imports by ordinal are omitted. Raise `ValueError` on a malformed image.
    """
    try:
        return _pe_imports(image)
    except (struct.error, IndexError) as error:
        raise ValueError(f"truncated or malformed PE image: {error}") from None


def _cstring(image: bytes, offset: int) -> str:
    end = image.index(b"\0", offset)
    return image[offset:end].decode("ascii")


def _pe_imports(image: bytes) -> set[tuple[str, str]]:
    if image[:2] != b"MZ":
        raise ValueError("not a PE image: no MZ signature")
    (pe,) = struct.unpack_from("<I", image, 0x3C)
    if image[pe : pe + 4] != b"PE\0\0":
        raise ValueError("not a PE image: no PE signature")
    coff = pe + 4
    sections, optional_size = struct.unpack_from("<2xH12xH2x", image, coff)
    optional = coff + 20
    (magic,) = struct.unpack_from("<H", image, optional)
    if magic == 0x10B:
        directories, thunk = optional + 96, "<I"
    elif magic == 0x20B:
        directories, thunk = optional + 112, "<Q"
    else:
        raise ValueError(f"unknown optional header magic {magic:#x}")
    (import_rva,) = struct.unpack_from("<I", image, directories + 8)
    if import_rva == 0:
        return set()
    table = optional + optional_size
    spans = [
        struct.unpack_from("<8xIIII", image, table + 40 * index)
        for index in range(sections)
    ]

    def offset(rva: int) -> int:
        for size, base, _, raw in spans:
            if base <= rva < base + size:
                return raw + rva - base
        raise ValueError(f"RVA {rva:#x} is in no section")

    width = struct.calcsize(thunk)
    ordinal = 1 << (8 * width - 1)
    found: set[tuple[str, str]] = set()
    descriptor = offset(import_rva)
    while True:
        lookup, _, _, name, address = struct.unpack_from(
            "<IIIII", image, descriptor
        )
        if name == 0:
            return found
        dll = _cstring(image, offset(name)).lower()
        entry = offset(lookup or address)
        while True:
            (value,) = struct.unpack_from(thunk, image, entry)
            if value == 0:
                break
            if not value & ordinal:
                found.add((dll, _cstring(image, offset(value) + 2)))
            entry += width
        descriptor += 20


def run_time_mutex(wheel: Path) -> list[str]:
    """Return the extension modules of the wheel that lack the import."""
    missing = []
    with zipfile.ZipFile(wheel) as archive:
        modules = [n for n in archive.namelist() if n.endswith(".pyd")]
        if not modules:
            raise ValueError(f"{wheel.name} holds no extension module")
        for name in sorted(modules):
            if MUTEX_INIT not in pe_imports(archive.read(name)):
                missing.append(name)
    return missing


def loaded_modules() -> dict[str, str]:
    """Map the lowercase name of each DLL in the process to its path."""
    from ctypes import wintypes

    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.EnumProcessModules.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HMODULE),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    psapi.GetModuleFileNameExW.argtypes = [
        wintypes.HANDLE,
        wintypes.HMODULE,
        wintypes.LPWSTR,
        wintypes.DWORD,
    ]
    process = kernel32.GetCurrentProcess()
    needed = wintypes.DWORD()
    count = 1024
    handles = (wintypes.HMODULE * count)()
    if not psapi.EnumProcessModules(
        process, handles, ctypes.sizeof(handles), ctypes.byref(needed)
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    modules: dict[str, str] = {}
    for handle in handles[: needed.value // ctypes.sizeof(wintypes.HMODULE)]:
        buffer = ctypes.create_unicode_buffer(32768)
        if psapi.GetModuleFileNameExW(process, handle, buffer, len(buffer)):
            modules[Path(buffer.value).name.lower()] = buffer.value
    return modules


STALE_CASES = ("system", "conda", "preloaded")

# The exit status of a process that dies of an access violation, as the
# unsigned value of the Windows exit code. A signed status, as PowerShell
# reports it, is the same value modulo 2**32.
ACCESS_VIOLATION = 0xC0000005

# The size of the data that `lock_both` trains on. The multi-worker training
# path runs only for a node with at least 10000 cases and two attributes. See
# `CanEvaluateQueueInParallel` in `src/formtree.cpp`.
STALE_ATTRIBUTES = 3
STALE_TRAINING_CASES = 12000

# Hold the handle of a preloaded runtime, so that the module stays loaded.
_preloaded: list[ctypes.CDLL] = []


def describe_status(status: int) -> str:
    """Describe the exit status of a child process."""
    if status & 0xFFFFFFFF == ACCESS_VIOLATION:
        return (
            f"access violation ({ACCESS_VIOLATION:#x}), which a mutex built "
            "by a constexpr constructor causes when it meets an old runtime"
        )
    return f"exit status {status}"


def check_planted(planted: Path, modules: Mapping[str, str]) -> None:
    """Require that the process loaded `planted` as its `msvcp140.dll`."""
    actual = modules.get("msvcp140.dll")
    assert actual is not None, f"msvcp140.dll is not loaded: {dict(modules)}"
    loaded = Path(actual).resolve()
    expected = planted.resolve()
    assert os.path.normcase(loaded) == os.path.normcase(expected), (
        f"the process loaded {loaded} and not the planted {expected}, so "
        "this case does not test an old runtime"
    )


def stale_data() -> tuple[str, str]:
    """Return a names file and training data with enough cases to train in
    parallel."""
    attributes = [f"a{index}" for index in range(STALE_ATTRIBUTES)]
    names = "low, high.\n" + "".join(f"{a}: continuous.\n" for a in attributes)
    generator = random.Random(0)
    rows = []
    for _ in range(STALE_TRAINING_CASES):
        values = [generator.randrange(100) for _ in attributes]
        label = "high" if sum(values) >= 150 else "low"
        rows.append(", ".join([*map(str, values), label]) + "\n")
    return names, "".join(rows)


def lock_both() -> None:
    """Take the `Predictor` lock and run the multi-worker training path."""
    import c50

    names, data = stale_data()
    model = c50.train(names, data)
    cases = "10, 10, 10, ?\n90, 90, 90, ?\n"
    sys.stdout.flush()
    print("locking the Predictor mutex", flush=True)
    result = model.prepare_predictor().predict_details(cases)
    assert result.labels == ["low", "high"], result.labels
    print("locking the multi-worker training path", flush=True)
    parallel = c50.train(names, data, split_workers=2)
    result = parallel.predict_details(cases)
    assert result.labels == ["low", "high"], result.labels


def run_stale_child(
    case: str,
    planted: Path,
    modules: Callable[[], Mapping[str, str]] = loaded_modules,
) -> None:
    """Run one case in this process. The caller isolates the process."""
    if case == "preloaded":
        _preloaded.append(ctypes.WinDLL(str(planted)))
    else:
        directory = Path(sys.executable).resolve().parent
        assert planted.resolve().parent == directory, (
            f"{planted} is not next to {sys.executable}"
        )
    import c50

    print(f"c50 from {c50.__file__}", flush=True)
    check_planted(planted, modules())
    print(f"planted runtime in use: {planted}", flush=True)
    lock_both()


def run_stale_case(
    case: str,
    planted: Path,
    run: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> bool:
    """Run one case in a child process and report whether it passed."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "stale-child",
        case,
        str(planted),
    ]
    print(f"stale runtime case {case}", flush=True)
    status = run(command, check=False).returncode
    if status == 0:
        print(f"case {case}: ok", flush=True)
        return True
    print(f"case {case}: FAILED with {describe_status(status)}", flush=True)
    return False


def check_wheel(wheel: Path) -> None:
    """Require a wheel that bundles no Microsoft runtime DLL."""
    bundled = bundled_runtime(wheel)
    assert not bundled, f"the wheel bundles runtime DLLs: {bundled}"
    print(f"{wheel.name} bundles no Microsoft runtime DLL")
    missing = run_time_mutex(wheel)
    assert not missing, (
        f"these extensions do not import {MUTEX_INIT[1]} from "
        f"{MUTEX_INIT[0]}, so they need a runtime as new as their "
        f"toolset: {missing}"
    )
    print(f"every extension imports {MUTEX_INIT[1]}")


def check_absent() -> None:
    """Require that `import c50` fails where the runtime is missing."""
    try:
        import c50
    except ImportError as error:
        print(f"import c50 failed without the runtime: {error}")
        return
    raise AssertionError(f"import c50 succeeded without the runtime: {c50}")


def check_system_runtime() -> None:
    """Require that the process uses the system `msvcp140.dll`."""
    import c50

    package = Path(c50.__file__).resolve().parent
    assert not (package.parent / "c50.libs").exists(), "the wheel has c50.libs"
    modules = loaded_modules()
    assert "msvcp140.dll" in modules, f"msvcp140.dll is not loaded: {modules}"
    system = Path(os.environ["SYSTEMROOT"], "System32", "msvcp140.dll")
    actual = Path(modules["msvcp140.dll"]).resolve()
    assert os.path.normcase(actual) == os.path.normcase(system.resolve()), (
        f"msvcp140.dll is {actual}, not {system}"
    )
    print(f"system runtime in use: {actual}")


def check_model(kind: c50.ModelKind) -> None:
    """Train a small model of the given kind and check its predictions."""
    import c50

    model = c50.Model.train(
        "low, high.\nx: continuous.\n",
        "0, low\n1, low\n3, high\n4, high\n",
        kind,
    )
    result = model.predict_details("0, ?\n4, ?\n")
    assert result.labels == ["low", "high"], result.labels
    print(f"{kind.name}: predicted {result.labels}")


def main(argv: list[str]) -> int:
    """Run the mode that the arguments name."""
    print(f"python {sys.version}")
    if len(argv) == 2 and argv[0] == "wheel":
        check_wheel(Path(argv[1]))
    elif len(argv) == 3 and argv[0] in ("stale", "stale-child"):
        if argv[1] not in STALE_CASES:
            print(f"unknown case {argv[1]!r}: use one of {STALE_CASES}")
            return 2
        planted = Path(argv[2])
        if argv[0] == "stale-child":
            run_stale_child(argv[1], planted)
        elif not run_stale_case(argv[1], planted):
            return 1
    elif argv == ["absent"]:
        check_absent()
    elif argv == ["present"]:
        import c50

        print(f"c50 from {c50.__file__}")
        check_system_runtime()
        check_model(c50.ModelKind.TREE)
        check_model(c50.ModelKind.RULES)
    else:
        print(
            "usage: windows_clean_import.py wheel WHEEL | absent | present"
            " | stale CASE DLL"
        )
        return 2
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
