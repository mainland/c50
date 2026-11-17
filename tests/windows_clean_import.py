# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Import a Windows wheel and train a model, with a runtime-DLL audit.

`tests/windows_clean_import.ps1` runs this script with an embeddable CPython in
a container that has no Visual C++ Redistributable. The script needs only the
standard library and the wheel. It checks three things:

- `c50` imports from the directory that holds the unpacked wheel.
- The wheel's bundled `msvcp140-<hash>.dll` is the copy of the C++ runtime
  that the process uses, and no system copy of `msvcp140.dll` is loaded.
- A text-input model of each kind trains and predicts.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import c50


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


def check_runtime() -> None:
    """Require the bundled C++ runtime and reject a system copy."""
    package = Path(c50.__file__).resolve().parent
    libs = package.parent / "c50.libs"
    bundled = sorted(libs.glob("msvcp140-*.dll"))
    assert len(bundled) == 1, f"expected one bundled msvcp140 in {libs}"
    modules = loaded_modules()
    name = bundled[0].name.lower()
    assert name in modules, f"{name} is not loaded: {sorted(modules)}"
    assert Path(modules[name]).resolve() == bundled[0].resolve()
    system = [path for key, path in modules.items() if key == "msvcp140.dll"]
    assert not system, f"a system C++ runtime is loaded: {system}"
    print(f"bundled runtime in use: {modules[name]}")


def check_model(kind: c50.ModelKind) -> None:
    """Train a small model of the given kind and check its predictions."""
    model = c50.Model.train(
        "low, high.\nx: continuous.\n",
        "0, low\n1, low\n3, high\n4, high\n",
        kind,
    )
    result = model.predict_details("0, ?\n4, ?\n")
    assert result.labels == ["low", "high"], result.labels
    print(f"{kind.name}: predicted {result.labels}")


def main() -> None:
    """Run the checks and print the interpreter and wheel locations."""
    print(f"python {sys.version}")
    print(f"c50 from {c50.__file__}")
    check_runtime()
    check_model(c50.ModelKind.TREE)
    check_model(c50.ModelKind.RULES)
    print("ok")


if __name__ == "__main__":
    main()
