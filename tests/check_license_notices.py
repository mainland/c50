# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check that tracked source files carry an SPDX license identifier.

Usage: python tests/check_license_notices.py

Run it from a Git checkout. Every tracked file of a kind that can carry a
comment must contain an SPDX-License-Identifier line, except the fixtures,
expected outputs, and unmodified imported RuleQuest files named below. The
script exits with status 77 when the directory is not a Git checkout.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path, PurePosixPath

COMMENT_SUFFIXES = {
    ".c", ".cmake", ".cpp", ".cff", ".dict", ".h", ".hpp", ".i", ".in",
    ".py", ".pyi", ".rst", ".sh", ".toml", ".yaml", ".yml",
}  # fmt: skip
COMMENT_NAMES = {
    "CMakeLists.txt", "Makefile", ".gitattributes", ".gitignore",
    "stubtest-allowlist.txt",
}  # fmt: skip
# Fixtures and expected outputs are data and carry no comment.
EXEMPT_PREFIXES = ("tests/expected/", "tests/fixtures/", "tests/fuzz/corpus/")
# Imported from the RuleQuest archive and not modified by this project.
EXEMPT_FILES = {"xval", "src/text.i"}
MARKER = b"SPDX-License-Identifier:"


def tracked_files() -> list[str] | None:
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"], capture_output=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return [name for name in result.stdout.decode().split("\0") if name]


def needs_notice(name: str) -> bool:
    if name in EXEMPT_FILES or name.startswith(EXEMPT_PREFIXES):
        return False
    path = PurePosixPath(name)
    return path.suffix in COMMENT_SUFFIXES or path.name in COMMENT_NAMES


def main() -> int:
    names = tracked_files()
    if names is None:
        print("not a Git checkout", file=sys.stderr)
        return 77
    missing = [
        name
        for name in names
        if needs_notice(name) and MARKER not in Path(name).read_bytes()
    ]
    for name in missing:
        print(f"missing SPDX-License-Identifier: {name}", file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
