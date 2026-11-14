# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check the license notices of tracked source files.

Usage: python tests/check_license_notices.py

Run it from a Git checkout. The checks are:

- Every tracked file of a kind that can carry a comment contains an
  SPDX-License-Identifier line, except the fixtures, expected outputs, and the
  unmodified imported files named below.
- Every file that descends from a file of C5.0 Release 2.07 (the `main`
  branch) keeps RuleQuest's copyright line and carries a `Modified` line, as
  section 5(a) of the GNU General Public License requires. The root
  `Makefile` is among them.
- Each unmodified imported file is byte-identical to its blob on `main`. The
  script holds those blob identifiers, so the check needs no `main` branch.
- `TrainModel` in `src/c50_model.cpp`, `src/c50_sort.h`, and
  `tests/sort_properties.cpp` follow code of Release 2.07 and credit
  RuleQuest.

When a local `main` or `origin/main` exists, the script also checks that every
source file of `main` has a counterpart in the tables below and that the
counterpart is a tracked file, so an import that is renamed and stripped of its
notices fails. A shallow clone without either ref skips that comparison and
says so on standard error.

The script exits with status 77 when the directory is not a Git checkout.
"""

from __future__ import annotations

import re
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
MARKER = b"SPDX-License-Identifier:"

# Files of Release 2.07 that this project renamed or moved, keyed by the
# current path. Each must keep RuleQuest's notice and say that it was modified.
MODIFIED_IMPORTS = frozenset({"Makefile"}).union(
    "src/" + name
    for name in (
        "attwinnow.cpp", "c50.cpp", "classify.cpp", "confmat.cpp",
        "construct.cpp", "contin.cpp", "defns.i", "discr.cpp", "extern.i",
        "formrules.cpp", "formtree.cpp", "getdata.cpp", "getnames.cpp",
        "global.cpp", "implicitatt.cpp", "info.cpp", "mcost.cpp",
        "modelfiles.cpp", "p-thresh.cpp", "prune.cpp", "report.cpp",
        "rules.cpp", "ruletree.cpp", "siftrules.cpp", "sort.cpp",
        "subset.cpp", "trees.cpp", "update.cpp", "utility.cpp", "xval.cpp",
    )
)  # fmt: skip
# Imported files that this project did not modify, with their Git blob
# identifiers on `main`. They keep their original notice and need no SPDX line.
UNMODIFIED_IMPORTS = {
    "src/text.i": "23e272f778c8a12ac3998bc4fb8cd21c407a1673",
    "xval": "b062799d9b6e63ead77b6a91d38f42074ce0ba49",
}
# Files that are not descendants of a file of `main` by name, but contain code
# that follows one and credit RuleQuest for it.
CREDITED_FILES = frozenset(
    {"src/c50_model.cpp", "src/c50_sort.h", "tests/sort_properties.cpp"}
)

RULEQUEST = re.compile(rb"copyright 2010 rulequest research pty ltd", re.I)
MODIFIED = re.compile(rb"^\s*(?:/\*|#|//)?\s*Modified\b", re.M)
# Files of `main` that have no counterpart to check.
UNNOTICED_MAIN_FILES = {"gpl.txt"}
# Files of `main` that keep their name at the root.
ROOT_MAIN_FILES = {"Makefile", "xval"}


def tracked_files() -> list[str] | None:
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"], capture_output=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return [name for name in result.stdout.decode().split("\0") if name]


def blob_id(name: str) -> str | None:
    result = subprocess.run(
        ["git", "hash-object", "--", name], capture_output=True, check=False
    )
    if result.returncode != 0:
        return None
    return result.stdout.decode().strip()


def needs_notice(name: str) -> bool:
    if name in UNMODIFIED_IMPORTS or name.startswith(EXEMPT_PREFIXES):
        return False
    path = PurePosixPath(name)
    return path.suffix in COMMENT_SUFFIXES or path.name in COMMENT_NAMES


def counterpart(main_name: str) -> str | None:
    """Return the current path of a source file of `main`."""
    if main_name in UNNOTICED_MAIN_FILES:
        return None
    if main_name in ROOT_MAIN_FILES:
        return main_name
    if main_name.endswith(".c"):
        return "src/" + main_name[:-2] + ".cpp"
    return "src/" + main_name


def main_files() -> tuple[str, set[str]] | None:
    for ref in ("main", "origin/main"):
        result = subprocess.run(
            ["git", "ls-tree", "--name-only", f"{ref}^{{commit}}"],
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            return ref, set(result.stdout.decode().split())
    return None


def check_table(tracked: set[str]) -> list[str]:
    found = main_files()
    if found is None:
        print(
            "note: no main or origin/main branch; skipped the comparison of "
            "the imported file tables with Release 2.07",
            file=sys.stderr,
        )
        return []
    ref, names = found
    known = MODIFIED_IMPORTS | set(UNMODIFIED_IMPORTS)
    problems = []
    for name in sorted(names):
        target = counterpart(name)
        if target is None:
            continue
        if target not in known:
            problems.append(f"{name} of {ref} has no entry for {target}")
        elif target not in tracked:
            problems.append(f"{name} of {ref} has no tracked file {target}")
    return problems


def check_file(name: str) -> list[str]:
    path = Path(name)
    if not path.is_file():
        return []
    text = path.read_bytes()
    problems = []
    if needs_notice(name) and MARKER not in text:
        problems.append(f"missing SPDX-License-Identifier: {name}")
    if name in MODIFIED_IMPORTS:
        if not RULEQUEST.search(text):
            problems.append(f"missing RuleQuest copyright line: {name}")
        if not MODIFIED.search(text):
            problems.append(f"missing Modified line: {name}")
    if name in CREDITED_FILES and not RULEQUEST.search(text):
        problems.append(f"missing RuleQuest credit: {name}")
    if (
        name in UNMODIFIED_IMPORTS
        and blob_id(name) != UNMODIFIED_IMPORTS[name]
    ):
        problems.append(f"differs from its Release 2.07 blob: {name}")
    return problems


def main() -> int:
    names = tracked_files()
    if names is None:
        print("not a Git checkout", file=sys.stderr)
        return 77
    problems = check_table(set(names))
    for name in names:
        problems.extend(check_file(name))
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
