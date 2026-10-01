"""Check that a source distribution holds exactly the tracked source files.

Usage: python tests/check_sdist.py <sdist.tar.gz>

Run it from a Git checkout of the revision that built the sdist. Apart from
the generated PKG-INFO, every archive member must be a tracked file, and every
tracked file must be in the archive unless pyproject.toml excludes it.
"""

from __future__ import annotations

import subprocess
import sys
import tarfile

# Tracked files that sdist.exclude in pyproject.toml deliberately omits.
EXCLUDED: set[str] = set()


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    with tarfile.open(sys.argv[1]) as archive:
        members = {
            member.name.split("/", 1)[1]
            for member in archive.getmembers()
            if member.isfile() and "/" in member.name
        }
    tracked = set(
        subprocess.run(
            ["git", "ls-files", "-z"], check=True, capture_output=True, text=True
        ).stdout.split("\0")
    ) - {""}
    unexpected = sorted(members - tracked - {"PKG-INFO"})
    missing = sorted(tracked - members - EXCLUDED)
    for name in unexpected:
        print(f"untracked file in sdist: {name}", file=sys.stderr)
    for name in missing:
        print(f"tracked file missing from sdist: {name}", file=sys.stderr)
    if unexpected or missing:
        return 1
    print(f"sdist holds the {len(members) - 1} tracked files and PKG-INFO")
    return 0


if __name__ == "__main__":
    sys.exit(main())
