#!/bin/sh
# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Check that the `headers` list of the Makefile names every header that the
# programs include, so that editing one rebuilds them. The list must hold the
# public header and each `.h` and `.i` file under `src/`.

set -eu

script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)
repo_dir=$(CDPATH= cd "$script_dir/.." && pwd)
cd "$repo_dir"

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/c50-makefile-headers.XXXXXX")
trap 'rm -rf "$work_dir"' EXIT HUP INT TERM

awk '
    /^headers[ \t]*=/ { listing = 1; next }
    listing && NF == 0 { exit }
    listing { gsub(/[\\ \t]/, ""); sub(/^\$\(SRC_DIR\)/, "src"); print }
' Makefile | LC_ALL=C sort > "$work_dir/listed"

{
    printf 'include/c50/c50.hpp\n'
    ls src/*.h src/*.i
} | LC_ALL=C sort > "$work_dir/present"

if test ! -s "$work_dir/listed"; then
    printf 'no headers list in the Makefile\n' >&2
    exit 1
fi

if ! diff -u "$work_dir/listed" "$work_dir/present" > "$work_dir/diff"; then
    printf 'The Makefile headers list differs from the headers present:\n' >&2
    cat "$work_dir/diff" >&2
    exit 1
fi
