#!/bin/sh
# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Build the imported C5.0 Release 2.07 program the way its Makefile does,
# without csh, strip, or the removal of the amalgamation.
#
# usage: build_reference.sh [SOURCE_DIR] OUTPUT
#
# SOURCE_DIR holds the files of the Release 2.07 import. Without it, the script
# extracts a ref from the enclosing repository with `git archive`. The ref is
# C50_REFERENCE_REF if set, else the import commit
# f97414a56b432baf4547193a47e6d695e5f5229b. A shallow clone
# that lacks the commit needs SOURCE_DIR or the variable.
#
# C50_REFERENCE_CFLAGS adds compiler flags. `-DVerbOpt` builds the
# verbose-diagnostics program, `c5.0dbg`.

set -eu

case $# in
1)
    output=$1
    source_dir=
    ;;
2)
    source_dir=$1
    output=$2
    ;;
*)
    printf 'usage: %s [SOURCE_DIR] OUTPUT\n' "$0" >&2
    exit 2
    ;;
esac

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/c50-reference.XXXXXX")
trap 'rm -rf "$work_dir"' EXIT HUP INT TERM

if test -z "$source_dir"; then
    script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)
    mkdir "$work_dir/main"
    top_dir=$(git -C "$script_dir" rev-parse --show-toplevel)
    ref=${C50_REFERENCE_REF:-f97414a56b432baf4547193a47e6d695e5f5229b}
    if ! git -C "$top_dir" rev-parse --verify --quiet \
        "$ref^{commit}" > /dev/null; then
        printf 'no Release 2.07 ref: %s\n' "$ref" >&2
        printf 'Set C50_REFERENCE_REF or pass SOURCE_DIR.\n' >&2
        exit 1
    fi
    git -C "$top_dir" archive "$ref" | tar -x -C "$work_dir/main"
    source_dir=$work_dir/main
fi

if test ! -f "$source_dir/defns.i" || test ! -f "$source_dir/Makefile"; then
    printf 'not a Release 2.07 source directory: %s\n' "$source_dir" >&2
    exit 1
fi

# The source list is the Makefile's `src` variable, one file per line.
sources=$(awk '
    /^src[ \t]*=/ { listing = 1; next }
    listing && NF == 0 { exit }
    listing { gsub(/[\\ \t]/, ""); print }
' "$source_dir/Makefile")

if test -z "$sources"; then
    printf 'no source list in %s/Makefile\n' "$source_dir" >&2
    exit 1
fi

output_dir=$(CDPATH= cd "$(dirname "$output")" && pwd)
output=$output_dir/$(basename "$output")

# Release 2.07 calls finite(), which the macOS headers no longer declare, and
# Apple Clang rejects calls to undeclared functions. isfinite() gives the same
# result for every double, so the substitution changes no arithmetic. glibc
# still declares finite(), so other systems compile the source unchanged.
platform_flags=
case $(uname -s) in
Darwin) platform_flags=-Dfinite=isfinite ;;
esac

(
    cd "$source_dir"
    # shellcheck disable=SC2086 # The list holds plain file names.
    cat defns.i $sources | grep -E -v 'defns.i|extern.i' > "$work_dir/c50gt.c"
    # -ffp-contract=off keeps the reference to x86-64 arithmetic on targets
    # with fused multiply-add. It changes nothing on x86-64.
    gcc -ffloat-store -w -O3 -ffp-contract=off $platform_flags \
        ${C50_REFERENCE_CFLAGS:-} -I. -o "$output" "$work_dir/c50gt.c" -lm
)
