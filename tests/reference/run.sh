#!/bin/sh
# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Compare a c5.0 program with C5.0 Release 2.07 on generated data.
#
# usage: run.sh CANDIDATE [REFERENCE_SOURCE_DIR]
#
# CANDIDATE is the library's c5.0 program. REFERENCE_SOURCE_DIR holds the
# files of the Release 2.07 import. Without it, the script extracts a ref from
# the repository, as `build_reference.sh` describes. Set C50_COMPARE_JOBS to
# change the number of parallel cases.

set -eu

if test $# -lt 1 || test $# -gt 2; then
    printf 'usage: %s CANDIDATE [REFERENCE_SOURCE_DIR]\n' "$0" >&2
    exit 2
fi

script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)
candidate=$1

if test ! -x "$candidate"; then
    printf 'C5.0 executable not found: %s\n' "$candidate" >&2
    exit 1
fi

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/c50-release-207.XXXXXX")
trap 'rm -rf "$work_dir"' EXIT HUP INT TERM

if test $# -eq 2; then
    "$script_dir/build_reference.sh" "$2" "$work_dir/refc5.0"
else
    "$script_dir/build_reference.sh" "$work_dir/refc5.0"
fi

python=${PYTHON:-python3}
"$python" "$script_dir/compare.py" \
    --reference "$work_dir/refc5.0" \
    --candidate "$candidate" \
    --jobs "${C50_COMPARE_JOBS:-2}"
