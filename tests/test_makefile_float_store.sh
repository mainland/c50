#!/bin/sh
# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Check that the Makefile passes -ffloat-store to the compiler of a 32-bit
# build and omits it from a 64-bit build. The Makefile probes the pointer size
# with the selected compiler and its flags, so a host that builds 64-bit by
# default must still add the flag for `CXX='g++ -m32'`. The check reads the
# commands that `make -n` prints and compiles nothing. It skips with status 77
# when `make` or a 32-bit compiler mode is unavailable.

set -eu

script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)
repo_dir=$(CDPATH= cd "$script_dir/.." && pwd)
cd "$repo_dir"

skip() {
    printf 'skipping the Makefile floating-point flag check: %s\n' "$1" >&2
    exit 77
}

command -v make > /dev/null 2>&1 || skip 'make is unavailable'

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/c50-makefile-float.XXXXXX")
trap 'rm -rf "$work_dir"' EXIT HUP INT TERM

pointer_size() {
    # shellcheck disable=SC2086
    printf '__SIZEOF_POINTER__\n' | $1 -E -P -x c++ - 2> /dev/null
}

# Print the compile commands of the production program and the report program.
commands() {
    make -n -B "$@" c5.0 report 2> /dev/null | grep -E ' -o (c5\.0|report) '
}

# Check every printed command for -ffloat-store when `expected` is yes, and for
# its absence when `expected` is no.
check() {
    label=$1
    expected=$2
    shift 2
    commands "$@" > "$work_dir/commands"
    if test ! -s "$work_dir/commands"; then
        printf '%s: make printed no compile command\n' "$label" >&2
        exit 1
    fi
    while IFS= read -r line; do
        case $line in
        *-ffloat-store*) present=yes ;;
        *) present=no ;;
        esac
        if test "$present" != "$expected"; then
            printf '%s: -ffloat-store present: %s, expected: %s\n  %s\n' \
                "$label" "$present" "$expected" "$line" >&2
            exit 1
        fi
    done < "$work_dir/commands"
}

if test "$(pointer_size g++)" = 8; then
    check 'default 64-bit compiler' no
fi

if test "$(pointer_size 'g++ -m32')" != 4; then
    skip "'g++ -m32' does not target 32-bit pointers"
fi

check 'CXX with -m32' yes CXX='g++ -m32'
check 'CXXFLAGS with -m32' yes CXXFLAGS='-m32 -std=c++17'
