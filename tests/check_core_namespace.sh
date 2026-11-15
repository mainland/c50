#!/bin/sh
# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Fail when the core archive defines an external symbol outside the `c50`
# namespace. A consumer that links the archive statically owns the global
# namespace, so a global core function or object with a common name, such as
# `ExecTime`, would conflict with the consumer's own definition. Symbols of the
# standard library (`std`, `__gnu_cxx`) are instantiated from templates and
# are shared by design, and the global allocation operators are the language's
# own. The check lists mangled names, so that it needs no demangler.
#
# The check fails when `nm` fails and when `nm` lists no symbol, so that it
# cannot pass without looking at the archive.
#
# usage: check_core_namespace.sh <nm> <archive>

set -eu

if test "$#" -ne 2; then
    printf 'usage: %s <nm> <archive>\n' "$0" >&2
    exit 2
fi

nm_command=$1
archive=$2
work=$(mktemp -d "${TMPDIR:-/tmp}/c50-core-namespace.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM

# `nm` writes to a file first so that its exit status stops the script.
LC_ALL=C "$nm_command" --defined-only --extern-only "$archive" \
    > "$work/listing"

# Keep the symbol column of each defined-symbol line. Lines that name an
# archive member or are blank have fewer than three fields.
awk 'NF >= 3 { print $3 }' "$work/listing" > "$work/symbols"
if ! test -s "$work/symbols"; then
    printf 'nm listed no symbol in %s\n' "$archive" >&2
    exit 1
fi

# The explicit allowlist is the global allocation operators and the symbols
# that the compiler and AddressSanitizer emit: the `DW.ref.` references for
# exception handling, Clang's `__clang_call_terminate`, and the sanitizer's
# registration flag. AddressSanitizer also prefixes a marker to the name of
# each global it instruments.
awk '
    {
        symbol = $0
        sub(/^__odr_asan_gen_/, "", symbol)
        if ( symbol ~ /^_Z(Z|GV|GR|T[VSI])?N[KRV]*(St|3c50|9__gnu_cxx)/ ) next
        if ( symbol ~ /^_Z(Z|GV|T[VSI])?St/ ) next
        if ( symbol ~ /^_Z(nw|na|dl|da)/ ) next
        if ( symbol ~ /^DW\.ref\./ ) next
        if ( symbol == "__clang_call_terminate" ) next
        if ( symbol == "___asan_globals_registered" ) next
        print symbol
    }' "$work/symbols" | LC_ALL=C sort -u > "$work/global"

if test -s "$work/global"; then
    printf 'external symbols outside the c50 namespace in %s:\n' \
        "$archive" >&2
    cat "$work/global" >&2
    exit 1
fi
