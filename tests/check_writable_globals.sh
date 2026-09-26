#!/bin/sh

set -eu

if test "$#" -ne 3; then
    printf 'usage: %s <nm> <archive> <expected>\n' "$0" >&2
    exit 2
fi

nm_command=$1
archive=$2
expected=$3
actual=$(mktemp "${TMPDIR:-/tmp}/c50-writable-globals.XXXXXX")
trap 'rm -f "$actual"' EXIT HUP INT TERM

LC_ALL=C "$nm_command" -A -P --defined-only "$archive" |
    awk '$3 ~ /^[bBcCdDgGsS]$/ {
        object = $1
        symbol = $2
        sub(/^.*\[/, "", object)
        sub(/\]:$/, "", object)
        if ( $3 ~ /^[bcdgs]$/ ) {
            sub(/\.[0-9]+$/, "", symbol)
            sub(/^.*\./, "", symbol)
        }
        # C++ type information and vtables are immutable ABI metadata placed
        # in relocation-read-only data sections by ELF compilers.
        if ( symbol ~ /^_ZT(I|V)/ ) next
        # AddressSanitizer uses this process-local flag to register its own
        # immutable global metadata. It is not application state.
        if ( symbol == "___asan_globals_registered" ) next
        # Clang AddressSanitizer promotes these function-local, const pointer
        # tables to relocation-read-only data. Their source types prevent both
        # the pointers and strings from being mutated.
        if ( object == "construct.cpp.o" &&
             symbol ~ /^(Extra|ExtraC|Multi|StdP|StdPC|StdR)$/ ) next
        if ( object == "xval.cpp.o" &&
             symbol ~ /^(Extra|ExtraC|FoldHead|StdP|StdPC)$/ ) next
        print object ":" symbol
    }' |
    LC_ALL=C sort > "$actual"

diff -u "$expected" "$actual"
