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
        print object ":" symbol
    }' |
    LC_ALL=C sort > "$actual"

diff -u "$expected" "$actual"
