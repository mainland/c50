#!/bin/sh
# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Train inputs whose flattened class tables hold more than INT_MAX elements,
# through the command-line program and the C++ API: a schema of 46,341 classes
# (confusion matrix and boost votes, about 8.6 GB of address space each) and a
# schema of 4,096 classes and 524,288 attribute values (the discrete-frequency
# table and the split workspace hold about 16 GiB of doubles each, so the case
# needs about 32 GiB resident). The run takes under a minute.
#
# usage: index_overflow.sh C5.0 LIBRARY_PROBE

set -eu

if test $# -ne 2; then
    printf 'usage: %s C5.0 LIBRARY_PROBE\n' "$0" >&2
    exit 2
fi
c50=$1
probe=$2

work=$(mktemp -d "${TMPDIR:-/tmp}/c50-index-overflow.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM

awk 'BEGIN {
    printf "outcome.\n\nx: continuous.\noutcome: "
    for (i = 0; i < 46341; i++) printf "%sc%d", (i ? ", " : ""), i
    print "."
}' > "$work/big.names"

# Six cases: the confusion matrix has 46,342^2 cells.
printf '1,c46340\n2,c46340\n3,c0\n4,c0\n5,c46340\n6,c0\n' > "$work/big.data"
cp "$work/big.names" "$work/boost.names"
# 46,343 cases: the boost votes have 46,343 * 46,342 cells.
awk 'BEGIN {
    for (i = 0; i < 46343; i++)
        printf "%d,%s\n", i % 7, (i % 3 ? "c46340" : "c0")
}' > "$work/boost.data"

# 4,096 classes and 524,288 values: the frequency table has 2^31 cells.
awk 'BEGIN {
    printf "outcome.\n\nx: "
    for (i = 0; i < 524288; i++) printf "%sv%d", (i ? "," : ""), i
    printf ".\noutcome: "
    for (i = 0; i < 4096; i++) printf "%sc%d", (i ? ", " : ""), i
    print "."
}' > "$work/values.names"
printf 'v524287,c4095\nv524287,c4095\nv0,c0\nv0,c0\nv1,c0\nv524286,c4095\n' \
    > "$work/values.data"

status=0

check()
{
    label=$1
    shift
    if "$@" > "$work/out" 2> "$work/err"; then
        printf 'PASS %s\n' "$label"
    else
        printf 'FAIL %s (exit status %s)\n' "$label" "$?"
        head -c 2000 "$work/err"
        status=1
    fi
}

check 'c5.0 evaluation' "$c50" -f "$work/big"
grep -q 'Evaluation on training data (6 cases)' "$work/out" ||
    { printf 'FAIL c5.0 evaluation output missing\n'; status=1; }
check 'c5.0 cross-validation' "$c50" -f "$work/big" -X 3
check 'library boosting' "$probe" "$work/boost.names" "$work/boost.data" 10
check 'c5.0 discrete frequencies' "$c50" -f "$work/values"
check 'library discrete frequencies' "$probe" "$work/values.names" \
    "$work/values.data" 1

exit $status
