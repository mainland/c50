#!/bin/sh
# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Read text of more than INT_MAX lines through the command-line program. The
# first check puts a bad value on line 2^31, and the diagnostic must print
# that line number. The second counts 2^31 + 1 cases with sampling on, which
# must end with the out-of-memory error and not a sample of a wrapped count.
# The files take 2 GiB and about 10.7 GB of TMPDIR, and each run reads one.
#
# usage: line_count.sh C5.0 [LINES]
#
# LINES is the number of lines before the bad value in the first check and
# the number of cases in the second. It defaults to 2147483647 for the first
# and 2147483649 for the second. A smaller value checks this script, because
# a count that fits in int must train.

set -eu

if test $# -lt 1 || test $# -gt 2; then
    printf 'usage: %s C5.0 [LINES]\n' "$0" >&2
    exit 2
fi
c50=$1
lines=${2:-}

work=$(mktemp -d "${TMPDIR:-/tmp}/c50-line-count.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM

printf 'outcome.\n\na: continuous.\noutcome: x, y.\n' > "$work/big.names"
status=0

# The bad value is on line $before + 1.
before=${lines:-2147483647}
{
    yes '' | head -n "$before"
    printf 'oops, x\n1, x\n2, x\n3, x\n4, y\n5, y\n6, y\n'
} > "$work/big.data"
expected=$((before + 1))
if "$c50" -f "$work/big" > "$work/out" 2> "$work/err" &&
   grep -q "^\*\*\* line $expected of " "$work/out"; then
    printf 'PASS line number %s\n' "$expected"
else
    printf 'FAIL line number %s\n' "$expected"
    grep '^\*\*\* line' "$work/out" | head -3
    head -c 2000 "$work/err"
    status=1
fi
rm -f "$work/big.data"

# With sampling, the case count must fit in int.
cases=${lines:-2147483649}
yes '1, x' | head -n "$cases" > "$work/big.data"
if "$c50" -f "$work/big" -S 0.1 > "$work/out" 2> "$work/err"; then
    result=trained
else
    result=failed
fi
if test "$cases" -gt 2147483647; then
    if test "$result" = failed &&
       grep -q 'unable to allocate sufficient memory' "$work/out" "$work/err"; then
        printf 'PASS sampling %s cases\n' "$cases"
    else
        printf 'FAIL sampling %s cases (%s)\n' "$cases" "$result"
        status=1
    fi
elif test "$result" = trained; then
    printf 'PASS sampling %s cases\n' "$cases"
else
    printf 'FAIL sampling %s cases (%s)\n' "$cases" "$result"
    head -c 2000 "$work/out"
    status=1
fi

exit $status
