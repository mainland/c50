#!/bin/sh
# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Run `c5.0` on more than 21,474,836 cases and check the percentages of its
# evaluation report. Each percentage is 100 times a case count over the number
# of cases, and a product in `int` passes INT_MAX above that count. Two runs
# cover the two kinds of line: 21,500,000 training cases that all use the
# attribute, and 22,000,000 test cases that the tree all misclassifies. Each
# run needs about 0.6 GiB and a minute or two.
#
# usage: report_percent.sh C5.0 [TRAINING_CASES [TEST_CASES]]

set -eu

if test $# -lt 1 || test $# -gt 3; then
    printf 'usage: %s C5.0 [TRAINING_CASES [TEST_CASES]]\n' "$0" >&2
    exit 2
fi
c50=$1
training_cases=${2:-21500000}
test_cases=${3:-22000000}

work=$(mktemp -d "${TMPDIR:-/tmp}/c50-report-percent.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM

status=0

# Half of the cases are class a and half are class b, and the attribute
# separates them, so the tree tests it for every case and makes no error.
printf 'a, b.\n\nx: p, q.\n' > "$work/usage.names"
half=$((training_cases / 2))
{
    yes 'p,a' | head -n "$half"
    yes 'q,b' | head -n "$((training_cases - half))"
} > "$work/usage.data"
"$c50" -f "$work/usage" > "$work/usage.out"
if grep -q '^	    100%  x$' "$work/usage.out"; then
    printf 'PASS attribute usage of %s cases\n' "$training_cases"
else
    printf 'FAIL attribute usage of %s cases\n' "$training_cases"
    sed -n '/Attribute usage/,$p' "$work/usage.out"
    status=1
fi
rm -f "$work/usage.data"

# Four training cases make a one-leaf tree of class a, and every test case is
# class b.
printf 'a, b.\n\nx: p, q.\n' > "$work/errors.names"
printf 'p,a\np,a\np,a\np,a\n' > "$work/errors.data"
yes 'p,b' | head -n "$test_cases" > "$work/errors.test"
"$c50" -f "$work/errors" > "$work/errors.out"
if grep -q "^[[:space:]]*1[[:space:]]*${test_cases}(100.0%)[[:space:]]*<<\$" "$work/errors.out"; then
    printf 'PASS error rate over %s test cases\n' "$test_cases"
else
    printf 'FAIL error rate over %s test cases\n' "$test_cases"
    grep '<<' "$work/errors.out" || true
    status=1
fi

exit $status
