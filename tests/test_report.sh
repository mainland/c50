#!/bin/sh
# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

set -eu

script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)
repo_dir=$(CDPATH= cd "$script_dir/.." && pwd)
expected="$script_dir/expected/report/basic.output"
binary=${REPORT_BINARY:-"$repo_dir/report"}
c50_binary=${C50_BINARY:-"$repo_dir/c5.0"}

if test ! -x "$binary"; then
    printf 'C5.0 report executable not found: %s\n' "$binary" >&2
    exit 1
fi

if test ! -x "$c50_binary"; then
    printf 'C5.0 executable not found: %s\n' "$c50_binary" >&2
    exit 1
fi

test_dir=$(mktemp -d "${TMPDIR:-/tmp}/c50-report-test.XXXXXX")
trap 'rm -rf "$test_dir"' EXIT HUP INT TERM

# Windows programs write CRLF line endings to text streams. Compare the text
# without carriage returns so that the fixtures apply on every platform.
diff_text()
{
    tr -d '\r' < "$1" > "$test_dir/diff-text.expected"
    tr -d '\r' < "$2" > "$test_dir/diff-text.actual"
    if ! diff -u "$test_dir/diff-text.expected" "$test_dir/diff-text.actual"
    then
        printf 'output differs: %s %s\n' "$1" "$2" >&2
        return 1
    fi
}

"$binary" 20 2 1 0 > "$test_dir/output.raw" <<'EOF'
3 1 (10.0%)
5 2 (20.0%)
EOF

sed -e 's/[[:space:]]*$//' "$test_dir/output.raw" \
    > "$test_dir/output.actual"
diff_text "$expected" "$test_dir/output.actual"

"$binary" 20 2 1 0 > "$test_dir/marked.raw" <<'EOF'
3 1 (10.0%)   <<
5 2 (20.0%)   <<
EOF

sed -e 's/[[:space:]]*$//' "$test_dir/marked.raw" \
    > "$test_dir/marked.actual"
diff_text "$expected" "$test_dir/marked.actual"

run_cross_validation()
{
    fixture_name=$1
    case_name=$2
    rules=$3
    shift 3

    case_dir="$test_dir/cross-validation-$case_name"
    mkdir "$case_dir"
    cp "$script_dir/fixtures/$fixture_name/$fixture_name.names" "$case_dir/"
    cp "$script_dir/fixtures/$fixture_name/$fixture_name.data" "$case_dir/"
    if test "$case_name" = costs; then
        cp "$script_dir/fixtures/$fixture_name/$fixture_name.costs" "$case_dir/"
    fi

    cases=$(awk 'END { print NR }' "$case_dir/$fixture_name.data")
    "$c50_binary" -f "$case_dir/$fixture_name" -X 2 -I 17 "$@" \
        > "$case_dir/cli.output"
    sed -n '/<</p' "$case_dir/cli.output" > "$case_dir/folds.marked"
    sed 's/[[:space:]]*<<[[:space:]]*$//' "$case_dir/folds.marked" \
        > "$case_dir/folds.plain"
    "$binary" "$cases" 2 1 "$rules" < "$case_dir/folds.marked" \
        > "$case_dir/marked.output"
    "$binary" "$cases" 2 1 "$rules" < "$case_dir/folds.plain" \
        > "$case_dir/plain.output"
    diff_text "$case_dir/plain.output" "$case_dir/marked.output"
}

run_cross_validation basic tree 0
run_cross_validation basic rules 1 -r
run_cross_validation boost boost 0 -t 5
run_cross_validation basic costs 0

# Release 2.07 sums the costs and their squares in float. Two folds of 30% and
# 10% errors with costs of 1.55 and 1.30 give a standard error of 0.12, where
# the same sums in double give 0.13.
run_float_summary()
{
    printf '2 3 (30.0%%) 1.55\n2 1 (10.0%%) 1.30\n' \
        > "$test_dir/float-summary.input"
    "$binary" 20 2 1 0 < "$test_dir/float-summary.input" \
        > "$test_dir/float-summary.raw"
    sed -e 's/[[:space:]]*$//' "$test_dir/float-summary.raw" \
        > "$test_dir/float-summary.actual"

    for float_expected in \
        '	Mean       2.0     20.0%   1.42' \
        '	SE         0.0     10.0%   0.12'
    do
        if ! grep -F -x -e "$float_expected" \
            "$test_dir/float-summary.actual" > /dev/null; then
            printf 'float summary did not print: %s\n' \
                "$float_expected" >&2
            cat "$test_dir/float-summary.actual" >&2
            return 1
        fi
    done
}

# Costs near the limit of float. A cost of 1e20 makes the sum of squares
# overflow float, so Release 2.07 prints a non-finite standard error. Four
# folds of 3e38 make the sum of costs overflow, so it prints a mean of inf as
# well. The text of a non-finite value depends on the C library, as inf,
# infinity, nan, -nan, or -nan(ind), and so does its padding, because the
# column has a fixed width that a long spelling overflows.
run_large_costs()
{
    large_name=$1
    large_folds=$2
    large_line=$3
    large_mean=$4

    large_input="$test_dir/large-$large_name.input"
    : > "$large_input"
    large_fold=0
    while test "$large_fold" -lt "$large_folds"; do
        printf '%s\n' "$large_line" >> "$large_input"
        large_fold=$((large_fold + 1))
    done
    "$binary" 4 "$large_folds" 1 0 < "$large_input" \
        > "$test_dir/large-$large_name.raw"
    sed -e 's/[[:space:]]*$//' "$test_dir/large-$large_name.raw" \
        > "$test_dir/large-$large_name.actual"

    if ! grep -E -i -x "$large_mean" \
        "$test_dir/large-$large_name.actual" > /dev/null ||
        ! grep -E -i -x \
            '	SE         0\.0      0\.0%[[:space:]]*-?(inf(inity)?|nan)(\(ind\))?' \
            "$test_dir/large-$large_name.actual" > /dev/null; then
        printf 'large costs (%s) printed unexpected text:\n' \
            "$large_name" >&2
        cat "$test_dir/large-$large_name.actual" >&2
        return 1
    fi
}

run_float_summary
run_large_costs squares 2 '2 1 (50.0%) 100000000000000000000' \
    '	Mean       2\.0     50\.0%  100000002004087734272\.00'
run_large_costs sum 4 \
    '2 1 (50.0%) 300000000000000000000000000000000000000' \
    '	Mean       2\.0     100\.0%[[:space:]]*-?(inf(inity)?|nan)(\(ind\))?'

run_failure()
{
    case_name=$1
    expected_name=$2
    shift 2

    set +e
    "$binary" "$@" </dev/null > "$test_dir/$case_name.actual" 2>&1
    failure_exit_code=$?
    set -e

    if test "$failure_exit_code" -ne 1; then
        printf '%s returned %d, expected 1\n' \
            "$case_name" "$failure_exit_code" >&2
        return 1
    fi

    diff_text "$script_dir/expected/report/$expected_name.output" \
        "$test_dir/$case_name.actual"
}

run_failure no-arguments usage
run_failure invalid-arguments usage invalid 2 1 0
run_failure missing-input missing-input 20 2 1 0

{
    printf '%200s%s\n' '' '3 1 (10.0%)'
    printf '%s\n' '5 2 (20.0%)'
} | "$binary" 20 2 1 0 > "$test_dir/long-line.raw"
normalize_output()
{
    sed -e 's/[[:space:]]*$//' "$1"
}
normalize_output "$test_dir/long-line.raw" > "$test_dir/long-line.actual"
diff_text "$expected" "$test_dir/long-line.actual"

set +e
printf '%s\n' 'composite malformed' '5 2 (20.0%)' | \
    "$binary" 20 2 1 0 > "$test_dir/malformed.actual" 2>&1
malformed_exit_code=$?
set -e
if test "$malformed_exit_code" -ne 1; then
    printf 'malformed input returned %d, expected 1\n' \
        "$malformed_exit_code" >&2
    exit 1
fi

for suffix in '<' 'extra' '<< extra'; do
    set +e
    printf '3 1 (10.0%%) %s\n5 2 (20.0%%)\n' "$suffix" | \
        "$binary" 20 2 1 0 > "$test_dir/invalid-suffix.actual" 2>&1
    suffix_exit_code=$?
    set -e
    if test "$suffix_exit_code" -ne 1; then
        printf 'invalid suffix returned %d, expected 1\n' \
            "$suffix_exit_code" >&2
        exit 1
    fi
done

printf 'Report regression tests passed\n'
