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

# Finite costs near the limit of float. Release 2.07 sums the squares of the
# costs in float, so a cost above about 1.8e19 makes the sum of squares
# infinite and the standard error nan, and a sum above the limit of float makes
# the mean infinite. The standard error of equal folds is 0.
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

    for large_expected in \
        "	Mean       2.0     ${large_mean%% *}%  ${large_mean#* }" \
        '	SE         0.0      0.0%   0.00'
    do
        if ! grep -F -x -e "$large_expected" \
            "$test_dir/large-$large_name.actual" > /dev/null; then
            printf 'large costs (%s) did not print: %s\n' \
                "$large_name" "$large_expected" >&2
            cat "$test_dir/large-$large_name.actual" >&2
            return 1
        fi
    done
}

run_large_costs squares 2 '2 1 (50.0%) 100000000000000000000' \
    '50.0 100000002004087734272.00'
run_large_costs sum 4 \
    '2 1 (50.0%) 300000000000000000000000000000000000000' \
    '100.0 300000000549775575777803994281145270272.00'

# Summaries of fold results. Release 2.07 sums the values and their squares in
# float. The mean and the standard error each keep that arithmetic, and so its
# text, whenever it is finite. Only a number that Release 2.07 prints as nan or
# inf is computed in double.
run_summary_case()
{
    summary_name=$1
    summary_cases=$2
    summary_input=$3
    summary_mean=$4
    summary_se=$5

    summary_folds=$(printf '%s\n' "$summary_input" | wc -l)
    printf '%s\n' "$summary_input" > "$test_dir/summary-$summary_name.input"
    "$binary" "$summary_cases" "$summary_folds" 1 0 \
        < "$test_dir/summary-$summary_name.input" \
        > "$test_dir/summary-$summary_name.raw"
    sed -e 's/[[:space:]]*$//' "$test_dir/summary-$summary_name.raw" \
        > "$test_dir/summary-$summary_name.actual"

    for summary_expected in "	Mean  $summary_mean" "	SE    $summary_se"
    do
        if ! grep -F -x -e "$summary_expected" \
            "$test_dir/summary-$summary_name.actual" > /dev/null; then
            printf 'summary (%s) did not print: %s\n' \
                "$summary_name" "$summary_expected" >&2
            cat "$test_dir/summary-$summary_name.actual" >&2
            return 1
        fi
    done
}

# The costs 16777217 and 16777219 are 16777216 and 16777220 in float. The sum
# of squares of the three costs is below the square of their sum in float, so
# the standard error is nan in float, but the mean is finite and keeps its
# float text.
run_summary_case rounded-costs 6 '2 1 (50.0%) 16777217
2 1 (50.0%) 16777217
2 1 (50.0%) 16777219' \
    '     2.0     50.0%  16777218.00' \
    '     0.0      0.0%   1.33'
# Unequal folds of 1e20, 2e20, and 3e20. The standard error is about 5.77e19,
# the square root of the sum of squared deviations divided by 2 and by 3.
run_summary_case unequal-large 6 '2 1 (50.0%) 100000000000000000000
2 1 (50.0%) 200000000000000000000
2 1 (50.0%) 300000000000000000000' \
    '     2.0     50.0%  200000004008175468544.00' \
    '     0.0      0.0%  57735028076023169024.00'
# Equal error rates without costs. The float sum of squares of the sizes is
# below the square of the sum, so Release 2.07 prints a standard error of
# -nan%.
run_summary_case equal-rate 33 '11 1 (9.1%)
11 1 (9.1%)
11 1 (9.1%)' \
    '    11.0      9.1%' \
    '     0.0      0.0%'

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
