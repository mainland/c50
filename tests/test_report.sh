#!/bin/sh

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

"$binary" 20 2 1 0 > "$test_dir/output.raw" <<'EOF'
3 1 (10.0%)
5 2 (20.0%)
EOF

sed -e 's/[[:space:]]*$//' "$test_dir/output.raw" \
    > "$test_dir/output.actual"
diff -u "$expected" "$test_dir/output.actual"

"$binary" 20 2 1 0 > "$test_dir/marked.raw" <<'EOF'
3 1 (10.0%)   <<
5 2 (20.0%)   <<
EOF

sed -e 's/[[:space:]]*$//' "$test_dir/marked.raw" \
    > "$test_dir/marked.actual"
diff -u "$expected" "$test_dir/marked.actual"

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
    diff -u "$case_dir/plain.output" "$case_dir/marked.output"
}

run_cross_validation basic tree 0
run_cross_validation basic rules 1 -r
run_cross_validation boost boost 0 -t 5
run_cross_validation basic costs 0

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

    diff -u "$script_dir/expected/report/$expected_name.output" \
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
diff -u "$expected" "$test_dir/long-line.actual"

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
