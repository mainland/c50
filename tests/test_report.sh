#!/bin/sh

set -eu

script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)
repo_dir=$(CDPATH= cd "$script_dir/.." && pwd)
expected="$script_dir/expected/report/basic.output"
binary=${REPORT_BINARY:-"$repo_dir/report"}

if test ! -x "$binary"; then
    printf 'C5.0 report executable not found: %s\n' "$binary" >&2
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

printf 'Report regression tests passed\n'
