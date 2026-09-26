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

printf 'Report regression tests passed\n'
