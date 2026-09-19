#!/bin/sh

set -eu

script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)
repo_dir=$(CDPATH= cd "$script_dir/.." && pwd)
binary=${C50_BINARY:-"$repo_dir/c5.0"}

if test ! -x "$binary"; then
    printf 'C5.0 executable not found: %s\n' "$binary" >&2
    exit 1
fi

test_dir=$(mktemp -d "${TMPDIR:-/tmp}/c50-cli-test.XXXXXX")
trap 'rm -rf "$test_dir"' EXIT HUP INT TERM

normalize_output()
{
    sed \
        -e 's|^C5\.0 \[Release 2\.07 GPL Edition\].*$|C5.0 [Release 2.07 GPL Edition] <timestamp>|' \
        -e 's|^Time: .* secs$|Time: <elapsed> secs|' \
        -e 's/[[:space:]]*$//' \
        "$1"
}

normalize_model()
{
    sed \
        -e 's|^id="See5/C5\.0 2\.07 GPL Edition [0-9-][0-9-]*"$|id="See5/C5.0 2.07 GPL Edition <date>"|' \
        "$1"
}

run_case()
{
    fixture_name=$1
    case_name=$2
    model_extension=$3
    shift 3

    fixture_dir="$script_dir/fixtures/$fixture_name"
    expected_dir="$script_dir/expected/$fixture_name"

    case_dir="$test_dir/$fixture_name-$case_name"
    mkdir "$case_dir"
    cp "$fixture_dir/$fixture_name.names" "$case_dir/"
    cp "$fixture_dir/$fixture_name.data" "$case_dir/"
    if test -f "$fixture_dir/$fixture_name.test"; then
        cp "$fixture_dir/$fixture_name.test" "$case_dir/"
    fi
    if test "${1:-}" = --with-costs; then
        cp "$fixture_dir/$fixture_name.costs" "$case_dir/"
        shift
    fi

    (
        cd "$case_dir"
        "$binary" -f "$fixture_name" "$@" > output.raw
    )

    normalize_output "$case_dir/output.raw" > "$case_dir/output.actual"
    diff -u "$expected_dir/$case_name.output" "$case_dir/output.actual"
    if test "$model_extension" != -; then
        normalize_model "$case_dir/$fixture_name.$model_extension" \
            > "$case_dir/model.actual"
        diff -u "$expected_dir/$case_name.$model_extension" \
            "$case_dir/model.actual"
    fi
}

run_case basic tree tree
run_case basic rules rules -r
run_case basic subsets tree -s
run_case basic winnow tree -w
run_case basic soft-thresholds tree -p
run_case basic sample tree -S 70 -I 17
run_case basic costs tree --with-costs
run_case basic cross-validation - -X 5 -I 17
run_case boost boost tree -t 5
run_case case-weight case-weight tree
run_case implicit implicit tree
run_case multiclass tree tree
run_case multiclass rules rules -r

printf 'CLI regression tests passed\n'
