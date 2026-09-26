#!/bin/sh

set -eu

script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)
repo_dir=$(CDPATH= cd "$script_dir/.." && pwd)
trainer=${C50_BINARY:-"$repo_dir/c5.0"}
predictor=${C50_PREDICTION_BINARY:-"$repo_dir/prediction-probe"}

if test ! -x "$trainer"; then
    printf 'C5.0 executable not found: %s\n' "$trainer" >&2
    exit 1
fi

if test ! -x "$predictor"; then
    printf 'C5.0 prediction probe not found: %s\n' "$predictor" >&2
    exit 1
fi

test_dir=$(mktemp -d "${TMPDIR:-/tmp}/c50-prediction-test.XXXXXX")
trap 'rm -rf "$test_dir"' EXIT HUP INT TERM

run_case()
{
    fixture_name=$1
    mode=$2
    shift 2

    fixture_dir="$script_dir/fixtures/$fixture_name"
    expected="$script_dir/expected/$fixture_name/predictions-$mode.csv"
    case_dir="$test_dir/$fixture_name-$mode"

    mkdir "$case_dir"
    cp "$fixture_dir/$fixture_name.names" "$case_dir/"
    cp "$fixture_dir/$fixture_name.data" "$case_dir/"
    cp "$fixture_dir/$fixture_name.test" "$case_dir/"

    (
        cd "$case_dir"
        "$trainer" -f "$fixture_name" "$@" >/dev/null
        "$predictor" "$case_dir/$fixture_name" "$mode"
    ) > "$case_dir/predictions.actual"

    diff -u "$expected" "$case_dir/predictions.actual"
}

run_invalid_model()
{
    fixture_dir="$script_dir/fixtures/basic"
    expected="$script_dir/expected/basic/invalid-model.output"
    case_dir="$test_dir/basic-invalid-model"

    mkdir "$case_dir"
    cp "$fixture_dir/basic.names" "$case_dir/"
    cp "$fixture_dir/basic.data" "$case_dir/"
    cp "$fixture_dir/basic.test" "$case_dir/"

    (
        cd "$case_dir"
        "$trainer" -f basic >/dev/null
        sed 's/att="group"/att="unknown"/' basic.tree > basic.tree.bad
        mv basic.tree.bad basic.tree
    )

    set +e
    "$predictor" "$case_dir/basic" tree > "$case_dir/output.actual" 2>&1
    exit_code=$?
    set -e

    if test "$exit_code" -ne 1; then
        printf 'invalid model returned %d, expected 1\n' "$exit_code" >&2
        return 1
    fi

    diff -u "$expected" "$case_dir/output.actual"
}

run_case basic tree
run_case basic rules -r
run_case boost tree -t 5
run_case multiclass tree
run_case multiclass rules -r
run_invalid_model

printf 'Prediction regression tests passed\n'
