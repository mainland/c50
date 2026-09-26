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
    # A run that stops at the error limit exits 1, so it names that status.
    expected_status=0
    if test "${1:-}" = --status; then
        expected_status=$2
        shift 2
    fi

    set +e
    (
        cd "$case_dir"
        "$binary" -f "$fixture_name" "$@" > output.raw
    )
    actual_status=$?
    set -e
    if test "$actual_status" -ne "$expected_status"; then
        printf '%s %s returned %d, expected %d\n' "$fixture_name" \
            "$case_name" "$actual_status" "$expected_status" >&2
        return 1
    fi

    normalize_output "$case_dir/output.raw" > "$case_dir/output.actual"
    diff -u "$expected_dir/$case_name.output" "$case_dir/output.actual"
    if test "$model_extension" != -; then
        normalize_model "$case_dir/$fixture_name.$model_extension" \
            > "$case_dir/model.actual"
        diff -u "$expected_dir/$case_name.$model_extension" \
            "$case_dir/model.actual"
    fi
}

run_invalid_definition()
{
    invalid_fixture_dir="$script_dir/fixtures/invalid-definition"
    invalid_expected_dir="$script_dir/expected/invalid-definition"
    invalid_case_dir="$test_dir/invalid-definition"

    mkdir "$invalid_case_dir"
    cp "$invalid_fixture_dir/invalid.names" "$invalid_case_dir/"
    cp "$invalid_fixture_dir/invalid.data" "$invalid_case_dir/"

    set +e
    (
        cd "$invalid_case_dir"
        "$binary" -f invalid > output.raw 2>&1
    )
    invalid_exit_code=$?
    set -e

    if test "$invalid_exit_code" -ne 1; then
        printf 'invalid definition returned %d, expected 1\n' \
            "$invalid_exit_code" >&2
        return 1
    fi

    normalize_output "$invalid_case_dir/output.raw" \
        > "$invalid_case_dir/output.actual"
    diff -u "$invalid_expected_dir/output" "$invalid_case_dir/output.actual"
}

run_missing_file_format()
{
    missing_case_dir="$test_dir/missing-file-format"
    mkdir "$missing_case_dir"

    set +e
    (
        cd "$missing_case_dir"
        "$binary" -f 'missing%s' > output.raw 2>&1
    )
    missing_exit_code=$?
    set -e

    if test "$missing_exit_code" -ne 1; then
        printf 'missing file returned %d, expected 1\n' \
            "$missing_exit_code" >&2
        return 1
    fi

    normalize_output "$missing_case_dir/output.raw" \
        > "$missing_case_dir/output.actual"
    diff -u "$script_dir/expected/missing-file-format/output" \
        "$missing_case_dir/output.actual"
}

# A failure that `Error` does not print, such as a classifier file that cannot
# be written, reports its message once on the output stream. Release 2.07
# exits 0 for the same write failure, so the exit status has no reference.
run_unwritable_classifier()
{
    if test ! -c /dev/full; then
        return 0
    fi
    full_case_dir="$test_dir/unwritable-classifier"
    mkdir "$full_case_dir"
    cp "$script_dir/fixtures/basic/basic.names" "$full_case_dir/"
    cp "$script_dir/fixtures/basic/basic.data" "$full_case_dir/"
    ln -s /dev/full "$full_case_dir/basic.tree"

    set +e
    (
        cd "$full_case_dir"
        "$binary" -f basic > output.raw 2>&1
    )
    full_exit_code=$?
    set -e

    if test "$full_exit_code" -ne 1; then
        printf 'unwritable classifier returned %d, expected 1\n' \
            "$full_exit_code" >&2
        return 1
    fi
    full_messages=$(grep -c -F '*** could not' "$full_case_dir/output.raw" || true)
    if test "$full_messages" -ne 1 ||
        ! grep -F -x -e '*** could not close classifier' \
            "$full_case_dir/output.raw" > /dev/null; then
        printf 'unwritable classifier did not report its failure once\n' >&2
        cat "$full_case_dir/output.raw" >&2
        return 1
    fi
}

# A bad data value that the run survives must not hide the message of a later
# failure. Release 2.07 reports neither failure and exits 0, so the exit status
# has no reference. This project reports the later failure once and exits 1.
# Reading a directory fails on Linux, macOS, and the BSDs only, so other
# platforms skip the read cases.
run_failure_after_bad_values()
{
    after_kind=$1
    after_message=$2
    case $after_kind in
        full)
            if test ! -c /dev/full; then
                return 0
            fi
            ;;
        *)
            case $(uname -s) in
                Linux|Darwin|*BSD) ;;
                *) return 0 ;;
            esac
            ;;
    esac

    after_dir="$test_dir/bad-values-then-$after_kind"
    mkdir "$after_dir"
    cp "$script_dir/fixtures/bad-values/bad-values.names" "$after_dir/"
    cp "$script_dir/fixtures/bad-values/bad-values.data" "$after_dir/"
    case $after_kind in
        full) ln -s /dev/full "$after_dir/bad-values.tree" ;;
        test) mkdir "$after_dir/bad-values.test" ;;
        costs) mkdir "$after_dir/bad-values.costs" ;;
    esac

    set +e
    (
        cd "$after_dir"
        "$binary" -f bad-values > output.raw 2>&1
    )
    after_exit_code=$?
    set -e

    if test "$after_exit_code" -ne 1; then
        printf 'bad values then %s returned %d, expected 1\n' \
            "$after_kind" "$after_exit_code" >&2
        return 1
    fi
    after_messages=$(grep -c -E "^\*\*\* could not " \
        "$after_dir/output.raw" || true)
    if test "$after_messages" -ne 1 ||
        ! grep -E -x -e "$after_message" "$after_dir/output.raw" > /dev/null
    then
        printf 'bad values then %s did not report its failure once\n' \
            "$after_kind" >&2
        cat "$after_dir/output.raw" >&2
        return 1
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
run_case print-rounding cross-validation - -X 3 -b -I 3
run_case boost boost tree -t 5
run_case case-weight case-weight tree
run_case implicit implicit tree
run_case implicit-unknown tree tree
run_case implicit-literals tree tree
run_case implicit-log tree tree
run_case implicit-divide tree tree
# The rounding error of division forms in double precision, as in Release 2.07,
# so that the cut is 0.1662537. A float subtraction gives
# 0.166254.
run_case implicit-divide-precision tree tree
run_case implicit-power tree tree
run_case multiclass tree tree
run_case multiclass rules rules -r
# Bad values in the data. The expected files and statuses come from Release
# 2.07, which continues with each bad value as unknown and exits 0 below ten
# errors, and stops with exit 1 at the tenth.
run_case bad-values tree tree
run_case error-limit tree - --status 1
run_invalid_definition
run_missing_file_format
run_unwritable_classifier
run_failure_after_bad_values full '\*\*\* could not close classifier'

printf 'CLI regression tests passed\n'
