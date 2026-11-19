#!/bin/sh
# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

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

# Program output may carry bytes that are not valid text, such as the
# `date-bytes` fixture's. BSD `sed` rejects them in a UTF-8 locale, so every
# `sed` runs byte-oriented in the C locale.
normalize_output()
{
    LC_ALL=C sed \
        -e 's|^C5\.0 \[Release 2\.07 GPL Edition\].*$|C5.0 [Release 2.07 GPL Edition] <timestamp>|' \
        -e 's|^Time: .* secs$|Time: <elapsed> secs|' \
        -e 's/[[:space:]]*$//' \
        "$1"
}

normalize_model()
{
    LC_ALL=C sed \
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

# Release 2.07 trains on a schema of more than 4,096 classes. The expected
# files come from Release 2.07 on the same names file, which has 4,097 classes
# and is generated here to keep the fixture small.
run_many_classes()
{
    many_case_dir="$test_dir/many-classes"
    mkdir "$many_case_dir"
    cp "$script_dir/fixtures/many-classes/many-classes.data" "$many_case_dir/"
    awk 'BEGIN {
        printf "outcome.\n\nx: continuous.\ng: p, q, r.\noutcome: "
        for (i = 0; i < 4097; i++) printf "%sc%d", (i ? ", " : ""), i
        print "."
    }' > "$many_case_dir/many-classes.names"

    for many_mode in tree rules; do
        many_options=
        if test "$many_mode" = rules; then
            many_options=-r
        fi
        (
            cd "$many_case_dir"
            # shellcheck disable=SC2086 # The options are one word or none.
            "$binary" -f many-classes $many_options > "$many_mode.raw"
        )
        normalize_output "$many_case_dir/$many_mode.raw" \
            > "$many_case_dir/$many_mode.actual"
        diff -u "$script_dir/expected/many-classes/$many_mode.output" \
            "$many_case_dir/$many_mode.actual"
        normalize_model "$many_case_dir/many-classes.$many_mode" \
            > "$many_case_dir/$many_mode.model"
        diff -u "$script_dir/expected/many-classes/$many_mode.$many_mode" \
            "$many_case_dir/$many_mode.model"
    done
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
# The case needs `/dev/full` to open for writing and fail on close. Only Linux
# and the BSDs behave so. macOS has no `/dev/full`, and under MSYS the device
# exists but a symbolic link to it cannot be opened for writing, so other
# platforms skip the case.
run_unwritable_classifier()
{
    case $(uname -s) in
        Linux|*BSD) ;;
        *) return 0 ;;
    esac
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

# The report must reach its destination or the program must say that it did
# not. Release 2.07 sends the report to standard output when `-o` cannot be
# opened, and exits 0 when a write to the `-o` file or to standard output
# fails. This project reports each failure on standard error and exits 1. The
# failed open runs everywhere. The other cases need a full device or a
# file-size limit, so they run on Linux and the BSDs, and the limit cases also
# need a shell that accepts `ulimit -f 0`. Each case that cannot run prints a
# skip message. Standard error goes to a pipe, because the limit applies to a
# file there too.
expect_report_failure()
{
    failure_name=$1
    failure_message=$2
    shift 2

    set +e
    failure_stderr=$("$@" 2>&1 >/dev/null)
    failure_code=$?
    set -e
    if test "$failure_code" -ne 1; then
        printf 'report failure %s returned %d, expected 1\n' \
            "$failure_name" "$failure_code" >&2
        return 1
    fi
    case $failure_stderr in
        "$failure_message"*) ;;
        *)
            printf 'report failure %s printed unexpected text:\n%s\n' \
                "$failure_name" "$failure_stderr" >&2
            return 1
            ;;
    esac
}

# A write that fails once, when later writes succeed, leaves the stream in its
# error state without an error number: the final flush succeeds, and `errno`
# keeps no trace of the earlier failure. The report must still fail, with the
# message `write error` in place of the system's description. The case injects
# one `ENOSPC` into the first write to the report with `strace`, and it needs
# more than one buffer of output, so that later writes can succeed.
# LeakSanitizer cannot run under `strace`, so the traced program runs without
# leak detection.
expect_transient_write_failure()
{
    transient_name=$1
    transient_target=$2
    transient_message=$3
    shift 3

    : > "$transient_target"
    set +e
    transient_stderr=$(
        ASAN_OPTIONS="${ASAN_OPTIONS:+$ASAN_OPTIONS:}detect_leaks=0" \
        strace -f -o "$report_dir/strace.log" -P "$transient_target" \
            -e trace=write -e inject=write:error=ENOSPC:when=1 \
            "$@" 2>&1 >"$report_dir/transient.stdout")
    transient_code=$?
    set -e
    if test "$transient_code" -ne 1; then
        printf 'transient write failure %s returned %d, expected 1\n' \
            "$transient_name" "$transient_code" >&2
        return 1
    fi
    case $transient_stderr in
        "$transient_message") ;;
        *)
            printf 'transient write failure %s printed unexpected text:\n%s\n' \
                "$transient_name" "$transient_stderr" >&2
            return 1
            ;;
    esac
}

run_report_failures()
{
    report_dir="$test_dir/report-failures"
    mkdir "$report_dir"
    cp "$script_dir/fixtures/basic/basic.names" "$report_dir/"
    cp "$script_dir/fixtures/basic/basic.data" "$report_dir/"

    set -- "$binary" -f "$report_dir/basic" -X 2 -I 17
    missing_path="$report_dir/missing/report.txt"
    # Under MSYS the program receives the path that the shell converts to a
    # Windows path, and it names that path in the message.
    open_path=$missing_path
    if command -v cygpath > /dev/null 2>&1; then
        open_path=$(cygpath -m "$missing_path")
    fi
    expect_report_failure open-failure \
        "*** could not open $open_path: " "$@" -o "$missing_path"
    if test -e "$missing_path"; then
        printf 'open failure created %s\n' "$missing_path" >&2
        return 1
    fi
    # The failed open must not send the report to standard output.
    set +e
    open_stdout=$("$@" -o "$missing_path" 2>/dev/null)
    set -e
    if test -n "$open_stdout"; then
        printf 'open failure wrote a report to standard output\n' >&2
        return 1
    fi

    case $(uname -s) in
        Linux|*BSD) ;;
        *)
            printf 'skipping the full-device and file-size report failures:' >&2
            printf ' they need Linux or a BSD\n' >&2
            return 0
            ;;
    esac
    if test ! -c /dev/full; then
        printf 'skipping the full-device and file-size report failures:' >&2
        printf ' /dev/full is missing\n' >&2
        return 0
    fi

    expect_report_failure full-device \
        '*** could not write /dev/full: ' "$@" -o /dev/full

    full_stdout_message='*** could not write standard output: '
    set +e
    full_stderr=$(
        "$binary" -f "$report_dir/basic" -X 2 -I 17 2>&1 >/dev/full)
    full_code=$?
    set -e
    case $full_stderr in
        "$full_stdout_message"*) ;;
        *)
            printf 'full standard output printed unexpected text:\n%s\n' \
                "$full_stderr" >&2
            return 1
            ;;
    esac
    if test "$full_code" -ne 1; then
        printf 'full standard output returned %d, expected 1\n' \
            "$full_code" >&2
        return 1
    fi

    if test "$(uname -s)" = Linux && command -v strace > /dev/null 2>&1 &&
        strace -o /dev/null -e trace=write \
            -e inject=write:error=ENOSPC:when=1 true > /dev/null 2>&1; then
        transient_path="$report_dir/transient.txt"
        set -- "$binary" -f "$report_dir/basic" -X 10 -t 3 -I 17
        expect_transient_write_failure standard-output \
            "$report_dir/transient.stdout" \
            '*** could not write standard output: write error' "$@"
        expect_transient_write_failure report-file "$transient_path" \
            "*** could not write $transient_path: write error" \
            "$@" -o "$transient_path"
        set -- "$binary" -f "$report_dir/basic" -X 2 -I 17
    else
        printf 'skipping the transient report write failure:' >&2
        printf ' it needs Linux and strace with fault injection\n' >&2
    fi

    # An ordinary file under a zero-byte limit fails on write, not on open.
    # Ignoring SIGXFSZ turns the signal into an error from the write.
    if ! (ulimit -f 0) > /dev/null 2>&1; then
        printf 'skipping the file-size report failure:' >&2
        printf ' the shell rejects ulimit -f 0\n' >&2
        return 0
    fi
    limit_path="$report_dir/limited.txt"
    set +e
    limit_result=$(
        (
            ulimit -f 0
            trap '' XFSZ
            "$@" -o "$limit_path" 2>&1 >/dev/null
            printf 'status=%d\n' "$?"
        )
    )
    set -e
    case $limit_result in
        "*** could not write $limit_path: "*"
status=1") ;;
        *)
            printf 'limited report printed unexpected text:\n%s\n' \
                "$limit_result" >&2
            return 1
            ;;
    esac
}

# A read error is not the end of the input. Release 2.07 treats it as the end,
# so a directory named `basic.test` evaluates 0 cases and exits 0. This
# project reports the failure once and exits 1, so the exit status has no
# reference. Reading a directory fails with EISDIR on Linux, macOS, and the
# BSDs. Other platforms, where fopen rejects a directory, skip the case.
run_read_error()
{
    case $(uname -s) in
        Linux|Darwin|*BSD) ;;
        *) return 0 ;;
    esac
    for read_error_input in data test; do
        read_error_dir="$test_dir/read-error-$read_error_input"
        mkdir "$read_error_dir"
        cp "$script_dir/fixtures/basic/basic.names" "$read_error_dir/"
        if test "$read_error_input" = data; then
            cp "$script_dir/fixtures/basic/basic.test" "$read_error_dir/"
        else
            cp "$script_dir/fixtures/basic/basic.data" "$read_error_dir/"
        fi
        mkdir "$read_error_dir/basic.$read_error_input"

        set +e
        (
            cd "$read_error_dir"
            "$binary" -f basic > output.raw 2>&1
        )
        read_error_exit_code=$?
        set -e

        if test "$read_error_exit_code" -ne 1; then
            printf 'read error in basic.%s returned %d, expected 1\n' \
                "$read_error_input" "$read_error_exit_code" >&2
            return 1
        fi
        read_error_messages=$(grep -c -E \
            "^\*\*\* could not read basic\.$read_error_input: " \
            "$read_error_dir/output.raw" || true)
        if test "$read_error_messages" -ne 1; then
            printf 'read error in basic.%s was not reported once\n' \
                "$read_error_input" >&2
            cat "$read_error_dir/output.raw" >&2
            return 1
        fi
    done
}

# A bad data value that the run survives must not hide the message of a later
# failure. Release 2.07 reports neither failure and exits 0, so the exit status
# has no reference. This project reports the later failure once and exits 1.
# Reading a directory fails on Linux, macOS, and the BSDs only, so other
# platforms skip the read cases. The `full` case also needs `/dev/full` to
# open for writing and fail on close, which holds on Linux and the BSDs only.
# macOS lacks the device and MSYS cannot open a link to it, so they skip it.
run_failure_after_bad_values()
{
    after_kind=$1
    after_message=$2
    case $after_kind in
        full)
            case $(uname -s) in
                Linux|*BSD) ;;
                *) return 0 ;;
            esac
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

# Training data with no cases. Release 2.07 crashes with a segmentation fault
# when a tree classifier trained on no cases evaluates a case, and so does a
# cross-validation fold that trains on no cases, so those runs have no
# reference status. This project classifies every case as the first class, as
# Release 2.07 does for rules, and exits 0. A cross-validation run reads
# `basic.test` as more data, so that case omits it.
run_zero_training()
{
    zero_test=yes
    if test "$1" = --no-test; then
        zero_test=no
        shift
    fi
    zero_name=$1
    zero_data=$2
    zero_evaluated=$3
    shift 3

    zero_dir="$test_dir/zero-training-$zero_name"
    mkdir "$zero_dir"
    cp "$script_dir/fixtures/basic/basic.names" "$zero_dir/"
    if test "$zero_test" = yes; then
        cp "$script_dir/fixtures/basic/basic.test" "$zero_dir/"
    fi
    printf '%s' "$zero_data" > "$zero_dir/basic.data"

    set +e
    (
        cd "$zero_dir"
        "$binary" -f basic "$@" > output.raw 2>&1
    )
    zero_exit_code=$?
    set -e

    if test "$zero_exit_code" -ne 0; then
        printf 'zero training cases (%s) returned %d, expected 0\n' \
            "$zero_name" "$zero_exit_code" >&2
        return 1
    fi
    if ! grep -F -x -e "$zero_evaluated" "$zero_dir/output.raw" > /dev/null; then
        printf 'zero training cases (%s) did not report: %s\n' \
            "$zero_name" "$zero_evaluated" >&2
        cat "$zero_dir/output.raw" >&2
        return 1
    fi
}

# Cross-validation with no case left to validate: an empty data file, data in
# which every case has an unknown class, or a sample that selects no case.
# Release 2.07 divides by zero and dies from SIGFPE. This project reports the
# error and exits 1. One case is refused as well, because a fold needs a case
# to hold out and a case to train on. Release 2.07 reduces the folds to one,
# trains on no case, prints a standard error of nan, and exits 0, or crashes
# when the tree classifies a case. The option `--one` expects the message
# for one case.
run_empty_cross_validation()
{
    empty_message='*** cannot cross-validate with no cases'
    if test "$1" = --one; then
        empty_message='*** cannot cross-validate with one case'
        shift
    fi
    empty_name=$1
    empty_names=$2
    empty_data=$3
    shift 3

    empty_dir="$test_dir/empty-cross-validation-$empty_name"
    mkdir "$empty_dir"
    printf '%s' "$empty_names" > "$empty_dir/empty.names"
    printf '%s' "$empty_data" > "$empty_dir/empty.data"

    set +e
    (
        cd "$empty_dir"
        "$binary" -f empty "$@" > output.raw 2>&1
    )
    empty_exit_code=$?
    set -e

    if test "$empty_exit_code" -ne 1; then
        printf 'cross-validation on no cases (%s) returned %d, expected 1\n' \
            "$empty_name" "$empty_exit_code" >&2
        cat "$empty_dir/output.raw" >&2
        return 1
    fi
    if ! grep -F -x -e "$empty_message" \
        "$empty_dir/output.raw" > /dev/null; then
        printf 'cross-validation on no cases (%s) did not report the error\n' \
            "$empty_name" >&2
        cat "$empty_dir/output.raw" >&2
        return 1
    fi
}

# Two cases are the least that cross-validation accepts. Each of the two folds
# trains on one case and holds out the other.
run_two_case_cross_validation()
{
    two_dir="$test_dir/two-case-cross-validation"
    mkdir "$two_dir"
    printf 'c.\nx: continuous.\nc: a, b.\n' > "$two_dir/two.names"
    printf '1, a\n2, b\n' > "$two_dir/two.data"

    set +e
    (
        cd "$two_dir"
        "$binary" -f two -X 2 -I 1 > output.raw 2>&1
    )
    two_exit_code=$?
    set -e

    if test "$two_exit_code" -ne 0; then
        printf 'cross-validation on two cases returned %d, expected 0\n' \
            "$two_exit_code" >&2
        cat "$two_dir/output.raw" >&2
        return 1
    fi
    if ! grep -F -x -e '[ Summary ]' "$two_dir/output.raw" > /dev/null ||
        grep -F -e 'cannot cross-validate' -e nan "$two_dir/output.raw" \
            > /dev/null; then
        printf 'cross-validation on two cases printed unexpected text:\n' >&2
        cat "$two_dir/output.raw" >&2
        return 1
    fi
}

# Rules with a costs file on data with no cases or with one class among three.
# The average cost of the rule normalization is then 0/0. Release 2.07 forms no
# rules, writes the first class or the only class as the default, and exits 0.
# This project does the same.
run_rule_costs_degenerate()
{
    degenerate_name=$1
    degenerate_data=$2
    shift 2

    degenerate_dir="$test_dir/rule-costs-$degenerate_name"
    mkdir "$degenerate_dir"
    cp "$script_dir/fixtures/multiclass/multiclass.names" "$degenerate_dir/"
    cp "$script_dir/fixtures/multiclass/multiclass.test" "$degenerate_dir/"
    printf '%s' "$degenerate_data" > "$degenerate_dir/multiclass.data"
    printf 'red, green: 2\nblue, red: 3\n' > "$degenerate_dir/multiclass.costs"

    set +e
    (
        cd "$degenerate_dir"
        "$binary" -f multiclass -r "$@" > output.raw 2>&1
    )
    degenerate_exit_code=$?
    set -e

    if test "$degenerate_exit_code" -ne 0; then
        printf 'rule costs on %s (%s) returned %d, expected 0\n' \
            "$degenerate_name" "$*" "$degenerate_exit_code" >&2
        cat "$degenerate_dir/output.raw" >&2
        return 1
    fi
    if ! grep -F -x -e 'Default class: red' \
        "$degenerate_dir/output.raw" > /dev/null; then
        printf 'rule costs on %s (%s) did not default to red\n' \
            "$degenerate_name" "$*" >&2
        cat "$degenerate_dir/output.raw" >&2
        return 1
    fi
    printf 'costs="1"\nentries="1"\nrules="0" default="red"\n' \
        > "$degenerate_dir/rules.expected"
    normalize_model "$degenerate_dir/multiclass.rules" |
        LC_ALL=C sed -e '1d' > "$degenerate_dir/rules.actual"
    if ! cmp -s "$degenerate_dir/rules.expected" \
        "$degenerate_dir/rules.actual"; then
        printf 'rule costs on %s (%s) wrote an unexpected model\n' \
            "$degenerate_name" "$*" >&2
        cat "$degenerate_dir/rules.actual" >&2
        return 1
    fi
}

run_invalid_option()
{
    option_name=$1
    shift

    set +e
    "$binary" "$@" > "$test_dir/$option_name.actual" 2>&1
    option_exit_code=$?
    set -e

    if test "$option_exit_code" -ne 1; then
        printf '%s returned %d, expected 1\n' \
            "$option_name" "$option_exit_code" >&2
        return 1
    fi
}

# A value that overflows, is not finite, or is out of range reports the
# "illegal value" diagnostic. A value that is not a number, or has trailing
# text, is an unrecognized option, as in Release 2.07.
expect_option_message()
{
    option_name=$1
    expected_message=$2
    shift 2

    run_invalid_option "$option_name" "$@"
    if ! grep -F -x -e "$expected_message" \
        "$test_dir/$option_name.actual" > /dev/null; then
        printf '%s did not report: %s\n' "$option_name" "$expected_message" >&2
        cat "$test_dir/$option_name.actual" >&2
        return 1
    fi
}

illegal='	** illegal value'
expect_option_message sample-nan \
    "$illegal nan -- should be between 0.1 and 99.9" -S nan
expect_option_message seed-overflow \
    "$illegal 1e+21 -- should be between -2147483648 and 2147483647" \
    -I 999999999999999999999
expect_option_message seed-beyond-int \
    "$illegal 1e+11 -- should be between -2147483648 and 2147483647" \
    -I 99999999999
# The bounds and the value of an integer option print with enough digits to
# differ, where six significant digits print both as 2.14748e+09.
expect_option_message seed-above-int \
    "$illegal 2147483648 -- should be between -2147483648 and 2147483647" \
    -I 2147483648
expect_option_message seed-below-int \
    "$illegal -2147483649 -- should be between -2147483648 and 2147483647" \
    -I -2147483649
expect_option_message trials-beyond-int \
    "$illegal 1e+11 -- should be between 3 and 1000" -t 99999999999
expect_option_message minimum-nan \
    "$illegal nan -- should be between 1 and 1e+06" -m nan
expect_option_message minimum-beyond-float \
    "$illegal 1e+39 -- should be between 1 and 1e+06" -m 1e39
expect_option_message confidence-infinite \
    "$illegal inf -- should be between 0 and 100" -c inf
expect_option_message confidence-overflow \
    "$illegal inf -- should be between 0 and 100" -c 1e999
expect_option_message confidence-range \
    "$illegal 101 -- should be between 0 and 100" -c 101
expect_option_message utility-overflow \
    "$illegal 1e+20 -- should be between 2 and 10000" -u 99999999999999999999
expect_option_message folds-trailing-text \
    "    **  Unrecognised option -X 5x" -X 5x
expect_option_message confidence-trailing-text \
    "    **  Unrecognised option -c 25x" -c 25x
expect_option_message trials-not-numeric \
    "    **  Unrecognised option -t many" -t many
long_stem=$(printf '%600s' '' | tr ' ' x)
run_invalid_option long-stem -f "$long_stem"

# An empty value is the number 0, because Release 2.07 only tests that strtod
# and strtol stopped at the terminating null. A value that underflows is its
# rounded value, not an overflow. The expected lines come from Release 2.07 on
# the same data.
expect_option_accepted()
{
    accepted_name=$1
    expected_line=$2
    shift 2

    accepted_dir="$test_dir/accepted-$accepted_name"
    mkdir "$accepted_dir"
    cp "$script_dir/fixtures/basic/basic.names" \
        "$script_dir/fixtures/basic/basic.data" "$accepted_dir/"
    set +e
    (cd "$accepted_dir" && "$binary" -f basic "$@" > output.raw 2>&1)
    accepted_status=$?
    set -e
    if test "$accepted_status" -ne 0; then
        printf '%s returned %d, expected 0\n' "$accepted_name" \
            "$accepted_status" >&2
        return 1
    fi
    if ! grep -F -x -e "$expected_line" "$accepted_dir/output.raw" \
        > /dev/null ||
        ! grep -F -x -e 'Decision tree:' "$accepted_dir/output.raw" \
        > /dev/null; then
        printf '%s did not train with: %s\n' "$accepted_name" \
            "$expected_line" >&2
        cat "$accepted_dir/output.raw" >&2
        return 1
    fi
}

expect_option_accepted seed-empty '	Random seed 0' -I ''
expect_option_accepted confidence-empty '	Pruning confidence level 0%' -c ''
expect_option_accepted confidence-underflow \
    '	Pruning confidence level 0%' -c 1e-400
expect_option_accepted confidence-subnormal \
    '	Pruning confidence level 0%' -c 1e-320
expect_option_message minimum-empty \
    "$illegal 0 -- should be between 1 and 1e+06" -m ''
expect_option_message trials-empty \
    "$illegal 0 -- should be between 3 and 1000" -t ''
expect_option_message sample-empty \
    "$illegal 0 -- should be between 0.1 and 99.9" -S ''
expect_option_message folds-empty \
    "$illegal 0 -- should be between 2 and 1000" -X ''

run_case basic tree tree
run_case basic rules rules -r
run_case basic subsets tree -s
run_case basic winnow tree -w
run_case basic soft-thresholds tree -p
run_case soft-threshold-ties soft-thresholds tree -p
run_case basic sample tree -S 70 -I 17
run_case basic costs tree --with-costs
run_case basic cross-validation - -X 5 -I 17
run_case print-rounding cross-validation - -X 3 -b -I 3
run_case boost boost tree -t 5
run_case case-weight case-weight tree
run_case winnow-case-weights winnow tree -w
run_case winnow-case-weight-costs winnow tree --with-costs -w
# Two-class costs and a low confidence factor raise the pessimistic error
# estimate of the root's subtree above the number of cases, and neither branch
# of the root is a subtree. Release 2.07 replaces the root with a branch that
# does not exist and crashes, in winnowing's trial tree for the second fixture,
# so these runs have no reference output. This project keeps the root's split.
run_case costs-pruning tree tree --with-costs -c 5
run_case costs-pruning rules rules --with-costs -c 5 -r
run_case winnow-costs-pruning winnow tree --with-costs -w -c 5
run_case implicit implicit tree
run_case implicit-unknown tree tree
run_case implicit-literals tree tree
run_case implicit-log tree tree
run_case implicit-divide tree tree
# Release 2.07 keeps the division precision multiplier in an int that wraps.
# The expected files come from Release 2.07, whose cuts differ from the
# unwrapped quotient, and whose multiplier reaches 0 in the last case, so
# that every value of r becomes unknown and the run stops at the error limit.
run_case implicit-divide-wrap tree tree
run_case implicit-divide-wrap rules rules -r
# The rounding error of division forms in double precision, as in Release 2.07,
# so that the cut is 0.1662537. A float subtraction gives
# 0.166254.
run_case implicit-divide-precision tree tree
run_case implicit-divide-negative tree tree
run_case implicit-divide-zero tree - --status 1
run_case implicit-power tree tree
# Bytes at or above 0x80 in date, time, and timestamp values. The expected files
# come from Release 2.07 on the same data.
run_case date-bytes tree tree
run_case multiclass tree tree
run_case multiclass rules rules -r
run_case multiclass sample tree -S 30 -I 17
# Bad values in the data. The expected files and statuses come from Release
# 2.07, which continues with each bad value as unknown and exits 0 below ten
# errors, and stops with exit 1 at the tenth.
run_case bad-values tree tree
run_case error-limit tree - --status 1
run_zero_training tree '' 'Evaluation on test data (7 cases):'
run_zero_training comments '# no cases
' 'Evaluation on test data (7 cases):'
run_zero_training rules '' 'Evaluation on test data (7 cases):' -r
run_zero_training boost '' 'Evaluation on test data (7 cases):' -t 3
run_zero_training winnow '' 'Evaluation on test data (7 cases):' -w
run_zero_training subsets '' 'Evaluation on test data (7 cases):' -s
empty_names_plain='c.
x: continuous.
c: a, b.
'
empty_names_bare='c.
c: a, b.
'
run_empty_cross_validation tree "$empty_names_plain" '' -X 3
run_empty_cross_validation two-folds "$empty_names_plain" '' -X 2
run_empty_cross_validation rules "$empty_names_plain" '' -X 3 -r
run_empty_cross_validation no-attributes "$empty_names_bare" '' -X 3
run_empty_cross_validation no-attributes-rules "$empty_names_bare" '' -X 3 -r
empty_names_group='c.
x: continuous.
g: a, b.
c: a, b.
'
run_empty_cross_validation unknown-class "$empty_names_group" '1, a, ?
2, b, ?
' -X 3
run_empty_cross_validation sample-selects-none "$empty_names_group" '1, a, a
2, b, b
' -X 2 -S 1
run_empty_cross_validation --one one-case "$empty_names_plain" '1, a
' -X 2
run_empty_cross_validation --one one-case-rules "$empty_names_plain" '1, a
' -X 2 -r
run_empty_cross_validation --one one-case-many-folds "$empty_names_plain" \
    '1, a
' -X 10
run_empty_cross_validation --one unknown-class-leaves-one "$empty_names_group" \
    '1, a, a
2, b, ?
' -X 2
run_empty_cross_validation --one sample-leaves-one "$empty_names_group" \
    '1, a, a
2, b, b
' -X 2 -S 50 -I 1
run_two_case_cross_validation
one_class_data='1, circle, red.
2, circle, red.
3, circle, red.
4, circle, red.
6, triangle, red.
7, triangle, red.
8, triangle, red.
9, triangle, red.
'
run_rule_costs_degenerate zero-cases ''
run_rule_costs_degenerate zero-cases-bagged '' -b
run_rule_costs_degenerate zero-cases-utility '' -u 2
run_rule_costs_degenerate one-class "$one_class_data"
run_rule_costs_degenerate one-class-bagged "$one_class_data" -b
run_rule_costs_degenerate one-class-utility "$one_class_data" -u 2
run_invalid_definition
run_many_classes
run_missing_file_format
run_unwritable_classifier
run_report_failures
run_read_error
run_failure_after_bad_values full '\*\*\* could not close classifier'
run_failure_after_bad_values test '\*\*\* could not read bad-values\.test: .*'
run_failure_after_bad_values costs \
    '\*\*\* could not read bad-values\.costs: .*'

printf 'CLI regression tests passed\n'
