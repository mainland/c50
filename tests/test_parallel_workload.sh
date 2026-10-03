#!/bin/sh

set -eu

if test "$#" -ne 1; then
    printf 'usage: %s <parallel-workload>\n' "$0" >&2
    exit 2
fi

binary=$1

# Require the same classifier digest from 1, 2, 4, and 8 split workers.
check_workers()
{
    label=$1
    shift
    reference=
    for workers in 1 2 4 8; do
        output=$("$binary" "$@" --workers "$workers")
        model=$(printf '%s\n' "$output" |
            sed -n '/"serialized_bytes"/p; /"stable_fnv1a64"/p')
        if test -z "$model"; then
            printf 'missing classifier digest for %s, %s workers\n' \
                "$label" "$workers" >&2
            exit 1
        fi
        if test "$workers" -eq 1; then
            reference=$model
        elif test "$model" != "$reference"; then
            printf 'classifier changed for %s, %s workers\n' \
                "$label" "$workers" >&2
            exit 1
        fi
    done
}

for rows in 9999 10000 20000; do
    check_workers "$rows rows" --rows "$rows" --features 12 \
        --categorical-features 4
done
check_workers "subset splits" --rows 20000 --features 12 \
    --categorical-features 4 --subsets
check_workers "stable ties" --rows 20000 --features 12 \
    --categorical-features 4 --value-levels 9 --ties stable

printf 'Parallel workload equivalence passed\n'
