#!/bin/sh

set -eu

if test "$#" -ne 1; then
    printf 'usage: %s <native-workload>\n' "$0" >&2
    exit 2
fi

binary=$1
for rows in 9999 10000 20000; do
    reference=
    for workers in 1 2 4 8; do
        output=$("$binary" --rows "$rows" --features 12 \
            --categorical-features 4 --workers "$workers")
        model=$(printf '%s\n' "$output" |
            sed -n '/"serialized_bytes"/p; /"stable_fnv1a64"/p')
        if test -z "$model"; then
            printf 'missing classifier digest for %s rows, %s workers\n' \
                "$rows" "$workers" >&2
            exit 1
        fi
        if test "$workers" -eq 1; then
            reference=$model
        elif test "$model" != "$reference"; then
            printf 'classifier changed for %s rows, %s workers\n' \
                "$rows" "$workers" >&2
            exit 1
        fi
    done
done

reference=
for workers in 1 2 4 8; do
    output=$("$binary" --rows 20000 --features 12 \
        --categorical-features 4 --subsets --workers "$workers")
    model=$(printf '%s\n' "$output" |
        sed -n '/"serialized_bytes"/p; /"stable_fnv1a64"/p')
    if test -z "$model"; then
        printf 'missing subset classifier digest for %s workers\n' \
            "$workers" >&2
        exit 1
    fi
    if test "$workers" -eq 1; then
        reference=$model
    elif test "$model" != "$reference"; then
        printf 'subset classifier changed for %s workers\n' \
            "$workers" >&2
        exit 1
    fi
done

printf 'Parallel workload equivalence passed\n'
