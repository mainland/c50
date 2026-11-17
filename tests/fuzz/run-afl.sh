#!/bin/sh
# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

set -eu

if [ "$#" -ne 6 ]; then
    echo "usage: run-afl.sh AFL-FUZZ CORPUS OUTPUT DICTIONARY SECONDS TARGET" >&2
    exit 2
fi

afl_fuzz=$1
corpus=$2
output=$3
dictionary=$4
seconds=$5
target=$6

mkdir -p "$output"
if [ -f "$output/default/fuzzer_stats" ]; then
    input=-
else
    input=$corpus
fi

# An empty dictionary argument means that the target has none.
if [ -n "$dictionary" ]; then
    set -- -x "$dictionary"
else
    set --
fi

"$afl_fuzz" -V "$seconds" -i "$input" -o "$output" "$@" \
    -G 4096 -m "${C50_AFL_MEMORY_LIMIT:-512}" \
    -t "${C50_AFL_TIMEOUT:-5000}" -- "$target"

for finding in crashes hangs; do
    set -- "$output/default/$finding"/id:*
    if [ -e "$1" ]; then
        echo "AFL++ saved a finding under $output/default/$finding" >&2
        exit 1
    fi
done
