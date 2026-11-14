#!/bin/sh
# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Fail when a source file calls a C++ math overload for `float` or
# `long double`. Release 2.07 is C, where log(), exp(), rint(), and the like
# take and return `double`, so a call that C++ resolves to `logf()` or
# `rintf()` computes in a different precision and can change a classifier.
#
# Each source compiles at `-O0 -fno-inline`, which keeps every call to an
# inline overload as a weak symbol that `nm` lists. An optimized library
# would inline `rintf()` and `fabsf()` and hide them. Each source compiles
# twice, once as the default build does and once with `-DVerbOpt`, because
# the documented verbose build compiles the code inside `Verbosity(...)`.
#
# The check fails when `nm` fails, when no object lists any symbol, and when
# it does not report the `float` overload that `math_overload_control.cpp`
# calls, so that it cannot pass without looking at the objects.
#
# usage: check_math_overloads.sh <c++ compiler> <nm> <source dir> <work dir>

set -eu

if test "$#" -ne 4; then
    printf 'usage: %s <c++ compiler> <nm> <source dir> <work dir>\n' \
        "$0" >&2
    exit 2
fi

cxx=$1
nm_command=$2
source_dir=$3
work_dir=$4

mkdir -p "$work_dir"
found="$work_dir/overloads.txt"
control_found="$work_dir/control.txt"
symbols="$work_dir/symbols.txt"
: > "$found"
: > "$symbols"

# Compile `$1` with the extra flags `$3` (empty or one word) into `$2`.
compile() {
    # shellcheck disable=SC2086
    "$cxx" -std=c++17 -O0 -fno-inline -fPIC -w -pthread $3 \
        -I "$source_dir/include" -I "$source_dir/src" \
        -c "$1" -o "$2"
}

# List the math overloads for `float` or `long double` that `$1` defines, one
# per line, and append every defined symbol to "$symbols". `nm` writes to a
# file first so that its exit status stops the script.
scan() {
    listing="$1.nm"
    LC_ALL=C "$nm_command" -C --defined-only "$1" > "$listing"
    cat "$listing" >> "$symbols"
    # A weak definition `std::rint(float)` appears in the object file of each
    # source that calls the overload. Names that are not math functions, and
    # the classification functions that C also leaves in `float`, do not
    # match. `isfinite` is exact for every precision.
    awk -v object="$(basename "$1")" '
        {
            symbol = $0
            sub(/^[0-9a-f]+ [A-Za-z] /, "", symbol)
            name = symbol
            sub(/\(.*$/, "", name)
            sub(/^(std::)?(__[0-9A-Za-z_]+::)?/, "", name)
            if ( name !~ /^(a?cosh?|a?sinh?|a?tanh?|atan2|cbrt|ceil|exp|exp2|expm1|fabs|abs|floor|fmod|frexp|hypot|ldexp|log|log10|log1p|log2|modf|nearbyint|pow|remainder|rint|round|sqrt|trunc|lround|llround|lrint|llrint|fmin|fmax|fdim|fma|copysign)$/ ) next
            if ( symbol !~ /\((.*[ (])?(float|long double)[,)]/ ) next
            print object ": " symbol
        }' "$listing"
}

compile "$source_dir/tests/math_overload_control.cpp" \
    "$work_dir/control.o" ""
scan "$work_dir/control.o" > "$control_found"
if ! test -s "$control_found"; then
    printf 'positive control: no float math overload found in %s\n' \
        "$work_dir/control.o" >&2
    printf 'The symbol pattern does not match this toolchain.\n' >&2
    exit 1
fi
: > "$symbols"

for source in "$source_dir"/src/*.cpp; do
    for variant in default verbopt; do
        object="$work_dir/$(basename "$source" .cpp).$variant.o"
        define=
        if test "$variant" = verbopt; then
            define=-DVerbOpt
        fi
        compile "$source" "$object" "$define"
        scan "$object" >> "$found"
    done
done

if ! test -s "$symbols"; then
    printf 'nm listed no symbol in any object file\n' >&2
    exit 1
fi

if test -s "$found"; then
    printf 'math overloads for float or long double:\n' >&2
    LC_ALL=C sort "$found" >&2
    printf 'Cast the whole argument to double where C converts it.\n' >&2
    exit 1
fi
