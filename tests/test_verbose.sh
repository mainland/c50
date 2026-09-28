#!/bin/sh

set -eu

script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)
binary=${C50_BINARY:-"$script_dir/../c5.0"}
test_dir=$(mktemp -d "${TMPDIR:-/tmp}/c50-verbose-test.XXXXXX")
trap 'rm -rf "$test_dir"' EXIT HUP INT TERM

cp "$script_dir/fixtures/basic/basic.names" "$test_dir/"
cp "$script_dir/fixtures/basic/basic.data" "$test_dir/"
(
    cd "$test_dir"
    "$binary" -f basic -s -v 2 -I 17 > output.raw
)

sed -n '/^<0> 20 cases$/,/best attribute group/p' \
    "$test_dir/output.raw" |
    sed -e 's/^[[:space:]]*//' \
        -e 's/[[:space:]][[:space:]]*/ /g' \
    > "$test_dir/root.actual"

diff -u - "$test_dir/root.actual" <<'EXPECTED'
<0> 20 cases
Att signal cut=4.750, inf 1.722, gain 0.258
Att group initial inf 1.559, gain 0.346, val=0.222
form subset alpha, beta: 2 subsets, inf 0.934, gain 0.346, val 0.286 **
final inf 0.934, gain 0.267, val=0.285
Att enabled inf 1.000, gain 0.000
av gain=0.263, MDL (2) = 0.050, min=0.263
best attribute group inf 0.934 gain 0.267 val 0.286
EXPECTED

sed -e 's|^id="See5/C5\.0 2\.07 GPL Edition [0-9-][0-9-]*"$|id="See5/C5.0 2.07 GPL Edition <date>"|' \
    "$test_dir/basic.tree" > "$test_dir/model.actual"
diff -u "$script_dir/expected/basic/subsets.tree" "$test_dir/model.actual"

(
    cd "$test_dir"
    "$binary" -f basic -I 17 -u 2 > utility.raw
    cp basic.rules utility.rules
    "$binary" -f basic -I 17 -u 2 -v 2 > pruning.raw
)
sed -e 's|^id="See5/C5\.0 2\.07 GPL Edition [0-9-][0-9-]*"$|id="See5/C5.0 2.07 GPL Edition <date>"|' \
    "$test_dir/basic.rules" > "$test_dir/pruned-model.actual"
sed -e 's|^id="See5/C5\.0 2\.07 GPL Edition [0-9-][0-9-]*"$|id="See5/C5.0 2.07 GPL Edition <date>"|' \
    "$test_dir/utility.rules" > "$test_dir/utility-model.actual"
diff -u "$test_dir/utility-model.actual" "$test_dir/pruned-model.actual"

printf 'Verbose CLI regression test passed\n'
