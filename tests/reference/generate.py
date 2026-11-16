#!/usr/bin/env python3
# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generate the datasets of the Release 2.07 comparison.

Each dataset is a stem with a `.names` file, a `.data` file, a `.test` file,
and optionally a `.costs` file. The generator uses only the standard library
and fixed seeds, so every run writes the same bytes on every platform.

usage: generate.py OUTPUT_DIRECTORY
"""

from __future__ import annotations

import functools
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Dataset:
    """Files of one generated dataset, keyed by extension."""

    stem: str
    files: dict[str, str]


def _value(rng: random.Random, number: float, missing: float) -> str:
    if rng.random() < missing:
        return "?"
    return f"{number:.2f}"


def _rows_mixed(rng: random.Random, count: int, weighted: bool) -> str:
    """Rows of a four-class dataset with every attribute kind."""
    groups = ["a", "b", "c", "d", "e"]
    sizes = ["small", "medium", "large", "huge"]
    colors = ["red", "green", "blue", "cyan", "pink", "gray", "gold", "teal"]
    lines = []
    for _ in range(count):
        x1 = rng.gauss(5.0, 2.5)
        x2 = rng.gauss(0.0, 1.0)
        x3 = rng.uniform(0.0, 10.0)
        group = rng.choice(groups)
        size = rng.choice(sizes)
        color = rng.choice(colors)
        score = x1 + 2.0 * x2 + sizes.index(size) + rng.gauss(0.0, 0.8)
        if group in ("a", "b"):
            score -= 3.0
        if color in ("red", "blue", "gold"):
            score += 2.0
        if score < 3.0:
            label = "low"
        elif score < 6.0:
            label = "mid"
        elif score < 9.0:
            label = "high"
        else:
            label = "top"
        # x2 does not apply to group "e".
        x2_field = "N/A" if group == "e" else _value(rng, x2, 0.06)
        fields = [
            _value(rng, x1, 0.05),
            x2_field,
            _value(rng, x3, 0.10),
            "?" if rng.random() < 0.05 else group,
            "?" if rng.random() < 0.05 else size,
            color,
            f"{rng.gauss(0.0, 1.0):.2f}",
            f"{rng.gauss(0.0, 1.0):.2f}",
            f"{rng.gauss(0.0, 1.0):.2f}",
            f"{rng.choice([0.5, 1.0, 1.5, 2.0, 3.0]):.1f}",
            label,
        ]
        if not weighted:
            del fields[9]
        lines.append(",".join(fields))
    return "\n".join(lines) + "\n"


_NAMES_UNWEIGHTED = """\
low, mid, high, top.

x1: continuous.
x2: continuous.
x3: continuous.
group: a, b, c, d, e.
size: [ordered] small, medium, large, huge.
color: red, green, blue, cyan, pink, gray, gold, teal.
noise1: continuous.
noise2: continuous.
noise3: continuous.
{weight}spread := x1 - x3.
big := x1 > 5.
"""

_NAMES_MIXED = _NAMES_UNWEIGHTED.format(weight="case weight: continuous.\n")
_NAMES_PLAIN = _NAMES_UNWEIGHTED.format(weight="")

_COSTS_MIXED = """\
mid, low: 2
high, low: 4
top, low: 6
low, mid: 3
high, mid: 2
top, mid: 3
low, high: 5
mid, high: 2
top, high: 2
low, top: 8
mid, top: 4
high, top: 2
"""


def _rows_binary(rng: random.Random, count: int) -> str:
    """Rows of a two-class dataset with noisy continuous attributes."""
    lines = []
    for _ in range(count):
        values = [rng.gauss(0.0, 1.0) for _ in range(6)]
        flag = rng.choice(["yes", "no"])
        score = (
            values[0]
            + 0.8 * values[1]
            - 0.5 * values[2]
            + (1.0 if flag == "yes" else -1.0)
            + rng.gauss(0.0, 1.2)
        )
        label = "pos" if score > 0.3 else "neg"
        fields = [_value(rng, v, 0.03) for v in values]
        fields.append(flag)
        fields.append(label)
        lines.append(",".join(fields))
    return "\n".join(lines) + "\n"


_NAMES_BINARY = """\
neg, pos.

v1: continuous.
v2: continuous.
v3: continuous.
v4: continuous.
v5: continuous.
v6: continuous.
flag: yes, no.
"""

_COSTS_BINARY = """\
pos, neg: 4
neg, pos: 1
"""


def _rows_discrete(rng: random.Random, count: int) -> str:
    """Rows of a three-class dataset with only discrete attributes."""
    lines = []
    for _ in range(count):
        a = rng.choice("pqrs")
        b = rng.choice(["one", "two", "three"])
        c = rng.choice(["u", "v", "w", "x", "y", "z"])
        score = "pqrs".index(a) + ["one", "two", "three"].index(b)
        score += 2 if c in ("u", "w") else 0
        score += rng.randrange(3)
        label = ["alpha", "beta", "gamma"][min(score // 3, 2)]
        fields = [
            "?" if rng.random() < 0.08 else a,
            "N/A" if b == "three" and rng.random() < 0.3 else b,
            "?" if rng.random() < 0.08 else c,
            label,
        ]
        lines.append(",".join(fields))
    return "\n".join(lines) + "\n"


_NAMES_DISCRETE = """\
alpha, beta, gamma.

first: p, q, r, s.
second: one, two, three.
third: u, v, w, x, y, z.
"""


def _noisy(rng: random.Random, label: str) -> str:
    """Flip a two-class label with probability 0.15."""
    if rng.random() < 0.15:
        return "high" if label == "low" else "low"
    return label


def _rows_divide_literal(rng: random.Random, count: int) -> str:
    """Rows for `x / 5000`, a six-decimal numerator over a wrapping divisor."""
    lines = []
    for _ in range(count):
        x = rng.uniform(0.1, 0.9)
        label = _noisy(rng, "high" if x > 0.5 else "low")
        lines.append(f"{x:.6f}, {label}")
    return "\n".join(lines) + "\n"


def _rows_divide_ratio(
    rng: random.Random, count: int, low: float, high: float
) -> str:
    """Rows for `a / b` with divisors `b` drawn from `[low, high]`."""
    lines = []
    for _ in range(count):
        a = rng.uniform(-5.0, 5.0)
        b = rng.uniform(low, high)
        label = _noisy(rng, "high" if a > 0.3 else "low")
        lines.append(f"{a:.6f}, {b:.6g}, {label}")
    return "\n".join(lines) + "\n"


def _rows_divide_integer(rng: random.Random, count: int) -> str:
    """Rows for `n / 2500000000`, an integer over a divisor above 1e9."""
    lines = []
    for _ in range(count):
        n = rng.randint(1, 10**6)
        lines.append(f"{n}, {_noisy(rng, 'high' if n > 400000 else 'low')}")
    return "\n".join(lines) + "\n"


def _rows_divide_nested(rng: random.Random, count: int) -> str:
    """Rows for a comparison whose operand divides by a zero-wrap divisor."""
    lines = []
    for _ in range(count):
        x = rng.uniform(-5.0, 5.0)
        y = rng.uniform(-5.0, 5.0)
        label = _noisy(rng, "high" if x > y else "low")
        lines.append(f"{x:.4f}, {y:.4f}, {label}")
    return "\n".join(lines) + "\n"


def _rows_divide_float(rng: random.Random, count: int) -> str:
    """Rows for `a / b` with numerators that need double precision.

    The numerators 0.498761 and 0.4988 straddle the cut. The rounding error of
    0.498761 / 3 must form in double precision to give the cut 0.1662537.
    """
    lines = []
    for index in range(count):
        if index % 4 < 2:
            a = 0.498761
        elif index % 4 == 2:
            a = 0.4988
        else:
            a = rng.choice([rng.uniform(0.0, 0.4), rng.uniform(0.6, 1.0)])
        label = "high" if a >= 0.4988 else "low"
        lines.append(f"{a:.6f}, 3, {label}")
    return "\n".join(lines) + "\n"


def _rows_prune_leaves(rng: random.Random) -> str:
    """Rows of 30 pure values of 22,203 cases each and one mixed value.

    A pure leaf of this size has an error estimate near the tie that decides
    whether pruning collapses the tree, and that estimate loses its digits when
    `N * (1 - exp(log(CF) / N))` runs in float.
    """
    lines = []
    for value in range(1, 31):
        lines.extend([f"v{value}, a"] * 22203)
    lines.extend(["r, a"] * 637)
    lines.extend(["r, b"] * 673)
    rng.shuffle(lines)
    return "\n".join(lines) + "\n"


def _rows_wide_names(rng: random.Random) -> str:
    """Rows of 100,001 cases whose subset test prints on two lines.

    The width of the case count in a printed subset test, 8 digits for 100,001,
    comes from a logarithm that must run in double precision.
    """
    lines = (
        ["x" * 30 + ", a"] * 50000
        + ["y" * 29 + ", a"] * 49001
        + ["zz, b"] * 500
        + ["ww, b"] * 500
    )
    rng.shuffle(lines)
    return "\n".join(lines) + "\n"


def _rows_far_timestamps(rng: random.Random, count: int) -> str:
    """Rows with timestamps in the years 3100 to 3300.

    A timestamp that far from the base year has a minute count above 5e8, which
    loses the minute in float arithmetic when it prints.
    """
    lines = []
    for _ in range(count):
        year = rng.randint(3100, 3299)
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        hour = rng.randint(0, 23)
        minute = rng.randint(0, 59)
        label = "hi" if (year, month) > (3200, 6) else "lo"
        if rng.random() < 0.05:
            label = "lo" if label == "hi" else "hi"
        lines.append(
            f"{year}/{month:02d}/{day:02d} {hour:02d}:{minute:02d}:00, {label}"
        )
    return "\n".join(lines) + "\n"


_NAMES_DIVIDE_LITERAL = """\
low, high.

x: continuous.
r := x / 5000.
attributes excluded: x.
"""

_NAMES_PRUNE_LEAVES = (
    "a, b.\n\nd: "
    + ", ".join([f"v{value}" for value in range(1, 31)] + ["r"])
    + ".\n"
)

_NAMES_WIDE = (
    "a, b.\n\nd: " + ", ".join(["x" * 30, "y" * 29, "zz", "ww"]) + ".\n"
)

_NAMES_TIMESTAMP = """\
lo, hi.

t: timestamp.
"""

_NAMES_DIVIDE_RATIO = """\
low, high.

a: continuous.
b: continuous.
r := a / b.
attributes excluded: a, b.
"""

_NAMES_DIVIDE_INTEGER = """\
low, high.

n: continuous.
r := n / 2500000000.
attributes excluded: n.
"""

_NAMES_DIVIDE_NESTED = """\
low, high.

x: continuous.
y: continuous.
s := x / 1e30 <= y.
"""


def datasets() -> list[Dataset]:
    """Return every dataset, generated from fixed seeds."""
    result = []
    # `mixed` has a case-weight attribute. `plain` holds the same kind of data
    # without one, because Release 2.07 winnows with case weights differently
    # from this project, as `docs/compatibility.md` documents.
    for stem, seed, names, weighted in [
        ("mixed", 20260101, _NAMES_MIXED, True),
        ("plain", 20260104, _NAMES_PLAIN, False),
    ]:
        rng = random.Random(seed)
        result.append(
            Dataset(
                stem,
                {
                    "names": names,
                    "data": _rows_mixed(rng, 2000, weighted),
                    "test": _rows_mixed(rng, 1000, weighted),
                    "costs": _COSTS_MIXED,
                },
            )
        )
    rng = random.Random(20260102)
    result.append(
        Dataset(
            "binary",
            {
                "names": _NAMES_BINARY,
                "data": _rows_binary(rng, 2500),
                "test": _rows_binary(rng, 1000),
                "costs": _COSTS_BINARY,
            },
        )
    )
    rng = random.Random(20260103)
    result.append(
        Dataset(
            "discrete",
            {
                "names": _NAMES_DISCRETE,
                "data": _rows_discrete(rng, 1500),
                "test": _rows_discrete(rng, 600),
            },
        )
    )
    # Bad values: a discrete value that the names file does not declare.
    # Release 2.07 continues past fewer than ten and exits 0, and stops at the
    # tenth with exit status 1.
    for stem, count in [("fewbad", 3), ("manybad", 12)]:
        rng = random.Random(20260103)
        lines = _rows_discrete(rng, 1500).split("\n")
        for index in range(count):
            fields = lines[index * 100 + 7].split(",")
            fields[0] = "zeta"
            lines[index * 100 + 7] = ",".join(fields)
        result.append(
            Dataset(
                stem,
                {
                    "names": _NAMES_DISCRETE,
                    "data": "\n".join(lines),
                    "test": _rows_discrete(random.Random(20260105), 600),
                },
            )
        )
    # Division in an implicit definition rounds with a precision multiplier
    # that wraps in Release 2.07's 32-bit int. `divide5000` wraps positive,
    # `divideneg` negative, `divideint` for an integer numerator, and
    # `dividemin` to the most negative value. `dividezero` wraps to 0, which
    # Release 2.07 turns into unknown values and stops at the error limit, and
    # `dividenested` propagates that value through a comparison.
    ratio = _rows_divide_ratio
    divide_cases: list[
        tuple[str, int, str, Callable[[random.Random, int], str]]
    ] = [
        ("divide5000", 20260111, _NAMES_DIVIDE_LITERAL, _rows_divide_literal),
        (
            "divideneg",
            20260112,
            _NAMES_DIVIDE_RATIO,
            functools.partial(ratio, low=1e5, high=1e6),
        ),
        (
            "dividemin",
            20260113,
            _NAMES_DIVIDE_RATIO,
            functools.partial(ratio, low=1.1e24, high=9.9e24),
        ),
        (
            "dividezero",
            20260114,
            _NAMES_DIVIDE_RATIO,
            functools.partial(ratio, low=1.1e25, high=9e26),
        ),
        ("divideint", 20260115, _NAMES_DIVIDE_INTEGER, _rows_divide_integer),
        ("dividenested", 20260116, _NAMES_DIVIDE_NESTED, _rows_divide_nested),
    ]
    for stem, seed, names, rows in divide_cases:
        rng = random.Random(seed)
        result.append(
            Dataset(
                stem,
                {
                    "names": names,
                    "data": rows(rng, 400),
                    "test": rows(rng, 200),
                },
            )
        )
    # Math expressions that Release 2.07 evaluates in double precision.
    # `dividefloat` rounds a quotient, `pruneleaves` estimates the errors of
    # leaves of 22,203 cases, `widenames` prints the width of a case count,
    # and `farstamps` prints timestamps that lie far from the base year.
    rng = random.Random(20260117)
    result.append(
        Dataset(
            "dividefloat",
            {
                "names": _NAMES_DIVIDE_RATIO,
                "data": _rows_divide_float(rng, 400),
                "test": _rows_divide_float(rng, 200),
            },
        )
    )
    rng = random.Random(20260118)
    result.append(
        Dataset(
            "pruneleaves",
            {
                "names": _NAMES_PRUNE_LEAVES,
                "data": _rows_prune_leaves(rng),
                "test": "r, a\nr, b\nv1, a\n",
            },
        )
    )
    rng = random.Random(20260119)
    result.append(
        Dataset(
            "widenames",
            {
                "names": _NAMES_WIDE,
                "data": _rows_wide_names(rng),
                "test": "zz, b\n",
            },
        )
    )
    rng = random.Random(20260123)
    result.append(
        Dataset(
            "farstamps",
            {
                "names": _NAMES_TIMESTAMP,
                "data": _rows_far_timestamps(rng, 2000),
                "test": _rows_far_timestamps(rng, 400),
            },
        )
    )
    return result


def write(directory: Path) -> None:
    """Write every dataset into `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    for dataset in datasets():
        for extension, text in dataset.files.items():
            path = directory / f"{dataset.stem}.{extension}"
            path.write_text(text, encoding="ascii", newline="\n")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    write(Path(argv[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
