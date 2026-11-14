# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Every count argument of both Python layers, run through the same values.

The low-level layer follows the rule in the ``c50`` package docstring. One
table drives every argument, so that an argument that converts differently
fails here.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import c50
import numpy as np
import pytest
from c50.sklearn import C50Classifier

NAMES = "low, high.\n\nsignal: continuous.\n"
TRAINING = "0, low\n1, low\n2, low\n3, high\n4, high\n5, high\n"
X = np.arange(12, dtype=np.float64).reshape(6, 2)
Y = np.asarray(["low", "low", "low", "high", "high", "high"])

# A tree of one level, so that every depth limit above zero accepts it.
ONE_LEVEL_NAMES = "no, yes.\nx: continuous.\n"
ONE_LEVEL_TREE = (
    'id="See5/C5.0 2.07 GPL Edition"\nentries="1"\n'
    'type="0" class="yes" freq="0,2"\n'
)
SNAPSHOT = c50.train(NAMES, TRAINING).inspect()
FITTED = C50Classifier().fit(X, Y)


@dataclass(frozen=True)
class CountArgument:
    """One count argument and the integers that its entry point rejects."""

    name: str
    # Call the entry point with a value and return its result.
    call: Callable[[Any], object]
    # Integers that raise ValueError, and integers other than 1 that work.
    invalid: tuple[int, ...]
    valid: tuple[int, ...]
    # Whether None is a valid value.
    none_ok: bool
    # Matches the message of a ValueError that names the argument.
    message: str


def train_with_options(**fields: Any) -> object:
    options = c50.Options()
    for name, value in fields.items():
        setattr(options, name, value)
    return c50.train(NAMES, TRAINING, options=options).serialized_data


LOW_LEVEL = [
    CountArgument(
        "split_workers",
        lambda value: (
            c50.train(NAMES, TRAINING, split_workers=value).serialized_data
        ),
        invalid=(0, -1, 9, 2**32, 2**70),
        valid=(8,),
        none_ok=False,
        message="split.worker",
    ),
    CountArgument(
        "max_tree_depth",
        lambda value: (
            c50.load(
                ONE_LEVEL_NAMES, ONE_LEVEL_TREE, max_tree_depth=value
            ).serialized_data
        ),
        invalid=(0, -1, 2**64, 2**70),
        valid=(1024,),
        none_ok=True,
        message="max_tree_depth",
    ),
    CountArgument(
        "Options.trials",
        lambda value: train_with_options(trials=value),
        invalid=(0, 1001, -1, 2**32, 2**70),
        valid=(2,),
        none_ok=False,
        message="trials",
    ),
    CountArgument(
        "Options.random_seed",
        lambda value: train_with_options(random_seed=value),
        invalid=(4096, -1, 2**32, 2**70),
        valid=(0, 4095),
        none_ok=False,
        message="seed",
    ),
    CountArgument(
        "c50.export_text",
        lambda value: c50.export_text(SNAPSHOT, max_depth=value),
        invalid=(-1,),
        valid=(0, 2**70),
        none_ok=True,
        message="max_depth",
    ),
    CountArgument(
        "C50Classifier.export_text",
        lambda value: FITTED.export_text(max_depth=value),
        invalid=(-1,),
        valid=(0, 2**70),
        none_ok=True,
        message="max_depth",
    ),
]


class IndexOnly:
    """An object that converts to an integer through __index__ alone."""

    def __init__(self, value: int = 1) -> None:
        self.value = value

    def __index__(self) -> int:
        return self.value


class HostileClass:
    """An object whose type check raises, as ``isinstance`` reports it."""

    @property
    def __class__(self) -> type:  # type: ignore[override]
        raise RuntimeError("hostile __class__")

    def __index__(self) -> int:
        return 1


def numpy_bool_subclass() -> list[object]:
    """Return an instance of a numpy.bool_ subclass, where one can exist."""
    try:
        subclass = type("Flag", (np.bool_,), {})
    except TypeError:
        return []
    return [subclass(True)]


INTEGER_TYPES = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]


def integer_values(argument: CountArgument) -> list[tuple[int, object]]:
    """Return each integer of the argument as a Python int and NumPy ints."""
    integers = (1, *argument.valid, *argument.invalid)
    values: list[tuple[int, object]] = []
    for integer in integers:
        values.append((integer, integer))
        for kind in INTEGER_TYPES:
            info = np.iinfo(kind)
            if info.min <= integer <= info.max:
                values.append((integer, kind(integer)))
    return values


def integer_cases(
    arguments: list[CountArgument],
) -> list[Any]:
    cases = []
    for argument in arguments:
        invalid = set(argument.invalid)
        for integer, value in integer_values(argument):
            label = f"{argument.name}-{type(value).__name__}-{integer}"
            cases.append(
                pytest.param(
                    argument, integer, value, integer in invalid, id=label
                )
            )
    return cases


@pytest.mark.parametrize(
    ("argument", "integer", "value", "rejected"),
    integer_cases(LOW_LEVEL),
)
def test_low_level_integers(
    argument: CountArgument, integer: int, value: object, rejected: bool
) -> None:
    if rejected:
        with pytest.raises(ValueError, match=argument.message):
            argument.call(value)
    else:
        assert argument.call(value) == argument.call(integer)


def not_integers() -> list[tuple[str, Callable[[], object]]]:
    values: list[tuple[str, Callable[[], object]]] = [
        ("True", lambda: True),
        ("np.True_", lambda: np.True_),
        ("np.False_", lambda: np.False_),
        ("0d-bool-array", lambda: np.asarray(True)),
        ("float", lambda: 1.0),
        ("str", lambda: "1"),
    ]
    for index, flag in enumerate(numpy_bool_subclass()):
        values.append((f"bool_subclass-{index}", lambda flag=flag: flag))
    return values


NOT_INTEGERS = not_integers()


@pytest.mark.parametrize("argument", LOW_LEVEL, ids=lambda a: a.name)
@pytest.mark.parametrize(
    "make",
    [make for _, make in NOT_INTEGERS],
    ids=[n for n, _ in NOT_INTEGERS],
)
def test_low_level_rejects_non_integers_with_type_error(
    argument: CountArgument, make: Callable[[], object]
) -> None:
    with pytest.raises(TypeError):
        argument.call(make())


@pytest.mark.parametrize("argument", LOW_LEVEL, ids=lambda a: a.name)
@pytest.mark.parametrize(
    "make",
    [lambda: IndexOnly(1), lambda: np.asarray(1)],
    ids=["index-only", "0d-int-array"],
)
def test_low_level_accepts_any_index_object(
    argument: CountArgument, make: Callable[[], object]
) -> None:
    assert argument.call(make()) == argument.call(1)


@pytest.mark.parametrize("argument", LOW_LEVEL, ids=lambda a: a.name)
def test_none(argument: CountArgument) -> None:
    if argument.none_ok:
        assert argument.call(None) is not None
    else:
        with pytest.raises(TypeError):
            argument.call(None)


@pytest.mark.parametrize("argument", LOW_LEVEL, ids=lambda a: a.name)
def test_a_failing_type_check_propagates(argument: CountArgument) -> None:
    # The error of the check is not a reason to accept or to replace the value.
    with pytest.raises(RuntimeError, match="hostile __class__"):
        argument.call(HostileClass())


# A NumPy that does not load must not make an ordinary count argument fail.
# These dummy modules stand for a lazily imported NumPy whose import fails, in
# the two ways that such a module can fail when an attribute is read.
BROKEN_NUMPY_SCRIPT = """
import importlib.util
import sys
import types

mode = sys.argv[1]
if mode == "lazy":
    # A lazy loader runs the failing import on the first attribute read.
    import pathlib, tempfile
    root = pathlib.Path(tempfile.mkdtemp())
    (root / "numpy").mkdir()
    (root / "numpy" / "__init__.py").write_text(
        "raise ImportError('numpy failed to load')\\n"
    )
    sys.path.insert(0, str(root))
    spec = importlib.util.find_spec("numpy")
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["numpy"] = module
    loader.exec_module(module)
    sys.path.remove(str(root))
else:
    module = types.ModuleType("numpy")

    def fail(name):
        raise ImportError("numpy failed to load")

    module.__getattr__ = fail
    sys.modules["numpy"] = module

import c50

names = %(names)r
data = %(data)r
c50.train(names, data)
c50.train(names, data, split_workers=2)
options = c50.Options()
options.trials = 3
assert options.trials == 3
c50.train(names, data, options=options)
print("ok")
"""

REAL_NUMPY_SCRIPT = """
import c50
import numpy as np

names = %(names)r
data = %(data)r
for flag in (np.True_, np.False_):
    try:
        c50.train(names, data, split_workers=flag)
    except TypeError:
        pass
    else:
        raise SystemExit("split_workers accepted a NumPy boolean")
    options = c50.Options()
    try:
        options.trials = flag
    except TypeError:
        pass
    else:
        raise SystemExit("trials accepted a NumPy boolean")
print("ok")
"""


def run_script(script: str, *arguments: str) -> str:
    source = script % {"names": NAMES, "data": TRAINING}
    result = subprocess.run(
        [sys.executable, "-c", source, *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.mark.parametrize("mode", ["lazy", "getattr"])
def test_a_numpy_that_fails_to_load_does_not_affect_integers(
    mode: str,
) -> None:
    assert run_script(BROKEN_NUMPY_SCRIPT, mode) == "ok"


def test_a_numpy_boolean_is_rejected_when_numpy_is_loaded() -> None:
    assert run_script(REAL_NUMPY_SCRIPT) == "ok"
