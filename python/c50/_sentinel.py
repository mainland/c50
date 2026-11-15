# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The marker for not-applicable values in estimator input."""

from __future__ import annotations

from typing import Final, final


@final
class _NotApplicableType:
    """The type of :data:`NOT_APPLICABLE`.

    The type has one instance. Calling it, copying, and unpickling return that
    instance, so identity tests are reliable.
    """

    __slots__ = ()

    # The class reports the public package, so that a pickle of the instance
    # names ``c50.NOT_APPLICABLE`` and not this private module.
    __module__ = "c50"

    _instance: _NotApplicableType | None = None

    def __new__(cls) -> _NotApplicableType:
        """Return the only instance."""
        if _NotApplicableType._instance is None:
            _NotApplicableType._instance = super().__new__(cls)
        return _NotApplicableType._instance

    def __repr__(self) -> str:
        """Return ``"c50.NOT_APPLICABLE"``."""
        return "c50.NOT_APPLICABLE"

    def __reduce__(self) -> str:
        """Pickle by reference to the public name."""
        return "NOT_APPLICABLE"

    def __copy__(self) -> _NotApplicableType:
        """Return the instance itself."""
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> _NotApplicableType:
        """Return the instance itself."""
        return self


NOT_APPLICABLE: Final = _NotApplicableType()
"""Marks a value of ``X`` that is not applicable to its case.

C5.0 distinguishes a missing value, which exists but is unknown, from a value
that does not apply to the case. The estimator passes a cell that holds this
object to C5.0 as ``N/A``. The object equals only itself, is not missing to
``pandas.isna``, and is not a number. See ``docs/estimator.md``.
"""
