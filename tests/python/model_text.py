# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compare serialized classifiers without their training date.

Each training run writes its local date into the `id` line of the model text,
so two models trained from the same inputs differ on that line when the runs
straddle midnight. Compare `without_date(model.serialized_data)` for models
that were trained separately. A model compared with its own load, pickle, or
round trip keeps its `id` line and may be compared as it is.
"""

from __future__ import annotations

import re

_ID_LINE = re.compile(r'^(id="[^"\n]*?) \d{4}-\d{2}-\d{2}"', re.ASCII)


def without_date(serialized: str) -> str:
    """Return a serialized classifier with the date in its id line removed.

    The date becomes the placeholder `<date>`. Every other character of the
    text is kept, so that a difference in the classifier is still a difference
    of the result.

    Raises:
        AssertionError: If the text does not start with an id line that ends in
            a date, which would hide the date from the comparison.
    """
    result, count = _ID_LINE.subn(r'\1 <date>"', serialized, count=1)
    assert count == 1, "serialized model lacks a dated id line"
    return result
