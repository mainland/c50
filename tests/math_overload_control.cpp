/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

// Positive control for `check_math_overloads.sh`. It calls the `float`
// overload of `rint()`, which the check must report. If the check lists
// nothing for this file, its pattern no longer matches the symbols that the
// compiler and standard library emit, and the check would pass on any source.

#include <cmath>

float control_rint(float value)
{
    return std::rint(value);
}
