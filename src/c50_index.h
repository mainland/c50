/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef C50_INDEX_H
#define C50_INDEX_H

#include <cstddef>
#include <limits>

namespace c50::core {

/* Index of element (row, column) of a table stored row by row with the given
   row stride, computed in size_t.  The tables that the learner indexes this
   way (confusion matrices, boost votes, discrete frequencies) are allocated
   with size_t counts that are checked for overflow, so a size_t index stays in
   bounds where the same product in int wraps past INT_MAX.  Arguments are
   nonnegative counts or indices.  */
inline std::size_t FlatIndex(std::size_t row, std::size_t stride,
                             std::size_t column)
{
    return row * stride + column;
}

/* True when a counter that holds Size can grow by Step and still hold the
   result in an int.  The text readers count cases and bytes in int.  Callers
   check this before the addition, because the sum of two ints that overflows
   is undefined.  Both arguments are nonnegative.  */
inline bool GrowsWithinInt(int Size, int Step)
{
    return Step <= std::numeric_limits<int>::max() - Size;
}

}  // namespace c50::core

#endif
