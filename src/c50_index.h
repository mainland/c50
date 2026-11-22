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

/* The arithmetic of the evaluation report in 64 bits.  A count of cases
   times 100, or a count of rules times a band number, passes INT_MAX for
   about 21 million cases or 214,000 rules, and the product in int then wraps.
   Each function widens before it multiplies and otherwise keeps the type and
   rounding of the int expression that it replaces, so that results below the
   limit are unchanged.  Counts are nonnegative.  */

/* The percentage that the report prints for a count out of Total cases. */
inline double CountPercent(long long Count, double Total)
{
    return 100.0 * Count / Total;
}

/* The whole percentage of Total cases that use an attribute, rounded.  Total
   is float, as in the report, so that the division is in float. */
inline int UsagePercent(long long Count, float Total)
{
    return static_cast<int>((100LL * Count) / Total + 0.5);
}

/* The rule count at which utility band u of Bands starts, before rounding. */
inline double UtilityBandCut(int Rules, int u, int Bands)
{
    return static_cast<double>(static_cast<long long>(Rules) * u /
                               static_cast<float>(Bands));
}

}  // namespace c50::core

#endif
