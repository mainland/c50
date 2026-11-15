/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "c50_index.h"

using namespace c50::core;

static int failures;

static void expect(const char *label, std::size_t actual, std::size_t expected)
{
    if (actual != expected) {
        std::fprintf(stderr, "%s: got %zu, expected %zu\n", label, actual,
                     expected);
        ++failures;
    }
}

int main()
{
    // Products below INT_MAX agree with the plain arithmetic.
    expect("origin", FlatIndex(0, 7, 0), 0);
    expect("column", FlatIndex(0, 7, 5), 5);
    expect("row", FlatIndex(3, 7, 0), 21);
    expect("small", FlatIndex(3, 7, 5), 26);

    if (sizeof(std::size_t) < 8) {
        std::puts("index-overflow: size_t is narrower than 64 bits, "
                  "boundary cases skipped");
        return failures == 0 ? 0 : 1;
    }

    // 46,341 classes: the confusion matrix has 46,342^2 cells, and its last
    // cell lies past INT_MAX.
    const std::uint64_t classes = 46341;
    const std::uint64_t stride = classes + 1;
    expect("last cell of 46,341 classes",
           FlatIndex(classes, stride, classes), classes * stride + classes);
    expect("past INT_MAX", FlatIndex(classes, stride, classes) > INT_MAX,
           true);

    // The first index past INT_MAX is not truncated.
    expect("INT_MAX", FlatIndex(INT_MAX, 1, 0), std::size_t{INT_MAX});
    expect("INT_MAX + 1", FlatIndex(INT_MAX, 1, 1), std::size_t{INT_MAX} + 1);
    expect("2 * INT_MAX", FlatIndex(INT_MAX, 2, 0),
           2 * std::size_t{INT_MAX});

    // 65,536 classes: the 32-bit product wrapped to a small index.
    expect("65,536 classes", FlatIndex(65534, 65537, 65536),
           std::size_t{65534} * 65537 + 65536);

    // Boost votes for 524,300 cases and 4,096 classes.
    expect("boost votes", FlatIndex(524299, 4097, 0),
           std::size_t{524299} * 4097);

    return failures == 0 ? 0 : 1;
}
