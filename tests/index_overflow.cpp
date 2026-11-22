/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>

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

// The expressions of the evaluation report before they were widened. They are
// defined only below the limit, where the product of 100 and the count fits in
// an int.
static double old_percent(int count, double total)
{
    return 100 * count / total;
}

static int old_usage_percent(int count, float total)
{
    return (int) ((100 * count) / total + 0.5);
}

static double old_utility_cut(int rules, int u, int bands)
{
    return (double) (rules * u / (float) bands);
}

static bool same_bits(double left, double right)
{
    return std::memcmp(&left, &right, sizeof left) == 0;
}

// Below the limit the widened arithmetic gives the same bits as the int
// expressions, so that printed percentages and utility bands do not change.
static void check_report_arithmetic_below_limit()
{
    std::uint64_t state = 88172645463325252ULL;
    const int limit = INT_MAX / 100;
    for (int trial = 0; trial < 200000; ++trial) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        const int total = 1 + static_cast<int>((state >> 8) % limit);
        const int count = static_cast<int>((state >> 33) %
                                           (static_cast<std::uint64_t>(total) + 1));
        if (!same_bits(CountPercent(count, total), old_percent(count, total)))
            expect("percent below the limit", 0, 1);
        if (UsagePercent(count, static_cast<float>(total)) !=
            old_usage_percent(count, static_cast<float>(total)))
            expect("usage percent below the limit", 0, 1);
    }
    for (int rules = 0; rules <= 400; ++rules) {
        for (int bands : {2, 3, 10, 7919, 10000}) {
            for (int u = 1; u < bands && u < 10000; u += 97) {
                if (static_cast<long long>(rules) * u > INT_MAX) continue;
                if (!same_bits(UtilityBandCut(rules, u, bands),
                               old_utility_cut(rules, u, bands)))
                    expect("utility cut below the limit", 0, 1);
            }
        }
    }
}

// Above 21,474,836 cases the product of 100 and a count passes INT_MAX. The
// percentage of every case, or of none, is exact, and so is one in between.
static void check_report_arithmetic_above_limit()
{
    const long long cases = 21474837;
    expect("all cases are errors", CountPercent(cases, cases) == 100.0, true);
    expect("no case is an error", CountPercent(0, cases) == 0.0, true);
    expect("half of the cases", CountPercent(cases / 2, 2 * (cases / 2)) ==
                                    50.0, true);
    expect("44,000,000 cases of 44,000,000",
           CountPercent(44000000, 44000000) == 100.0, true);
    expect("every case uses the attribute",
           static_cast<std::size_t>(
               UsagePercent(cases, static_cast<float>(cases))),
           100);
    expect("half of the cases use the attribute",
           static_cast<std::size_t>(
               UsagePercent(21474838, 42949676.0F)),
           50);
    // 214,749 rules in band 10,000 of 10,000 passes INT_MAX when multiplied.
    expect("utility cut past INT_MAX",
           UtilityBandCut(300000, 9999, 10000) > 2.9e5, true);
}

int main()
{
    check_report_arithmetic_below_limit();
    check_report_arithmetic_above_limit();

    // Products below INT_MAX agree with the plain arithmetic.
    expect("origin", FlatIndex(0, 7, 0), 0);
    expect("column", FlatIndex(0, 7, 5), 5);
    expect("row", FlatIndex(3, 7, 0), 21);
    expect("small", FlatIndex(3, 7, 5), 26);

    // The readers count cases and bytes in int and test before the addition.
    expect("grow from zero", GrowsWithinInt(0, 32768), true);
    expect("grow to INT_MAX", GrowsWithinInt(INT_MAX - 32768, 32768), true);
    expect("grow past INT_MAX", GrowsWithinInt(INT_MAX - 32767, 32768), false);
    expect("count at INT_MAX", GrowsWithinInt(INT_MAX, 1), false);
    expect("count below INT_MAX", GrowsWithinInt(INT_MAX - 1, 1), true);
    expect("case space at the last block",
           GrowsWithinInt(INT_MAX - 2049, 2049), true);
    expect("case space past the last block",
           GrowsWithinInt(INT_MAX - 2048, 2049), false);

    if constexpr (sizeof(std::size_t) < 8) {
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
