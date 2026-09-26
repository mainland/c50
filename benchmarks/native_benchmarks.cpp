/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <tuple>
#include <vector>

#include "c50_sort.h"

#include <catch2/catch_test_macros.hpp>

namespace
{

using record_key = std::tuple<C50SortValue, int, float>;

record_key key(const SortRec &record)
{
    return {record.V, record.C, record.W};
}

bool valid_sort(const std::vector<SortRec> &before,
                const std::vector<SortRec> &after)
{
    if ( before.size() != after.size() ) return false;

    for ( std::size_t index = 1; index < after.size(); ++index )
    {
        if ( after[index].V < after[index - 1].V ) return false;
    }

    std::vector<record_key> before_keys;
    std::vector<record_key> after_keys;
    before_keys.reserve(before.size());
    after_keys.reserve(after.size());
    std::transform(before.begin(), before.end(),
                   std::back_inserter(before_keys), key);
    std::transform(after.begin(), after.end(),
                   std::back_inserter(after_keys), key);
    std::sort(before_keys.begin(), before_keys.end());
    std::sort(after_keys.begin(), after_keys.end());
    return before_keys == after_keys;
}

void sort_records(std::vector<SortRec> &records)
{
    Cachesort(0, static_cast<int>(records.size()) - 1, records.data());
}

class deterministic_generator
{
public:
    explicit deterministic_generator(std::uint64_t seed) : state_(seed) {}

    std::uint64_t next()
    {
        state_ ^= state_ >> 12;
        state_ ^= state_ << 25;
        state_ ^= state_ >> 27;
        return state_ * UINT64_C(2685821657736338717);
    }

private:
    std::uint64_t state_;
};

std::vector<SortRec> random_records(std::size_t count,
                                    std::size_t distinct_values)
{
    deterministic_generator generator(UINT64_C(1729));
    std::vector<SortRec> records;
    records.reserve(count);
    for ( std::size_t index = 0; index < count; ++index )
    {
        const auto value = static_cast<C50SortValue>(
            static_cast<std::int64_t>(generator.next() % distinct_values) -
            static_cast<std::int64_t>(distinct_values / 2));
        records.push_back(
            {value, static_cast<int>(index % 3 + 1), 1.0F});
    }
    return records;
}

} // namespace

TEST_CASE("Cachesort preserves and orders records", "[sort][property]")
{
    constexpr std::array<C50SortValue, 3> alphabet = {-1.0F, 0.0F, 1.0F};
    std::size_t combinations = 1;

    for ( std::size_t length = 0; length <= 8; ++length )
    {
        for ( std::size_t encoded = 0; encoded < combinations; ++encoded )
        {
            std::size_t remaining = encoded;
            std::vector<SortRec> records;
            records.reserve(length);
            for ( std::size_t index = 0; index < length; ++index )
            {
                const C50SortValue value =
                    alphabet[remaining % alphabet.size()];
                remaining /= alphabet.size();
                records.push_back(
                    {value, static_cast<int>(index % 3 + 1),
                     static_cast<float>(index + 1)});
            }

            const std::vector<SortRec> original = records;
            sort_records(records);
            CAPTURE(length, encoded);
            REQUIRE(valid_sort(original, records));
        }
        combinations *= alphabet.size();
    }
}

TEST_CASE("Cachesort handles representative large inputs", "[sort][property]")
{
    for ( const std::size_t distinct_values : {8U, 1000U, 100000U} )
    {
        std::vector<SortRec> records =
            random_records(100000, distinct_values);
        const std::vector<SortRec> original = records;
        sort_records(records);
        CAPTURE(distinct_values);
        REQUIRE(valid_sort(original, records));
    }
}
