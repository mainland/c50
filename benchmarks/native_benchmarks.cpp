/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <tuple>
#include <utility>
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

/* Reference copy of Cachesort from C5.0 Release 2.07 GPL Edition,
   Copyright 2010 Rulequest Research Pty Ltd. Its tie order defines the
   classifier-compatible result. */
void rulequest_cachesort(int Fp, int Lp, SortRec *SRec)
{
    while ( Fp < Lp )
    {
        const C50SortValue Thresh = SRec[(Fp + Lp) / 2].V;
        int Middle, High;

        for ( Middle = Fp; SRec[Middle].V < Thresh; Middle++ )
            ;
        for ( High = Lp; SRec[High].V > Thresh; High-- )
            ;
        for ( int i = Middle; i <= High; )
        {
            const C50SortValue Val = SRec[i].V;
            if ( Val < Thresh )
            {
                std::swap(SRec[Middle], SRec[i]);
                Middle++;
                i++;
            }
            else if ( Val > Thresh )
            {
                std::swap(SRec[High], SRec[i]);
                High--;
            }
            else
            {
                i++;
            }
        }

        rulequest_cachesort(Fp, Middle - 1, SRec);
        Fp = High + 1;
    }
}

bool same_order(const std::vector<SortRec> &left,
                const std::vector<SortRec> &right)
{
    return std::equal(left.begin(), left.end(), right.begin(), right.end(),
                      [](const SortRec &a, const SortRec &b)
                      {
                          return key(a) == key(b);
                      });
}

void require_rulequest_order(std::vector<SortRec> records)
{
    std::vector<SortRec> expected = records;
    rulequest_cachesort(0, static_cast<int>(expected.size()) - 1,
                        expected.data());
    sort_records(records);
    REQUIRE(same_order(records, expected));
}

} // namespace

TEST_CASE("Cachesort reproduces the RuleQuest tie order", "[sort][property]")
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
                records.push_back(
                    {alphabet[remaining % alphabet.size()],
                     static_cast<int>(index + 1), 1.0F});
                remaining /= alphabet.size();
            }
            CAPTURE(length, encoded);
            require_rulequest_order(records);
        }
        combinations *= alphabet.size();
    }

    for ( const std::size_t distinct_values : {2U, 8U, 1000U, 100000U} )
    {
        std::vector<SortRec> records =
            random_records(100000, distinct_values);
        for ( std::size_t index = 0; index < records.size(); ++index )
        {
            records[index].C = static_cast<int>(index + 1);
        }
        CAPTURE(distinct_values);
        require_rulequest_order(records);
    }
}

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

TEST_CASE("Cachesort handles finite floating-point extremes",
          "[sort][property]")
{
    const C50SortValue maximum =
        std::numeric_limits<C50SortValue>::max();
    const C50SortValue denormal =
        std::numeric_limits<C50SortValue>::denorm_min();
    const std::array<C50SortValue, 10> values = {
        -maximum, -maximum / 2, -1, -denormal, -0.0,
        0.0, denormal, 1, maximum / 2, maximum
    };
    deterministic_generator generator(UINT64_C(314159));

    for ( std::size_t repetition = 0; repetition < 100; ++repetition )
    {
        std::vector<SortRec> records;
        records.reserve(values.size());
        for ( std::size_t index = 0; index < values.size(); ++index )
        {
            records.push_back(
                {values[index], static_cast<int>(index + 1),
                 static_cast<float>(index + 1)});
        }
        for ( std::size_t index = records.size(); index > 1; --index )
        {
            const std::size_t swap_index = generator.next() % index;
            std::swap(records[index - 1], records[swap_index]);
        }

        const std::vector<SortRec> original = records;
        sort_records(records);
        CAPTURE(repetition);
        REQUIRE(valid_sort(original, records));
    }
}
