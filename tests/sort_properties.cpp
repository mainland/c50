/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "c50_sort.h"

using namespace c50::core;

namespace c50::core
{
/* Sort L[Fp..Lp] of case numbers, defined in rules.cpp. */
void ListSort(int *L, int Fp, int Lp);
}

namespace
{

void require(bool condition, const std::string &what)
{
    if ( !condition ) throw std::runtime_error(what);
}

std::string context(const char *name, std::size_t value)
{
    return std::string(name) + " " + std::to_string(value);
}

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

void require_rulequest_order(std::vector<SortRec> records,
                             const std::string &what)
{
    std::vector<SortRec> expected = records;
    rulequest_cachesort(0, static_cast<int>(expected.size()) - 1,
                        expected.data());
    sort_records(records);
    require(same_order(records, expected), "RuleQuest order, " + what);
}

void require_stable_order(std::vector<SortRec> records,
                          const std::string &what, int first = 0)
{
    const int last = static_cast<int>(records.size()) - 1;
    std::vector<SortRec> expected = records;
    if ( first <= last )
    {
        std::stable_sort(expected.begin() + first, expected.end(),
                         [](const SortRec &left, const SortRec &right)
                         {
                             return left.V < right.V;
                         });
    }

    std::vector<SortRec> record_scratch(records.size());
    std::vector<C50SortKey> keys(records.size());
    std::vector<C50SortKey> key_scratch(records.size());
    StableCachesort(first, last, records.data(), record_scratch.data(),
                    keys.data(), key_scratch.data());
    require(same_order(records, expected), "stable order, " + what);
}

std::vector<SortRec> numbered(std::vector<SortRec> records)
{
    for ( std::size_t index = 0; index < records.size(); ++index )
    {
        records[index].C = static_cast<int>(index + 1);
    }
    return records;
}

void stable_matches_std_stable_sort()
{
    constexpr std::array<C50SortValue, 3> alphabet = {-1.0F, 0.0F, 1.0F};
    std::size_t combinations = 1;

    for ( std::size_t length = 0; length <= 8; ++length )
    {
        for ( std::size_t encoded = 0; encoded < combinations; ++encoded )
        {
            std::size_t remaining = encoded;
            std::vector<SortRec> records;
            for ( std::size_t index = 0; index < length; ++index )
            {
                records.push_back(
                    {alphabet[remaining % alphabet.size()],
                     static_cast<int>(index + 1), 1.0F});
                remaining /= alphabet.size();
            }
            require_stable_order(
                records, context("length", length) + ", " +
                             context("encoding", encoded));
        }
        combinations *= alphabet.size();
    }

    // Lengths on both sides of the insertion-sort limit, and large ranges.
    for ( const std::size_t count : {47U, 48U, 49U, 50U, 1000U, 100000U} )
    {
        for ( const std::size_t distinct_values : {2U, 8U, 1000U, 100000U} )
        {
            require_stable_order(
                numbered(random_records(count, distinct_values)),
                context("count", count) + ", " +
                    context("distinct values", distinct_values));
        }
    }
}

void stable_orders_extremes()
{
    const C50SortValue maximum = std::numeric_limits<C50SortValue>::max();
    const C50SortValue denormal =
        std::numeric_limits<C50SortValue>::denorm_min();
    const std::array<C50SortValue, 10> values = {
        -maximum, -maximum / 2, -1, -denormal, -0.0,
        0.0, denormal, 1, maximum / 2, maximum
    };
    deterministic_generator generator(UINT64_C(271828));

    for ( const std::size_t count : {10U, 40U, 200U, 5000U} )
    {
        std::vector<SortRec> records;
        for ( std::size_t index = 0; index < count; ++index )
        {
            records.push_back(
                {values[generator.next() % values.size()], 0, 1.0F});
        }
        require_stable_order(numbered(records), context("count", count));
        require_stable_order(numbered(records), context("count", count),
                             static_cast<int>(count / 3));
    }
}

void reproduces_rulequest_order()
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
            require_rulequest_order(
                records, context("length", length) + ", " +
                             context("encoding", encoded));
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
        require_rulequest_order(records,
                                context("distinct values", distinct_values));
    }
}

void preserves_and_orders_records()
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
            require(valid_sort(original, records),
                    "sorted records, " + context("length", length) + ", " +
                        context("encoding", encoded));
        }
        combinations *= alphabet.size();
    }
}

void handles_large_inputs()
{
    for ( const std::size_t distinct_values : {8U, 1000U, 100000U} )
    {
        std::vector<SortRec> records =
            random_records(100000, distinct_values);
        const std::vector<SortRec> original = records;
        sort_records(records);
        require(valid_sort(original, records),
                "large input, " + context("distinct values", distinct_values));
    }
}

void handles_extremes()
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
        require(valid_sort(original, records),
                "extremes, " + context("repetition", repetition));
    }
}

/* Reference copy of ListSort from C5.0 Release 2.07 GPL Edition, Copyright
   2010 Rulequest Research Pty Ltd. Its result defines the list that Compress
   reads. */
void rulequest_listsort(int *L, int Fp, int Lp)
{
    if ( Fp < Lp )
    {
        const int Thresh = L[(Fp + Lp) / 2];
        int Middle, High;

        for ( Middle = Fp; L[Middle] < Thresh; Middle++ )
            ;
        for ( High = Lp; L[High] > Thresh; High-- )
            ;
        for ( int i = Middle; i <= High; )
        {
            if ( L[i] < Thresh )
            {
                std::swap(L[Middle], L[i]);
                Middle++;
                i++;
            }
            else if ( L[i] > Thresh )
            {
                std::swap(L[High], L[i]);
                High--;
            }
            else
            {
                i++;
            }
        }

        rulequest_listsort(L, Fp, Middle - 1);
        rulequest_listsort(L, High + 1, Lp);
    }
}

/* A list as ListSort receives it: element 0 holds the length. */
std::vector<int> list_of(const std::vector<int> &elements)
{
    std::vector<int> list;
    list.reserve(elements.size() + 1);
    list.push_back(static_cast<int>(elements.size()));
    list.insert(list.end(), elements.begin(), elements.end());
    return list;
}

void require_list_order(const std::vector<int> &elements,
                        const std::string &what)
{
    std::vector<int> expected = list_of(elements);
    std::vector<int> actual = expected;
    rulequest_listsort(expected.data(), 1, expected[0]);
    ListSort(actual.data(), 1, actual[0]);
    require(actual == expected, "ListSort order, " + what);
}

/* Every list of up to eight elements from three values, which includes
   equal elements, and large random lists, give the list of Release 2.07. */
void list_sort_reproduces_rulequest_order()
{
    std::size_t combinations = 1;
    for ( std::size_t length = 0; length <= 8; ++length )
    {
        for ( std::size_t encoded = 0; encoded < combinations; ++encoded )
        {
            std::size_t remaining = encoded;
            std::vector<int> elements;
            for ( std::size_t index = 0; index < length; ++index )
            {
                elements.push_back(static_cast<int>(remaining % 3));
                remaining /= 3;
            }
            require_list_order(elements,
                               context("length", length) + ", " +
                                   context("encoding", encoded));
        }
        combinations *= 3;
    }

    deterministic_generator generator(UINT64_C(577215));
    for ( const std::size_t distinct_values : {8U, 1000U, 1000000U} )
    {
        std::vector<int> elements;
        for ( std::size_t index = 0; index < 100000; ++index )
        {
            elements.push_back(
                static_cast<int>(generator.next() % distinct_values));
        }
        require_list_order(elements,
                           context("distinct values", distinct_values));
    }
}

/* After a condition is deleted, the list of covered cases holds the promoted
   cases in ascending order, followed by the original cases in descending
   order. When the two sets alternate, the middle element is the second
   largest at every level of the partition sort of Release 2.07, which
   recurses once per two elements and overflows the stack. */
void list_sort_handles_ascending_then_descending()
{
    constexpr int count = 400000;
    std::vector<int> elements;
    elements.reserve(count);
    for ( int index = 0; index < count / 2; ++index )
    {
        elements.push_back(2 * index);
    }
    for ( int index = count / 2 - 1; index >= 0; --index )
    {
        elements.push_back(2 * index + 1);
    }

    std::vector<int> list = list_of(elements);
    ListSort(list.data(), 1, list[0]);
    require(list[0] == count, "the length is untouched");
    // The elements are 0 to count - 1, so the sorted list counts upward.
    require(list[1] == 0 &&
                std::adjacent_find(list.begin() + 1, list.end(),
                                   [](int left, int right)
                                   {
                                       return right != left + 1;
                                   }) == list.end(),
            "the alternating list is sorted");
}

} // namespace

int main()
{
    try
    {
        stable_matches_std_stable_sort();
        stable_orders_extremes();
        reproduces_rulequest_order();
        preserves_and_orders_records();
        handles_large_inputs();
        handles_extremes();
        list_sort_reproduces_rulequest_order();
        list_sort_handles_ascending_then_descending();
    }
    catch ( const std::exception &error )
    {
        std::cerr << "sort property failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
