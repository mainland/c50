/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <c50/c50.hpp>

namespace
{

struct scenario_digest
{
    std::size_t model_size;
    std::uint64_t model_hash;
    std::uint64_t prediction_hash;
};

// Captured from the C++ core with the imported RuleQuest sorting algorithms.
// The model hash omits the volatile invocation header but retains the complete
// classifier body.
constexpr std::array<scenario_digest, 32> reference_digests = {{
    {232, UINT64_C(0xf78cfba68a4a8610), UINT64_C(0x420e0c3eccf71a11)},
    {246, UINT64_C(0x5ae900cf3b39d6b9), UINT64_C(0x384bb9b1f5ec8268)},
    {234, UINT64_C(0x72d3b91942e6eed6), UINT64_C(0x8f7e143616aa533d)},
    {242, UINT64_C(0x2b11c9fda6ec772e), UINT64_C(0x814d7d6b0bcd89c0)},
    {952, UINT64_C(0xb0cf9a042229048f), UINT64_C(0x41787f571d788e51)},
    {40, UINT64_C(0xad2da3796d78788f), UINT64_C(0x449d0908703c99fa)},
    {530, UINT64_C(0x5b0d64357de86b79), UINT64_C(0x28c0b7a22a431207)},
    {387, UINT64_C(0x109a07eb5b4e1cc0), UINT64_C(0xddc03da7471608c1)},
    {1688, UINT64_C(0x7dc1147ed41885dc), UINT64_C(0xa5b34e3e2588be67)},
    {40, UINT64_C(0xa491d079689c21b8), UINT64_C(0x36f74700d110d5bf)},
    {253, UINT64_C(0x6f8caef0b189c935), UINT64_C(0x91d207d5ab5860eb)},
    {40, UINT64_C(0xad2da3796d78788f), UINT64_C(0xecac29d909d98664)},
    {262, UINT64_C(0x8af257b4e67ea574), UINT64_C(0x8bc25b2727b5e051)},
    {40, UINT64_C(0xa491d079689c21b8), UINT64_C(0x5026c4fe1f0e9599)},
    {201, UINT64_C(0xb4a4acb1bb3dd225), UINT64_C(0xbf5ff748444cdf56)},
    {141, UINT64_C(0xea66d2e82cdc9488), UINT64_C(0xe6782819554b3847)},
    {781, UINT64_C(0xa83558e90fc7d152), UINT64_C(0x313184f9a6e9c2c7)},
    {244, UINT64_C(0x4b80b8522e60fbdd), UINT64_C(0xd508b70aa577c905)},
    {76, UINT64_C(0x598cf02f206b6551), UINT64_C(0x14878899e4cdf6ff)},
    {40, UINT64_C(0xa491d079689c21b8), UINT64_C(0xf3705f0df0254d8f)},
    {964, UINT64_C(0x6fc08c660ed22588), UINT64_C(0xde64e560d4d408ac)},
    {479, UINT64_C(0x3899aa43b5e19e8f), UINT64_C(0x61079969a800a1f1)},
    {744, UINT64_C(0xa72f50733a39b7b2), UINT64_C(0x22548560ef7e2310)},
    {244, UINT64_C(0x2cbd6465ab0786e8), UINT64_C(0xa0f3bfc74a139d76)},
    {235, UINT64_C(0xc805bbff876f8d36), UINT64_C(0x6c989840177cb2c0)},
    {149, UINT64_C(0xaa99ab2f6e993757), UINT64_C(0x4f3f22d8c824500b)},
    {50, UINT64_C(0xb2fb172e79ff80e9), UINT64_C(0x3f849b0147bea2e1)},
    {247, UINT64_C(0xad35b235dfb9078b), UINT64_C(0xb2c8dcb0aba097ce)},
    {382, UINT64_C(0xf94e3da51a394ae2), UINT64_C(0x33cc875d710a1046)},
    {726, UINT64_C(0x567e6451c4444bcb), UINT64_C(0x9d4749a48caf77f5)},
    {216, UINT64_C(0x65728dae74eb8cf0), UINT64_C(0xeb66eb2aa135b443)},
    {251, UINT64_C(0x8897613aab8167ae), UINT64_C(0x9cae6cb9cb0bca4b)},
}};

class stable_hasher
{
public:
    void append_byte(unsigned char value)
    {
        state_ ^= value;
        state_ *= UINT64_C(1099511628211);
    }

    void append_bytes(const char *data, std::size_t size)
    {
        for ( std::size_t index = 0; index < size; ++index )
        {
            append_byte(static_cast<unsigned char>(data[index]));
        }
    }

    void append_uint64(std::uint64_t value)
    {
        for ( unsigned int shift = 0; shift < 64; shift += 8 )
        {
            append_byte(static_cast<unsigned char>(value >> shift));
        }
    }

    void append_double(double value)
    {
        std::uint64_t bits;
        static_assert(sizeof(bits) == sizeof(value));
        static_assert(std::numeric_limits<double>::is_iec559);
        std::memcpy(&bits, &value, sizeof(bits));
        append_uint64(bits);
    }

    std::uint64_t value() const { return state_; }

private:
    std::uint64_t state_ = UINT64_C(14695981039346656037);
};

class random_generator
{
public:
    explicit random_generator(std::uint32_t seed) : state_(seed) {}

    std::uint32_t next()
    {
        state_ = state_ * 1664525U + 1013904223U;
        return state_;
    }

private:
    std::uint32_t state_;
};

bool same_predictions(const c50::predictions &left,
                      const c50::predictions &right)
{
    const std::size_t row_count = left.size();
    const std::size_t class_count = left.class_count();

    if ( row_count != right.size() ||
         class_count != right.class_count() )
    {
        std::cerr << "prediction dimensions " << row_count << 'x'
                  << class_count << " != "
                  << right.size() << 'x'
                  << right.class_count() << '\n';
        return false;
    }

    for ( std::size_t class_index = 0; class_index < class_count;
          ++class_index )
    {
        if ( std::string(left.class_name(class_index)) !=
             right.class_name(class_index) )
        {
            std::cerr << "class name " << class_index << ": "
                      << left.class_name(class_index)
                      << " != "
                      << right.class_name(class_index)
                      << '\n';
            return false;
        }
    }

    for ( std::size_t row = 0; row < row_count; ++row )
    {
        if ( left.class_index(row) !=
                 right.class_index(row) ||
             left.confidence(row) !=
                 right.confidence(row) )
        {
            std::cerr << "row " << row << ": class/confidence "
                      << left.class_index(row) << '/'
                      << left.confidence(row) << " != "
                      << right.class_index(row) << '/'
                      << right.confidence(row) << '\n';
            return false;
        }
        for ( std::size_t class_index = 0; class_index < class_count;
              ++class_index )
        {
            if ( left.score(row, class_index) !=
                 right.score(row, class_index) )
            {
                std::cerr << "row " << row << ", class " << class_index
                          << ": score "
                          << left.score(row, class_index)
                          << " != "
                          << right.score(row, class_index)
                          << '\n';
                return false;
            }
        }
    }
    return true;
}

std::string stable_model_data(const c50::model &model)
{
    const std::string serialized = model.serialized_data();
    const std::string::size_type newline = serialized.find('\n');

    return newline == std::string::npos ? serialized
                                        : serialized.substr(newline + 1);
}

std::uint64_t stable_model_hash(const std::string &model_data)
{
    stable_hasher hash;
    hash.append_bytes(model_data.data(), model_data.size());
    return hash.value();
}

std::uint64_t stable_prediction_hash(const c50::predictions &predictions)
{
    stable_hasher hash;
    const std::size_t row_count = predictions.size();
    const std::size_t class_count = predictions.class_count();

    hash.append_uint64(row_count);
    hash.append_uint64(class_count);
    for ( std::size_t class_index = 0; class_index < class_count;
          ++class_index )
    {
        const char *name =
            predictions.class_name(class_index).c_str();
        const std::size_t size = std::strlen(name);
        hash.append_uint64(size);
        hash.append_bytes(name, size);
    }
    for ( std::size_t row = 0; row < row_count; ++row )
    {
        hash.append_uint64(predictions.class_index(row));
        hash.append_double(predictions.confidence(row));
        for ( std::size_t class_index = 0; class_index < class_count;
              ++class_index )
        {
            hash.append_double(
                predictions.score(row, class_index));
        }
    }
    return hash.value();
}

void append_value(std::ostringstream &output, double value,
                  const char *const *categories)
{
    if ( std::isnan(value) )
    {
        output << '?';
    }
    else if ( categories )
    {
        output << categories[static_cast<std::size_t>(value)];
    }
    else
    {
        output << value;
    }
}

bool report_failure(c50::context &context, std::size_t scenario,
                    const char *operation)
{
    std::cerr << "scenario " << scenario << ": " << operation;
    (void) context;
    std::cerr << '\n';
    return false;
}

bool matches_reference(std::size_t scenario,
                       const scenario_digest &actual)
{
    const scenario_digest &expected = reference_digests[scenario];
    if ( actual.model_size == expected.model_size &&
         actual.model_hash == expected.model_hash &&
         actual.prediction_hash == expected.prediction_hash )
    {
        return true;
    }

    std::cerr << "scenario " << scenario << ": reference digest mismatch\n"
              << "  expected: model size " << expected.model_size
              << ", model hash 0x" << std::hex << std::setw(16)
              << std::setfill('0') << expected.model_hash
              << ", prediction hash 0x" << std::setw(16)
              << expected.prediction_hash << '\n'
              << "  actual:   model size " << std::dec << actual.model_size
              << ", model hash 0x" << std::hex << std::setw(16)
              << actual.model_hash << ", prediction hash 0x"
              << std::setw(16) << actual.prediction_hash << std::dec << '\n';
    return false;
}

bool run_scenario(c50::context &context, std::size_t scenario,
                  scenario_digest &digest)
{
    static const char names[] =
        "class_0, class_1, class_2.\n\n"
        "feature_0: continuous.\n"
        "feature_1: continuous.\n"
        "feature_2: value_0, value_1, value_2.\n";
    static const char *const categories[] = {
        "value_0", "value_1", "value_2"
    };
    static const char *const classes[] = {
        "class_0", "class_1", "class_2"
    };
    static const char costs_data[] = "class_0, class_1: 3\n";
    const std::size_t row_count = 18 + scenario % 13;
    const std::size_t feature_count = 3;
    const bool use_costs = scenario % 4 == 0;
    const std::string costs = use_costs ? costs_data : "";
    random_generator random(static_cast<std::uint32_t>(scenario + 1));
    std::vector<double> values(row_count * feature_count);
    std::vector<std::size_t> class_indices(row_count);
    std::ostringstream training;
    std::ostringstream cases;
    c50::options options;

    training << std::setprecision(17);
    cases << std::setprecision(17);
    for ( std::size_t row = 0; row < row_count; ++row )
    {
        double *record = &values[row * feature_count];
        record[0] = random.next() % 11 == 0
                        ? std::numeric_limits<double>::quiet_NaN()
                        : static_cast<int>(random.next() % 17) / 2.0 - 4;
        record[1] = random.next() % 13 == 0
                        ? std::numeric_limits<double>::quiet_NaN()
                        : static_cast<int>(random.next() % 21) / 2.0 - 5;
        record[2] = random.next() % 17 == 0
                        ? std::numeric_limits<double>::quiet_NaN()
                        : random.next() % 3;

        std::size_t class_index;
        if ( row < 3 )
        {
            class_index = row;
        }
        else
        {
            const std::size_t signal =
                (!std::isnan(record[0]) && record[0] > 0 ? 1 : 0) +
                (!std::isnan(record[2])
                     ? static_cast<std::size_t>(record[2])
                     : 0);
            class_index = (row + signal) % 3;
        }
        class_indices[row] = class_index;

        for ( std::size_t feature = 0; feature < feature_count; ++feature )
        {
            if ( feature )
            {
                training << ", ";
                cases << ", ";
            }
            const char *const *category_names =
                feature == 2 ? categories : NULL;
            append_value(training, record[feature], category_names);
            append_value(cases, record[feature], category_names);
        }
        training << ", " << classes[class_index] << '\n';
        cases << ", ?\n";
    }

    const c50::dense_dataset training_dataset(
        values.data(), row_count, feature_count, class_indices.data());
    const c50::dense_dataset case_dataset(values.data(), row_count, feature_count);

    options.trials = 1 + scenario % 3;
    options.subset_splits = scenario % 2;
    options.winnow = scenario % 5 == 0;
    options.global_pruning = scenario % 3 != 0;
    options.probabilistic_thresholds = scenario % 4 == 1;
    options.minimum_cases = 1 + scenario % 3;
    options.confidence_factor = scenario % 3 == 0 ? 0.05 : (scenario % 3 == 1 ? 0.25 : 0.5);
    if ( scenario % 6 == 0 )
    {
        options.sample_fraction = 0.75;
        options.random_seed = (scenario * 97) % 4096;
    }

    const std::string training_data = training.str();
    const std::string case_data = cases.str();
    const auto kind = scenario % 2 ? c50::model_kind::rules : c50::model_kind::tree;
    auto text_model = c50::model::train(context, kind, names, training_data,
                                      options, costs);
    auto dense_model = c50::model::train(context, kind, names, training_dataset,
                                       options, costs);
    if ( stable_model_data(text_model) != stable_model_data(dense_model) )
    {
        return report_failure(context, scenario, "serialized model mismatch");
    }
    auto text_predictions = text_model.predict(context, case_data);
    auto dense_predictions = dense_model.predict(context, case_dataset);
    if ( ! same_predictions(text_predictions, dense_predictions) )
    {
        return report_failure(context, scenario, "prediction mismatch");
    }
    const std::string stable_model = stable_model_data(text_model);
    digest = {stable_model.size(), stable_model_hash(stable_model),
              stable_prediction_hash(text_predictions)};
    auto loaded_model = c50::model::load(context, kind, names,
                                        text_model.serialized_data(), costs);
    auto loaded_predictions = loaded_model.predict(context, case_dataset);
    return same_predictions(text_predictions, loaded_predictions) ||
           report_failure(context, scenario, "reloaded prediction mismatch");
}

} // namespace

int main(int argc, char **argv)
{
    const bool emit_reference =
        argc == 2 && std::string(argv[1]) == "--emit-reference";
    if ( argc != 1 && ! emit_reference )
    {
        std::cerr << "usage: c50_reference_compatibility [--emit-reference]\n";
        return 2;
    }

    c50::context context;

    for ( std::size_t scenario = 0; scenario < 32; ++scenario )
    {
        scenario_digest digest;
        if ( ! run_scenario(context, scenario, digest) ) return 1;
        if ( emit_reference )
        {
            std::cout << "    {" << digest.model_size << ", UINT64_C(0x"
                      << std::hex << std::setw(16) << std::setfill('0')
                      << digest.model_hash << "), UINT64_C(0x"
                      << std::setw(16) << digest.prediction_hash
                      << ")}, // scenario " << std::dec << scenario << '\n';
        }
        else if ( ! matches_reference(scenario, digest) )
        {
            return 1;
        }
    }
    return 0;
}
