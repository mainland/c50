/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <c50/c50.hpp>

namespace
{

struct configuration
{
    std::size_t rows = 100000;
    std::size_t features = 50;
    std::size_t categorical_features = 5;
    std::size_t categories = 8;
    std::uint64_t seed = 1729;
    std::size_t split_workers = 1;
    bool subsets = false;
    bool stable_ties = false;
    std::size_t value_levels = 0;
};

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

    double unit_interval()
    {
        return static_cast<double>(next() >> 11) * 0x1.0p-53;
    }

private:
    std::uint64_t state_;
};

std::size_t parse_size(std::string_view option, std::string_view value)
{
    std::size_t parsed = 0;
    const auto result =
        std::from_chars(value.data(), value.data() + value.size(), parsed);
    if ( result.ec != std::errc{} || result.ptr != value.data() + value.size() )
    {
        throw std::invalid_argument(std::string(option) +
                                    " requires a nonnegative integer");
    }
    return parsed;
}

configuration parse_arguments(int argc, char **argv)
{
    configuration config;
    for ( int index = 1; index < argc; ++index )
    {
        const std::string_view option(argv[index]);
        if ( option == "--help" )
        {
            std::cout
                << "Usage: parallel-workload [OPTIONS]\n"
                << "  --rows N                  Training rows (default 100000)\n"
                << "  --features N              Feature columns (default 50)\n"
                << "  --categorical-features N  Trailing categorical columns "
                   "(default 5)\n"
                << "  --categories N            Values per categorical column "
                   "(default 8)\n"
                << "  --seed N                  Deterministic generator seed "
                   "(default 1729)\n"
                << "  --workers N               Split workers (1 to 8, default 1)\n"
                << "  --subsets                 Enable subset splits\n"
                << "  --ties ORDER              Tie order: reference or stable "
                   "(default reference)\n"
                << "  --value-levels N          Round continuous values to N "
                   "levels (0 disables, default 0)\n";
            std::exit(0);
        }
        if ( option == "--subsets" )
        {
            config.subsets = true;
            continue;
        }
        if ( index + 1 >= argc )
        {
            throw std::invalid_argument(std::string(option) +
                                        " requires a value");
        }
        const std::string_view value(argv[++index]);
        if ( option == "--rows" )
        {
            config.rows = parse_size(option, value);
        }
        else if ( option == "--features" )
        {
            config.features = parse_size(option, value);
        }
        else if ( option == "--categorical-features" )
        {
            config.categorical_features = parse_size(option, value);
        }
        else if ( option == "--categories" )
        {
            config.categories = parse_size(option, value);
        }
        else if ( option == "--seed" )
        {
            config.seed = parse_size(option, value);
        }
        else if ( option == "--workers" )
        {
            config.split_workers = parse_size(option, value);
        }
        else if ( option == "--ties" )
        {
            if ( value != "reference" && value != "stable" )
            {
                throw std::invalid_argument(
                    "ties must be reference or stable");
            }
            config.stable_ties = value == "stable";
        }
        else if ( option == "--value-levels" )
        {
            config.value_levels = parse_size(option, value);
        }
        else
        {
            throw std::invalid_argument("unknown option: " +
                                        std::string(option));
        }
    }

    if ( config.rows == 0 ) throw std::invalid_argument("rows must be positive");
    if ( config.split_workers < 1 || config.split_workers > 8 )
    {
        throw std::invalid_argument("workers must be between 1 and 8");
    }
    if ( config.features == 0 )
    {
        throw std::invalid_argument("features must be positive");
    }
    if ( config.categorical_features > config.features )
    {
        throw std::invalid_argument(
            "categorical-features must not exceed features");
    }
    if ( config.value_levels == 1 )
    {
        throw std::invalid_argument("value-levels must be 0 or at least two");
    }
    if ( config.categories < 2 )
    {
        throw std::invalid_argument("categories must be at least two");
    }
    if ( config.rows >
         std::numeric_limits<std::size_t>::max() / config.features )
    {
        throw std::invalid_argument("rows times features is too large");
    }
    return config;
}

std::string make_names(const configuration &config)
{
    std::ostringstream names;
    names << "class_0, class_1, class_2.\n\n";
    const std::size_t categorical_start =
        config.features - config.categorical_features;
    for ( std::size_t feature = 0; feature < config.features; ++feature )
    {
        names << "feature_" << feature << ": ";
        if ( feature < categorical_start )
        {
            names << "continuous.\n";
            continue;
        }
        for ( std::size_t category = 0; category < config.categories;
              ++category )
        {
            if ( category ) names << ", ";
            names << "category_" << category;
        }
        names << ".\n";
    }
    return names.str();
}

void generate_data(const configuration &config, std::vector<double> &values,
                   std::vector<std::size_t> &classes)
{
    deterministic_generator generator(config.seed);
    const std::size_t categorical_start =
        config.features - config.categorical_features;
    const std::size_t informative = std::min<std::size_t>(8, config.features);

    values.resize(config.rows * config.features);
    classes.resize(config.rows);
    for ( std::size_t row = 0; row < config.rows; ++row )
    {
        double signal = 0.0;
        for ( std::size_t feature = 0; feature < config.features; ++feature )
        {
            double value;
            if ( feature < categorical_start )
            {
                value = 2.0 * generator.unit_interval() - 1.0;
                if ( config.value_levels )
                {
                    // Evenly spaced levels in [-1, 1] create equal values.
                    const double steps =
                        static_cast<double>(config.value_levels - 1);
                    value = std::round((value + 1.0) / 2.0 * steps) /
                                steps * 2.0 - 1.0;
                }
            }
            else
            {
                value = static_cast<double>(generator.next() %
                                            config.categories);
            }
            if ( feature < informative )
            {
                const double weight =
                    1.0 - 0.75 * static_cast<double>(feature) /
                              static_cast<double>(std::max<std::size_t>(
                                  informative - 1, 1));
                signal += weight * value;
            }
            if ( generator.next() % 100 == 0 )
            {
                value = std::numeric_limits<double>::quiet_NaN();
            }
            values[row * config.features + feature] = value;
        }
        classes[row] = signal < -0.5 ? 0 : signal < 0.5 ? 1 : 2;
    }
}

std::uint64_t stable_model_hash(const c50::model &model)
{
    std::string_view serialized(model.serialized_data());
    const std::size_t newline = serialized.find('\n');
    if ( newline != std::string_view::npos )
    {
        serialized.remove_prefix(newline + 1);
    }

    std::uint64_t hash = UINT64_C(14695981039346656037);
    for ( const unsigned char byte : serialized )
    {
        hash ^= byte;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

int run(const configuration &config)
{
    std::vector<double> values;
    std::vector<std::size_t> classes;
    generate_data(config, values, classes);
    const std::string names = make_names(config);

    c50::context context;
    context.split_workers(static_cast<unsigned>(config.split_workers));
    const c50::dense_dataset dataset(values.data(), config.rows, config.features,
                                     classes.data());
    c50::options options;
    options.minimum_cases = 20;
    options.subset_splits = config.subsets;
    options.ties = config.stable_ties ? c50::tie_order::stable
                                      : c50::tie_order::reference;

    const auto model = c50::model::train(context, c50::model_kind::tree,
                                        names, dataset, options);
    const std::size_t serialized_size = model.serialized_data().size();

    std::cout
        << "{\n"
        << "  \"model\": {\n"
        << "    \"serialized_bytes\": " << serialized_size << ",\n"
        << "    \"stable_fnv1a64\": \"" << std::hex
        << std::setw(16) << std::setfill('0')
        << stable_model_hash(model) << std::dec << "\"\n"
        << "  },\n"
        << "  \"workload\": {\n"
        << "    \"categories\": " << config.categories << ",\n"
        << "    \"categorical_features\": "
        << config.categorical_features << ",\n"
        << "    \"features\": " << config.features << ",\n"
        << "    \"rows\": " << config.rows << ",\n"
        << "    \"seed\": " << config.seed << ",\n"
        << "    \"split_workers\": " << config.split_workers << ",\n"
        << "    \"subsets\": " << (config.subsets ? "true" : "false") << ",\n"
        << "    \"ties\": \""
        << (config.stable_ties ? "stable" : "reference") << "\",\n"
        << "    \"value_levels\": " << config.value_levels << "\n"
        << "  }\n"
        << "}\n";
    return 0;
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        return run(parse_arguments(argc, argv));
    }
    catch ( const std::exception &error )
    {
        std::cerr << "parallel-workload: " << error.what() << '\n';
        return 1;
    }
}
