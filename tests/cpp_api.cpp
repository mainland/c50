/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <string>
#include <type_traits>
#include <utility>

#include <c50/c50.hpp>

static_assert(!std::is_copy_constructible<c50::context>::value,
              "contexts must not be copyable");
static_assert(std::is_move_constructible<c50::context>::value,
              "contexts must be movable");
static_assert(!std::is_copy_constructible<c50::model>::value,
              "models must not be copyable");
static_assert(std::is_move_constructible<c50::model>::value,
              "models must be movable");
static_assert(!std::is_copy_constructible<c50::predictions>::value,
              "predictions must not be copyable");

int main()
{
    const std::string names =
        "low, high.\n\n"
        "signal: continuous.\n";
    const std::string training =
        "0, low\n1, low\n2, low\n3, high\n4, high\n5, high\n";
    const std::string cases = "0, ?\n5, ?\n";
    const double dense_training_values[] = {0, 1, 2, 3, 4, 5};
    const std::size_t dense_training_classes[] = {0, 0, 0, 1, 1, 1};
    const double dense_case_values[] = {0, 5};
    c50::context context;
    c50::options options;

    context.split_workers(2);
    try
    {
        context.split_workers(0);
        return 1;
    }
    catch ( const c50::exception &error )
    {
        if ( error.code() != c50::error_code::invalid_argument ) return 1;
    }
    context.split_workers(1);

    options.trials = 1;
    options.subset_splits = true;
    options.winnow = false;
    options.global_pruning = true;
    options.probabilistic_thresholds = false;
    options.ignore_costs = false;
    options.minimum_cases = 2;
    options.confidence_factor = 0.25;
    options.sample_fraction = 0;
    options.random_seed = 0;
    if ( options.trials != 1 || ! options.subset_splits ||
         options.winnow || ! options.global_pruning ||
         options.probabilistic_thresholds || options.ignore_costs ||
         options.minimum_cases != 2 ||
         options.confidence_factor != 0.25 ||
         options.sample_fraction != 0 || options.random_seed != 0 )
    {
        return 1;
    }

    c50::model trained = c50::model::train(
        context, c50::model_kind::tree, names, training, options);
    if ( trained.kind() != c50::model_kind::tree ||
         trained.names_data() != names || trained.serialized_data().empty() ||
         ! trained.costs_data().empty() ) return 1;

    const c50::dense_dataset dense_training(
        dense_training_values, 6, 1, dense_training_classes);
    c50::model dense_trained = c50::model::train(
        context, c50::model_kind::tree, names, dense_training, options);
    if ( dense_trained.serialized_data() != trained.serialized_data() ) return 1;

    c50::predictions result = trained.predict(context, cases);
    if ( result.size() != 2 || result.class_count() != 2 ||
         std::string(result.class_name(0)) != "low" ||
         std::string(result.class_name(1)) != "high" ||
         result.class_index(0) != 0 || result.class_index(1) != 1 ) return 1;

    const c50::dense_dataset dense_cases(dense_case_values, 2, 1);
    c50::predictions dense_result = dense_trained.predict(context, dense_cases);
    if ( dense_result.size() != 2 || dense_result.class_index(0) != 0 ||
         dense_result.class_index(1) != 1 ) return 1;

    c50::model loaded = c50::model::load(
        context, trained.kind(), trained.names_data(),
        trained.serialized_data(), trained.costs_data());
    c50::model moved = std::move(loaded);
    result = moved.predict(context, cases);
    if ( result.class_index(0) != 0 || result.class_index(1) != 1 ) return 1;

    try
    {
        static_cast<void>(c50::model::train(
            context, c50::model_kind::tree, names, "malformed", options));
        return 1;
    }
    catch ( const c50::exception &error )
    {
        if ( error.code() != c50::error_code::parse_error ||
             std::string(error.what()).empty() ) return 1;
    }

    return 0;
}
