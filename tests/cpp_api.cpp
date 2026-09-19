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
    c50::context context;
    c50::options options;

    options.trials(1)
           .subset_splits(true)
           .winnow(false)
           .global_pruning(true)
           .probabilistic_thresholds(false)
           .ignore_costs(false)
           .minimum_cases(2)
           .confidence_factor(0.25)
           .sample_fraction(0)
           .random_seed(0);
    if ( options.trials() != 1 || ! options.subset_splits() ||
         options.winnow() || ! options.global_pruning() ||
         options.probabilistic_thresholds() || options.ignore_costs() ||
         options.minimum_cases() != 2 ||
         options.confidence_factor() != 0.25 ||
         options.sample_fraction() != 0 || options.random_seed() != 0 )
    {
        return 1;
    }

    c50::model trained = c50::model::train(
        context, c50::model_kind::tree, names, training, options);
    if ( trained.kind() != c50::model_kind::tree ||
         trained.names_data() != names || trained.serialized_data().empty() ||
         ! trained.costs_data().empty() ) return 1;

    c50::predictions result = trained.predict(context, cases);
    if ( result.size() != 2 || result.class_count() != 2 ||
         std::string(result.class_name(0)) != "low" ||
         std::string(result.class_name(1)) != "high" ||
         result.class_index(0) != 0 || result.class_index(1) != 1 ) return 1;

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
        if ( error.status() != C50_STATUS_PARSE_ERROR ||
             std::string(error.what()).empty() ) return 1;
    }

    return context.native_handle() && trained.native_handle() &&
           result.native_handle() ? 0 : 1;
}
