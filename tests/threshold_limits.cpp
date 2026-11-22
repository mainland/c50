/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

/* Training on values near the limits of float. The threshold between two
   neighbouring values is half of their float sum, and that sum overflows for
   two large values of the same sign. */

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <c50/c50.hpp>

namespace {

const std::string names = "a, b.\n\nx: continuous.\n";

void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

struct pair_of_values {
    const char *low;
    const char *high;
    double low_value;
    double high_value;
};

/* Five cases of each class. The sum of each pair overflows a float. */
const pair_of_values negative = {"-3e38", "-2e38", -3e38, -2e38};
const pair_of_values positive = {"2e38", "3e38", 2e38, 3e38};

/* Values within 1% of the largest finite float. */
const pair_of_values negative_limit = {"-3.4e38", "-3e36", -3.4e38, -3e36};
const pair_of_values positive_limit = {"3e36", "3.4e38", 3e36, 3.4e38};

std::string text_cases(const pair_of_values &pair)
{
    std::string text;
    for (int i = 0; i < 5; ++i) text += std::string(pair.low) + ",a\n";
    for (int i = 0; i < 5; ++i) text += std::string(pair.high) + ",b\n";
    return text;
}

void check_predictions(c50::context &context, const c50::model &model,
                       const pair_of_values &pair)
{
    const std::string cases =
        std::string(pair.low) + ",?\n" + std::string(pair.high) + ",?\n";
    const c50::predictions result = model.predict(context, cases);
    require(result.size() == 2, "two predictions");
    require(result.class_index(0) == 0, "the lower value is class a");
    require(result.class_index(1) == 1, "the higher value is class b");
}

void train_pair(c50::context &context, const pair_of_values &pair,
                c50::model_kind kind, bool soft_thresholds = false)
{
    c50::options options;
    options.probabilistic_thresholds = soft_thresholds;
    const c50::model text = c50::model::train(context, kind, names,
                                              text_cases(pair), options);
    check_predictions(context, text, pair);
    require(text.serialized_data().find("inf") == std::string::npos,
            "the model holds no infinite value");

    std::vector<double> values;
    std::vector<std::size_t> classes;
    for (int i = 0; i < 5; ++i) {
        values.push_back(pair.low_value);
        classes.push_back(0);
    }
    for (int i = 0; i < 5; ++i) {
        values.push_back(pair.high_value);
        classes.push_back(1);
    }
    const c50::dense_dataset dense(values.data(), values.size(), 1,
                                   classes.data());
    const c50::model from_dense =
        c50::model::train(context, kind, names, dense, options);
    check_predictions(context, from_dense, pair);
    require(from_dense.serialized_data().size() > 0, "a serialized model");
}

/* A cut at the negated largest float is written as -3.4028235e+38, which
   rounds beyond the float range when read back as a double. This checks the
   trained text only, and not that it loads. */
void bounds_at_largest_float_are_finite(c50::context &context)
{
    const std::string data =
        "-3.4028234e38,a\n-3.4028234e38,a\n-3.4028234e38,a\n"
        "-1.1e31,b\n-1.1e31,b\n-1.1e31,b\n";
    c50::options options;
    options.probabilistic_thresholds = true;
    const c50::model model = c50::model::train(
        context, c50::model_kind::tree, names, data, options);
    require(model.serialized_data().find("inf") == std::string::npos,
            "the bounds at the largest float are finite");
}

/* The soft threshold of two values next to -FLT_MAX puts a bound at the
   negated largest float, which the writer prints as -3.4028235e+38. The
   loaded model must predict as the trained one does. */
void bound_at_largest_float_loads(c50::context &context)
{
    const std::string lowest = "-3.4028235e+38";
    const std::string data =
        "-3.4028234e38,a\n-3.4028234e38,a\n-3.4028234e38,a\n"
        "-1.1e31,b\n-1.1e31,b\n-1.1e31,b\n";
    const std::string probes =
        "-3.4028234e38,?\n-3.4028235e+38,?\n-3e38,?\n-1e38,?\n"
        "-1.1e31,?\n0,?\n";
    c50::options options;
    options.probabilistic_thresholds = true;
    for (const auto kind :
         {c50::model_kind::tree, c50::model_kind::rules}) {
        const c50::model trained =
            c50::model::train(context, kind, names, data, options);
        require(trained.serialized_data().find(lowest) != std::string::npos,
                "the model has a bound at -FLT_MAX");
        const c50::model loaded = c50::model::load(
            context, kind, names, trained.serialized_data());
        const c50::predictions expected = trained.predict(context, probes);
        const c50::predictions actual = loaded.predict(context, probes);
        require(expected.size() == actual.size(), "one prediction per probe");
        for (std::size_t row = 0; row < expected.size(); ++row)
            require(expected.class_index(row) == actual.class_index(row),
                    "the loaded model predicts as the trained one");
        require(actual.class_index(0) == 0, "the lower value is class a");
        require(actual.class_index(4) == 1, "the higher value is class b");
    }
}

/* A value whose float has the bits of an unknown or a not-applicable marker
   is read as known by the split and as missing by the threshold adjustment,
   so every case of the node can end up below the adjusted cut. Softening the
   cut then must not read past the node. */
const char *const marker_values[] = {"1.5777218104420236e-30", "1e-45"};

void soft_thresholds_train_on_marker_values(c50::context &context)
{
    c50::options options;
    options.probabilistic_thresholds = true;
    const std::string marker_names = "no, yes.\n\nx: continuous.\n";
    for (const char *marker : marker_values) {
        const std::string value = marker;
        const std::string data =
            "5,no\n5,no\n" + value + ",yes\n" + value + ",yes\n";
        for (const auto kind :
             {c50::model_kind::tree, c50::model_kind::rules}) {
            const c50::model text = c50::model::train(
                context, kind, marker_names, data, options);
            require(text.serialized_data().size() > 0, "a serialized model");
        }

        const double dense_value = std::stod(value);
        const std::vector<double> values = {5, 5, dense_value, dense_value};
        const std::vector<std::size_t> classes = {0, 0, 1, 1};
        const c50::dense_dataset dense(values.data(), values.size(), 1,
                                       classes.data());
        for (const auto kind :
             {c50::model_kind::tree, c50::model_kind::rules}) {
            const c50::model model = c50::model::train(
                context, kind, marker_names, dense, options);
            require(model.serialized_data().size() > 0,
                    "a model from dense data");
        }
    }
}

/* A value that is not a number is stored as unknown. Text training rejects it
   with a parse error, and must not read past the node on the way. */
void soft_thresholds_reject_a_bad_value(c50::context &context)
{
    c50::options options;
    options.probabilistic_thresholds = true;
    const std::string marker_names = "no, yes.\n\nx: continuous.\n";
    bool rejected = false;
    try {
        (void) c50::model::train(context, c50::model_kind::tree,
                                 marker_names,
                                 "5,no\n5,no\nbad,yes\nbad,yes\n", options);
    } catch (const c50::exception &error) {
        rejected = error.code() == c50::error_code::parse_error;
    }
    require(rejected, "a bad value is rejected as a parse error");
}

} // namespace

int main()
{
    try {
        c50::context context;
        for (const auto kind :
             {c50::model_kind::tree, c50::model_kind::rules}) {
            train_pair(context, negative, kind);
            train_pair(context, positive, kind);
        }

        // Soft thresholds widen the cut into bounds around the midpoint of
        // the two values, which must stay finite.
        for (const auto &pair :
             {negative, positive, negative_limit, positive_limit}) {
            train_pair(context, pair, c50::model_kind::tree, true);
        }
        bounds_at_largest_float_are_finite(context);
        bound_at_largest_float_loads(context);
        soft_thresholds_train_on_marker_values(context);
        soft_thresholds_reject_a_bad_value(context);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
