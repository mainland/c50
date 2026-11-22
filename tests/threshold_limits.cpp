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
                c50::model_kind kind)
{
    const c50::options options;
    const c50::model text = c50::model::train(context, kind, names,
                                              text_cases(pair), options);
    check_predictions(context, text, pair);

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
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
