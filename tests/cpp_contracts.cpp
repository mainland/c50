/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <c50/c50.hpp>

#include <cmath>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

static const std::string names = "low, high.\nsignal: continuous.\n";
static const std::string data =
    "0, low\n1, low\n2, low\n3, high\n4, high\n5, high\n";

static c50::model train(c50::model_kind kind)
{
    c50::context context;
    return c50::model::train(context, kind, names, data);
}

static void ownership_and_recovery()
{
    auto model = train(c50::model_kind::tree);
    auto result = [&] {
        c50::context context;
        auto temporary = train(c50::model_kind::rules);
        return temporary.predict(context, "0, ?\n5, ?\n");
    }();
    require(result.size() == 2 && result.class_index(0) == 0 &&
            result.class_index(1) == 1 &&
            std::string(result.class_name(1)) == "high", "result lifetime");
    c50::context context;
    for (int iteration = 0; iteration < 10; ++iteration) {
        try {
            model.predict(context, "not a number, ?\n");
            throw std::runtime_error("invalid input accepted");
        } catch (const c50::exception &) {
        }
        auto prediction = model.predict(context, "0, ?\n");
        require(prediction.size() == 1 && prediction.class_index(0) == 0,
                "context recovery");
    }
}

static void concurrency()
{
    auto shared = train(c50::model_kind::tree);
    std::promise<void> start;
    auto ready = start.get_future().share();
    std::vector<std::future<void>> workers;
    for (unsigned index = 0; index < 4; ++index) {
        workers.push_back(std::async(std::launch::async, [&, index] {
            c50::context context;
            ready.wait();
            for (unsigned i = 0; i < 20; ++i) {
                const bool reverse = index % 2;
                auto own = c50::model::train(
                    context, i % 2 ? c50::model_kind::tree : c50::model_kind::rules,
                    names, reverse ?
                    "0, high\n1, high\n2, high\n3, low\n4, low\n5, low\n" : data);
                auto prediction = own.predict(context, "0, ?\n");
                require(prediction.class_index(0) == (reverse ? 1u : 0u),
                        "concurrent training");
                prediction = shared.predict(context, "5, ?\n");
                require(prediction.class_index(0) == 1, "shared model prediction");
            }
        }));
    }
    start.set_value();
    for (auto &worker : workers) worker.get();
}

static std::string model_body(const c50::model &model)
{
    const auto bytes = model.serialized_data();
    return bytes.substr(bytes.find('\n') + 1); // The first line contains the date.
}

static void reused_context_matches_fresh()
{
    // Winnowing reads state that the previous training on a context left
    // behind unless each training resets it.
    const std::string names =
        "a, b, c.\nx: continuous.\ny: continuous.\ncase weight: continuous.\n";
    c50::options winnow;
    winnow.winnow = true;
    for (unsigned multiplier = 1; multiplier < 40; ++multiplier) {
        std::string data;
        for (unsigned row = 0; row < 60; ++row) {
            const unsigned x = row * 7 % 19, y = row * multiplier % 23;
            data += std::to_string(x) + ", " + std::to_string(y) + ", " +
                    std::to_string(row % 5 + 1) + ", " +
                    "abc"[(x + y + row % 3) % 3] + "\n";
        }
        for (auto kind : {c50::model_kind::tree, c50::model_kind::rules}) {
            c50::context fresh, reused;
            auto expected = c50::model::train(fresh, kind, names, data, winnow);
            c50::model::train(reused, kind, names, data, winnow);
            auto model = c50::model::train(reused, kind, names, data, winnow);
            require(model_body(model) == model_body(expected),
                    "reused context changed a winnowed classifier");
        }
    }
}

static void parallel_equivalence()
{
    constexpr std::size_t rows = 10001;
    constexpr std::size_t columns = 6;
    std::vector<double> values(rows * columns);
    std::vector<std::size_t> classes(rows);
    std::string schema = "a, b, c.\n";
    for (std::size_t column = 0; column < columns; ++column)
        schema += "x" + std::to_string(column) +
                  (column == 5 ? ": red, green, blue.\n" : ": continuous.\n");
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column)
            values[row * columns + column] = column == 5 ? row % 3 :
                (row * (column * 16 + 17) % 997) / 997.0;
        classes[row] = values[row * columns] < 0.3 ? 0 :
                       values[row * columns + 1] < 0.6 ? 1 : 2;
    }
    const c50::dense_dataset training(values.data(), rows, columns, classes.data());
    const c50::dense_dataset cases(values.data(), rows, columns);
    for (auto kind : {c50::model_kind::tree, c50::model_kind::rules}) {
        c50::options options;
        options.trials = 3;
        options.subset_splits = true;
        options.minimum_cases = 10;
        c50::context context;
        auto reference = c50::model::train(context, kind, schema, training, options);
        auto expected = reference.predict(context, cases);
        for (unsigned count : {2u, 4u, 8u}) {
            context.split_workers(count);
            auto model = c50::model::train(context, kind, schema, training, options);
            require(model_body(model) == model_body(reference), "parallel classifier");
            auto actual = model.predict(context, cases);
            require(actual.size() == rows && actual.class_count() == 3,
                    "parallel result shape");
            for (std::size_t row = 0; row < rows; ++row) {
                require(actual.class_index(row) == expected.class_index(row) &&
                        actual.confidence(row) == expected.confidence(row),
                        "parallel prediction");
                for (std::size_t column = 0; column < 3; ++column)
                    require(actual.score(row, column) == expected.score(row, column),
                            "parallel score");
            }
        }
    }
}

int main()
{
    try {
        ownership_and_recovery();
        concurrency();
        parallel_equivalence();
        reused_context_matches_fresh();
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
