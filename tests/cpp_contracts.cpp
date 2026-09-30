/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <c50/c50.hpp>

#include <cmath>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
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

static void shared_model_concurrency()
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

static void require_predictions_equal(const c50::predictions &actual,
                                      const c50::predictions &expected)
{
    require(actual.size() == expected.size() &&
            actual.class_count() == expected.class_count(), "parallel result shape");
    for (std::size_t column = 0; column < actual.class_count(); ++column)
        require(std::string(actual.class_name(column)) == expected.class_name(column),
                "parallel class names");
    for (std::size_t row = 0; row < actual.size(); ++row) {
        require(actual.class_index(row) == expected.class_index(row) &&
                actual.confidence(row) == expected.confidence(row), "parallel prediction");
        for (std::size_t column = 0; column < actual.class_count(); ++column)
            require(actual.score(row, column) == expected.score(row, column), "parallel score");
    }
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

static void winnowing_uses_case_weights()
{
    // Winnowing must use the relative case weights of construction.
    const std::string names = "no, yes.\nsignal: continuous.\n"
                              "noise: continuous.\ncase weight: continuous.\n";
    std::string data;
    for (unsigned row = 0; row < 60; ++row) {
        const unsigned signal = row * 7 % 21, noise = row * 11 % 17;
        data += std::to_string(signal) + ", " + std::to_string(noise) + ", " +
                std::to_string(row % 4 + 1) + (signal > 10 ? ", yes\n" : ", no\n");
    }
    c50::options winnow;
    winnow.winnow = true;
    for (auto kind : {c50::model_kind::tree, c50::model_kind::rules}) {
        c50::context context;
        auto model = c50::model::train(context, kind, names, data, winnow);
        require(model.serialized_data().find("att=\"signal\"") != std::string::npos,
                "winnowing with case weights discarded the informative attribute");
    }
}

static void concurrent_parallel_fits()
{
    constexpr std::size_t rows = 10000, columns = 6, fits = 4;
    std::vector<double> values(rows * columns);
    std::vector<std::vector<std::size_t>> labels(fits, std::vector<std::size_t>(rows));
    std::string schema = "no, yes.\n";
    for (std::size_t column = 0; column < columns; ++column)
        schema += "x" + std::to_string(column) + ": red, green, blue, yellow.\n";
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column)
            values[row * columns + column] = (row / (column + 1)) % 4;
        for (std::size_t fit = 0; fit < fits; ++fit)
            labels[fit][row] = (values[row * columns] < 2 ? 0 : 1) ^ (fit / 2);
    }
    c50::options options;
    options.subset_splits = true;
    const c50::dense_dataset cases(values.data(), rows, columns);
    std::vector<c50::model> references;
    std::vector<c50::predictions> expected;
    for (std::size_t fit = 0; fit < fits; ++fit) {
        c50::context context;
        const c50::dense_dataset data(values.data(), rows, columns, labels[fit].data());
        const auto kind = fit % 2 ? c50::model_kind::rules : c50::model_kind::tree;
        references.push_back(c50::model::train(context, kind, schema, data, options));
        expected.push_back(references.back().predict(context, cases));
    }
    std::promise<void> start;
    auto ready = start.get_future().share();
    std::vector<std::future<void>> workers;
    for (std::size_t fit = 0; fit < fits; ++fit) {
        workers.push_back(std::async(std::launch::async, [&, fit] {
            c50::context context;
            context.split_workers(4);
            const c50::dense_dataset data(values.data(), rows, columns, labels[fit].data());
            const auto kind = fit % 2 ? c50::model_kind::rules : c50::model_kind::tree;
            ready.wait();
            for (unsigned repeat = 0; repeat < 3; ++repeat) {
                auto model = c50::model::train(context, kind, schema, data, options);
                require(model_body(model) == model_body(references[fit]),
                        "simultaneous parallel classifier");
                require_predictions_equal(model.predict(context, cases), expected[fit]);
                require_predictions_equal(references[0].predict(context, cases), expected[0]);
            }
        }));
    }
    start.set_value();
    for (auto &worker : workers) worker.get();
}

static void parallel_schema_equivalence()
{
    constexpr std::size_t rows = 20001;
    std::string schema = "a, b, c.\n";
    for (unsigned column = 0; column < 5; ++column)
        schema += "x" + std::to_string(column) + ": continuous.\n";
    schema += "rank: [ordered] low, medium, high.\n"
              "color: red, green, blue, yellow.\ncase weight: continuous.\n";
    const char *ranks[] = {"low", "medium", "high"};
    const char *colors[] = {"red", "green", "blue", "yellow"};
    const char *labels[] = {"a", "b", "c"};
    const std::string costs = "a, b: 3\nb, c: 2\nc, a: 4\n";
    std::string cases;
    for (std::size_t row = 0; row < rows; ++row) {
        for (unsigned column = 0; column < 5; ++column) {
            const auto value = row * (column * 16 + 17) % 997;
            cases += (row + column) % 37 == 0 ? "?, " :
                     (row + column) % 53 == 0 ? "N/A, " : std::to_string(value) + ", ";
        }
        cases += row % 29 == 0 ? "?, " : std::string(ranks[row % 3]) + ", ";
        cases += row % 31 == 0 ? "N/A, " : std::string(colors[row % 4]) + ", ";
        cases += std::to_string(row % 5 + 1) + ", ";
        cases += labels[row * 17 % 997 < 300 ? 0 : row * 33 % 997 < 600 ? 1 : 2];
        cases += '\n';
    }
    for (auto kind : {c50::model_kind::tree, c50::model_kind::rules}) {
        for (auto [sample, ties] : {std::pair{0.0, c50::tie_order::reference},
                                    std::pair{0.75, c50::tie_order::reference},
                                    std::pair{0.0, c50::tie_order::stable},
                                    std::pair{0.75, c50::tie_order::stable}}) {
            c50::options options;
            options.ties = ties;
            options.subset_splits = true;
            options.minimum_cases = 20;
            options.sample_fraction = sample;
            options.random_seed = 73;
            c50::context context;
            auto reference = c50::model::train(context, kind, schema, cases, options, costs);
            auto expected = reference.predict(context, cases);
            for (unsigned workers : {2u, 4u, 8u}) {
                context.split_workers(workers);
                auto model = c50::model::train(context, kind, schema, cases, options, costs);
                require(model_body(model) == model_body(reference), "mixed-schema classifier");
                require_predictions_equal(model.predict(context, cases), expected);
                auto loaded = c50::model::load(context, kind, model.names_data(),
                                               model.serialized_data(), model.costs_data());
                require_predictions_equal(loaded.predict(context, cases), expected);
            }
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
    for (std::size_t row = 0; row < rows; ++row) {
        if (row % 23 == 0) values[row * columns + 4] = std::numeric_limits<double>::quiet_NaN();
        if (row % 31 == 0) values[row * columns + 5] = std::numeric_limits<double>::quiet_NaN();
    }
    const c50::dense_dataset training(values.data(), rows, columns, classes.data());
    const c50::dense_dataset cases(values.data(), rows, columns);
    for (auto [kind, ties] : {std::pair{c50::model_kind::tree, c50::tie_order::reference},
                              std::pair{c50::model_kind::rules, c50::tie_order::reference},
                              std::pair{c50::model_kind::tree, c50::tie_order::stable},
                              std::pair{c50::model_kind::rules, c50::tie_order::stable}}) {
        c50::options options;
        options.ties = ties;
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
        shared_model_concurrency();
        parallel_equivalence();
        parallel_schema_equivalence();
        concurrent_parallel_fits();
        reused_context_matches_fresh();
        winnowing_uses_case_weights();
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
