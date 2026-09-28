/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <c50/c50.hpp>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

std::string read(const std::string &path)
{
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot read fixture: " + path);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

void equal(const c50::predictions &actual, const c50::predictions &expected)
{
    require(actual.size() == expected.size() && actual.class_count() == expected.class_count(),
            "prediction shape");
    for (std::size_t label = 0; label < actual.class_count(); ++label)
        require(actual.class_name(label) == expected.class_name(label), "class names");
    for (std::size_t row = 0; row < actual.size(); ++row) {
        require(actual.class_index(row) == expected.class_index(row) &&
                actual.confidence(row) == expected.confidence(row), "label and confidence");
        for (std::size_t label = 0; label < actual.class_count(); ++label)
            require(actual.score(row, label) == expected.score(row, label), "exact scores");
    }
}

void text_cases(const std::string &names, const std::string &training,
                const std::string &cases, const std::string &costs = {}, unsigned trials = 1)
{
    for (auto kind : {c50::model_kind::tree, c50::model_kind::rules}) {
        c50::context context;
        c50::options options;
        options.trials = trials;
        options.subset_splits = true;
        options.probabilistic_thresholds = true;
        auto model = c50::model::train(context, kind, names, training, options, costs);
        auto expected = model.predict(context, cases);
        auto predictor = model.prepare_predictor();
        for (int repeat = 0; repeat < 5; ++repeat) {
            equal(predictor.predict(cases), expected);
            equal(predictor.predict(""), model.predict(context, ""));
        }
        auto worker = [&] {
            auto own = model.prepare_predictor();
            for (int repeat = 0; repeat < 5; ++repeat) equal(own.predict(cases), expected);
        };
        auto first = std::async(std::launch::async, worker);
        auto second = std::async(std::launch::async, worker);
        first.get();
        second.get();
    }
}

void dense_and_recovery()
{
    const std::string names = "low, high.\nx: continuous.\ncolor: red, blue.\n";
    const std::string training = "0, red, low\n1, red, low\n3, blue, high\n4, blue, high\n";
    const double values[] = {0, 0, 4, 1, std::numeric_limits<double>::quiet_NaN(), 1};
    const c50::dense_dataset cases(values, 3, 2);
    for (auto kind : {c50::model_kind::tree, c50::model_kind::rules}) {
        c50::context context;
        auto model = c50::model::train(context, kind, names, training);
        auto expected = model.predict(context, cases);
        auto predictor = model.prepare_predictor();
        for (int repeat = 0; repeat < 5; ++repeat) {
            equal(predictor.predict(cases), expected);
            equal(predictor.predict("0, red, ?\n4, blue, ?\n?, blue, ?"), expected);
            try {
                predictor.predict("broken, red, ?\n4, blue, ?\n");
                throw std::runtime_error("invalid case accepted");
            } catch (const c50::exception &error) {
                require(error.code() == c50::error_code::parse_error, "parse failure kind");
            }
            equal(predictor.predict(cases), expected);
            try {
                const double invalid[] = {1, 99};
                predictor.predict(c50::dense_dataset(invalid, 1, 2));
                throw std::runtime_error("invalid category accepted");
            } catch (const c50::exception &error) {
                require(error.code() == c50::error_code::invalid_argument, "dense failure kind");
            }
            equal(predictor.predict(cases), expected);
        }
        auto independent = [&] {
            auto temporary = c50::model::train(context, kind, names, training);
            return temporary.prepare_predictor();
        }();
        auto moved = std::move(independent);
        predictor = std::move(moved);
        equal(predictor.predict(cases), expected);
        auto retained = [&] {
            auto temporary = model.prepare_predictor();
            return temporary.predict(cases);
        }();
        equal(retained, expected);
    }
}
} // namespace

int main(int argc, char **argv)
{
    try {
        require(argc == 2, "expected fixture directory");
        const std::string root = argv[1];
        for (const std::string name : {"basic", "implicit", "case-weight", "multiclass", "boost"}) {
            const auto stem = root + "/" + name + "/" + name;
            const auto training = read(stem + ".data");
            text_cases(read(stem + ".names"), training, training,
                       name == "basic" ? read(stem + ".costs") : "", name == "boost" ? 3 : 1);
        }
        text_cases("low, high.\ncolor: discrete 5.\n",
                   "red, low\nred, low\nblue, high\nblue, high\n",
                   "green, ?\nred, ?\n?, ?\nN/A, ?\nblue, ?\n");
        dense_and_recovery();
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
