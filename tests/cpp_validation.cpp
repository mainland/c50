/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <c50/c50.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

static void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}
template<class F> static void rejects(c50::error_code code, F operation)
{
    try { operation(); }
    catch (const c50::exception &error) {
        require(error.code() == code, error.what());
        require(*error.what(), "missing diagnostic");
        return;
    }
    throw std::runtime_error("invalid input accepted");
}
template<class F> static void bounds(F operation)
{
    try { operation(); }
    catch (const std::out_of_range &) { return; }
    throw std::runtime_error("invalid index accepted");
}
static const std::string names = "low, high.\nsignal: continuous.\ngroup: alpha, beta.\n";
static const std::string header = "id=\"See5/C5.0 2.07 GPL Edition 2026-09-19\"\n";
static const std::string tree = header +
    "entries=\"1\"\n"
    "type=\"2\" class=\"low\" att=\"signal\" forks=\"3\" cut=\"1\" freq=\"1,1\"\n"
    "type=\"0\" class=\"low\" freq=\"1,1\"\n"
    "type=\"0\" class=\"low\" freq=\"1,0\"\n"
    "type=\"0\" class=\"high\" freq=\"0,1\"\n";
static const auto invalid = c50::error_code::invalid_argument;
static const auto parse = c50::error_code::parse_error;

static void model_validation()
{
    c50::context context;
    const auto kind = c50::model_kind::tree;
    auto model = c50::model::load(context, kind, names, tree);
    auto result = model.predict(context, "0.5, alpha, ?\n2, beta, ?\n?, alpha, ?\nN/A, beta, ?\n");
    require(result.size() == 4 && result.class_count() == 2 &&
            result.class_name(0) == "low" && result.class_name(1) == "high", "metadata");
    for (std::size_t row = 0; row < 4; ++row) {
        require(result.class_index(row) == (row == 1 ? 1u : 0u), "class prediction");
        require(result.confidence(row) == (row == 3 ? 0.5 : 1), "confidence");
    }
    require(result.score(0, 0) == 1 && result.score(0, 1) == 0 &&
            result.score(1, 0) == 0 && result.score(1, 1) == 1 &&
            result.score(2, 0) == 1 && result.score(2, 1) == 1, "scores");
    bounds([&] { result.class_name(2); });
    bounds([&] { result.class_index(4); });
    bounds([&] { result.confidence(4); });
    bounds([&] { result.score(4, 0); });
    bounds([&] { result.score(0, 2); });
    result = model.predict(context, std::string_view{});
    require(result.size() == 0 && result.class_count() == 2, "empty text batch");
    rejects(parse, [&] { model.predict(context, "0.5"); });
    rejects(parse, [&] { model.predict(context, "0.5, gamma, ?\n"); });
    const std::string nul(1, '\0');
    rejects(invalid, [&] { model.predict(context, nul); });
    rejects(invalid, [&] { c50::model::load(context, kind, names + nul, tree); });
    rejects(invalid, [&] { c50::model::load(context, kind, names, tree + nul); });
    rejects(invalid, [&] { c50::model::load(context, kind, names, tree, nul); });
    rejects(invalid, [&] { c50::model::load(context, static_cast<c50::model_kind>(10), names, tree); });
    rejects(invalid, [&] { c50::model::load(context, kind, "", tree); });
    rejects(invalid, [&] { c50::model::load(context, kind, names, ""); });
    for (const auto &bad : {header, header + "entries=\"0\"\n",
                            tree.substr(0, tree.rfind("type=\"0\""))})
        rejects(parse, [&] { c50::model::load(context, kind, names, bad); });
    const std::string rules = header + "entries=\"1\"\nrules=\"1\" default=\"low\"\n"
        "conds=\"0\" cover=\"1\" ok=\"1\" lift=\"1\" class=\"low\"\n";
    auto rule_model = c50::model::load(context, c50::model_kind::rules, names, rules);
    require(rule_model.kind() == c50::model_kind::rules, "rules kind");
    const std::string bad_rules = header + "entries=\"1\"\nrules=\"1\" default=\"low\"\n"
        "conds=\"1\" cover=\"1\" ok=\"1\" lift=\"1\" class=\"low\"\ntype=\"2\" att=\"signal\"";
    rejects(parse, [&] { c50::model::load(context, c50::model_kind::rules, names, bad_rules); });
    const std::string cost_tree = header + "costs=\"1\"\nentries=\"1\"\n"
        "type=\"0\" class=\"low\" freq=\"6,4\"\n";
    rejects(c50::error_code::io_error, [&] { c50::model::load(context, kind, names, cost_tree); });
    auto cost_model = c50::model::load(context, kind, names, cost_tree, "low, high: 5\n");
    c50::context other;
    for (int iteration = 0; iteration < 8; ++iteration) {
        require(cost_model.predict(context, "1, alpha, ?\n").class_index(0) == 1, "cost prediction");
        require(model.predict(other, "0.5, alpha, ?\n").class_index(0) == 0, "independent context");
        require(model.predict(context, "0.5, alpha, ?\n").class_index(0) == 0, "cost isolation");
    }
}

static void continuous_text_values()
{
    c50::context context;
    auto model = c50::model::load(context, c50::model_kind::tree, names, tree);
    for (const std::string value : {"nan", "inf", "-inf", "1e9999"}) {
        rejects(parse, [&] { model.predict(context, value + ", alpha, ?\n"); });
        require(model.predict(context, "2, alpha, ?\n").class_index(0) == 1,
                "prediction after rejecting nonfinite text");
    }
    auto result = model.predict(context,
        "3.4028234663852886e38, alpha, ?\n"
        "-3.4028234663852886e38, alpha, ?\n"
        "1e-40, alpha, ?\n"
        "0, alpha, ?\n-0, alpha, ?\n");
    require(result.size() == 5, "finite text row count");
    for (std::size_t row = 0; row < result.size(); ++row) {
        const std::size_t expected = row == 0 ? 1 : 0;
        require(result.class_index(row) == expected && result.confidence(row) == 1 &&
                result.score(row, expected) == 1 && result.score(row, 1 - expected) == 0,
                "finite text prediction");
    }
}

static void training_validation()
{
    c50::context context;
    const auto kind = c50::model_kind::tree;
    double values[] = {0, 0, 1, 1, 2, 0, 3, 1, 4, 0, 5, 1};
    std::size_t classes[] = {0, 0, 0, 1, 1, 1};
    c50::dense_dataset dataset(values, 6, 2, classes);
    auto model = c50::model::train(context, kind, names, dataset);
    c50::options options;
    for (unsigned trials : {0u, 1001u}) {
        options.trials = trials;
        rejects(invalid, [&] { c50::model::train(context, kind, names, dataset, options); });
    }
    for (double c50::options::*member : {&c50::options::minimum_cases,
         &c50::options::confidence_factor, &c50::options::sample_fraction}) {
        for (double value : {-1.0, 1000001.0, std::numeric_limits<double>::quiet_NaN()}) {
            options = {};
            options.*member = value;
            rejects(invalid, [&] { c50::model::train(context, kind, names, dataset, options); });
        }
    }
    options = {};
    options.random_seed = 4096;
    rejects(invalid, [&] { c50::model::train(context, kind, names, dataset, options); });
    auto bad = dataset;
    bad.row_stride = 1;
    rejects(invalid, [&] { c50::model::train(context, kind, names, bad); });
    bad = dataset; bad.class_indices = nullptr;
    rejects(invalid, [&] { c50::model::train(context, kind, names, bad); });
    bad = dataset; bad.values = nullptr;
    rejects(invalid, [&] { model.predict(context, bad); });
    bad = dataset; bad.row_count = std::numeric_limits<std::size_t>::max();
    rejects(invalid, [&] { model.predict(context, bad); });
    bad = dataset; bad.row_stride = std::numeric_limits<std::size_t>::max();
    rejects(invalid, [&] { model.predict(context, bad); });
    bad = dataset; bad.feature_count = 1;
    rejects(invalid, [&] { model.predict(context, bad); });
    values[1] = 2;
    rejects(invalid, [&] { model.predict(context, dataset); });
    values[1] = 0; values[0] = std::numeric_limits<double>::infinity();
    rejects(invalid, [&] { model.predict(context, dataset); });
    values[0] = 0; classes[0] = 2;
    rejects(invalid, [&] { c50::model::train(context, kind, names, dataset); });
    classes[0] = 0;
    rejects(c50::error_code::unsupported, [&] {
        c50::model::train(context, kind, "low, high.\nsignal: continuous.\ngroup: discrete 2.\n", dataset);
    });
    bad = dataset; bad.row_count = 0; bad.values = nullptr;
    auto empty = model.predict(context, bad);
    require(empty.size() == 0 && empty.class_count() == 2, "empty dense batch");
    rejects(invalid, [&] { c50::model::train(context, kind, names, bad); });
}

static void boosted_missing_values()
{
    const std::string schema = "a, b, c, d.\nx: continuous.\ny: continuous.\nz: continuous.\n";
    const std::string data = "?, ?, -1.5, a\n?, ?, ?, b\n?, 0.5, -3, c\n?, 4, ?, d\n"
                             "0, 0.5, ?, a\n-4, -3, ?, b\n-3, -1.5, 2.5, c\n-1.5, ?, -3, d\n";
    c50::context context;
    c50::options options;
    options.trials = 3;
    options.minimum_cases = 1;
    auto model = c50::model::train(context, c50::model_kind::rules, schema, data, options);
    auto result = model.predict(context, data);
    require(result.size() == 8 && result.class_count() == 4, "boosted missing shape");
    for (std::size_t row = 0; row < result.size(); ++row) {
        require(std::isfinite(result.confidence(row)), "boosted missing confidence");
        for (std::size_t index = 0; index < result.class_count(); ++index)
            require(std::isfinite(result.score(row, index)), "boosted missing score");
    }
}
int main()
{
    try { model_validation(); continuous_text_values(); training_validation(); boosted_missing_values(); }
    catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
