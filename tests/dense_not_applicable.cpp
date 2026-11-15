/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

// Check that dense input with a not-applicable mask trains the classifier,
// and gives the predictions and attribute usage, that the equivalent text
// input with "N/A" gives.

#include <c50/c50.hpp>
#include "model_text.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool value, const std::string &message)
{
    if (!value) throw std::runtime_error(message);
}

const char names_text[] =
    "lo, mid, hi.\n"
    "x: continuous.\n"
    "y: continuous.\n"
    "g: a, b, c.\n"
    "h: p, q, r, s, t.\n";
const char *const class_names[] = {"lo", "mid", "hi"};
const char *const g_names[] = {"a", "b", "c"};
const char *const h_names[] = {"p", "q", "r", "s", "t"};
const std::size_t features = 4;

/* Cells that the mask hides hold this kind of value. */
enum class filler { real, nan, invalid };

struct table {
    std::size_t rows = 0;
    std::vector<double> values;               // NaN is missing.
    std::vector<unsigned char> not_applicable; // Nonzero is N/A.
    std::vector<std::size_t> classes;
};

std::uint32_t next(std::uint32_t &state)
{
    state = state * 1664525u + 1013904223u;
    return state >> 8;
}

/* Build a table. The class depends on N/A as well as on values, so the
   learner has a reason to split on N/A. No N/A when with_na is false. */
table make_table(std::size_t rows, std::uint32_t seed, bool with_na,
                 bool every_feature_row)
{
    table result;
    std::uint32_t state = seed;
    result.rows = rows;
    for (std::size_t row = 0; row < rows; ++row) {
        const double x = (next(state) % 80) / 4.0;
        const double y = (next(state) % 80) / 4.0;
        const double g = static_cast<double>(next(state) % 3);
        const double h = static_cast<double>(next(state) % 5);
        bool na[features] = {false, false, false, false};
        bool missing[features] = {false, false, false, false};
        if (with_na) {
            na[0] = next(state) % 4 == 0;
            na[1] = next(state) % 6 == 0;
            na[2] = next(state) % 5 == 0;
            na[3] = next(state) % 4 == 0;
            if (every_feature_row && row % 17 == 3)
                na[0] = na[1] = na[2] = na[3] = true;
        }
        missing[0] = next(state) % 11 == 0;
        missing[3] = next(state) % 9 == 0;
        const double cell[features] = {x, y, g, h};
        for (std::size_t f = 0; f < features; ++f) {
            result.values.push_back(
                missing[f] ? std::numeric_limits<double>::quiet_NaN() : cell[f]);
            result.not_applicable.push_back(na[f] ? 1 : 0);
        }
        std::size_t label;
        if (na[0]) label = 0;
        else if (na[2]) label = 2;
        else if (h == 3 || h == 4) label = x > 10 ? 2 : 1;
        else label = x > 12 ? 1 : 0;
        if (with_na && na[3]) label = (label + 1) % 3;
        result.classes.push_back(label);
    }
    return result;
}

/* Replace the cells under the mask. */
table fill_masked(table data, filler kind)
{
    for (std::size_t row = 0; row < data.rows; ++row)
        for (std::size_t f = 0; f < features; ++f) {
            const std::size_t at = row * features + f;
            if (!data.not_applicable[at] || kind == filler::real) continue;
            if (kind == filler::nan)
                data.values[at] = std::numeric_limits<double>::quiet_NaN();
            else
                data.values[at] = f < 2 ? std::numeric_limits<double>::infinity()
                                        : (f == 2 ? -3.0 : 99.0);
        }
    return data;
}

std::string number(double value)
{
    std::ostringstream out;
    out << value;
    return out.str();
}

/* The text equivalent of a table. Labels are included when asked. */
std::string text_of(const table &data, bool labels)
{
    std::string text;
    for (std::size_t row = 0; row < data.rows; ++row) {
        for (std::size_t f = 0; f < features; ++f) {
            const std::size_t at = row * features + f;
            if (data.not_applicable[at]) text += "N/A";
            else if (std::isnan(data.values[at])) text += "?";
            else if (f < 2) text += number(data.values[at]);
            else if (f == 2) text += g_names[static_cast<std::size_t>(data.values[at])];
            else text += h_names[static_cast<std::size_t>(data.values[at])];
            text += ",";
        }
        text += labels ? class_names[data.classes[row]] : "?";
        text += "\n";
    }
    return text;
}

c50::dense_dataset dense_of(const table &data, bool use_mask, bool labels)
{
    return c50::dense_dataset(
        data.values.empty() ? nullptr : data.values.data(), data.rows,
        features, labels ? data.classes.data() : nullptr, 0,
        use_mask ? data.not_applicable.data() : nullptr);
}

void same_predictions(const c50::predictions &a, const c50::predictions &b,
                      const std::string &label)
{
    require(a.size() == b.size() && a.class_count() == b.class_count(),
            label + ": prediction shape differs");
    for (std::size_t row = 0; row < a.size(); ++row) {
        require(a.class_index(row) == b.class_index(row),
                label + ": label differs at row " + std::to_string(row));
        require(a.confidence(row) == b.confidence(row),
                label + ": confidence differs at row " + std::to_string(row));
        for (std::size_t k = 0; k < a.class_count(); ++k)
            require(a.score(row, k) == b.score(row, k),
                    label + ": score differs at row " + std::to_string(row));
    }
}

struct scenario {
    const char *label;
    c50::model_kind kind;
    c50::options settings;
    const char *costs;
    bool train_with_na;
    bool predict_with_na;
};

/* Count the splits on N/A in a model's serialized text, so that a scenario
   that never tests N/A cannot pass for the wrong reason. */
std::size_t na_tests(const std::string &text)
{
    std::size_t count = 0;
    for (std::size_t at = text.find("N/A"); at != std::string::npos;
         at = text.find("N/A", at + 1))
        ++count;
    return count;
}

void run(const scenario &s, filler kind, std::size_t &na_models)
{
    const std::string label =
        std::string(s.label) + " filler " + std::to_string(static_cast<int>(kind));
    const table train_real = make_table(160, 12345u, s.train_with_na, true);
    const table train_masked = fill_masked(train_real, kind);
    const table cases_real = make_table(70, 777u, s.predict_with_na, true);
    const table cases_masked = fill_masked(cases_real, kind);

    c50::context text_workspace, dense_workspace;
    const auto text_model = c50::model::train(
        text_workspace, s.kind, names_text, text_of(train_real, true),
        s.settings, s.costs);
    const auto dense_model = c50::model::train(
        dense_workspace, s.kind, names_text, dense_of(train_masked, true, true),
        s.settings, s.costs);
    require(without_stamp(text_model.serialized_data()) ==
                without_stamp(dense_model.serialized_data()),
            label + ": serialized models differ");
    if (na_tests(text_model.serialized_data())) ++na_models;

    const std::string cases_text = text_of(cases_real, false);
    c50::context workspace;
    const auto expected = text_model.predict(workspace, cases_text);
    same_predictions(expected,
                     text_model.predict(workspace, dense_of(cases_masked, true, false)),
                     label + ": model prediction");
    same_predictions(expected,
                     dense_model.predict(workspace, dense_of(cases_masked, true, false)),
                     label + ": dense model prediction");
    auto predictor = dense_model.prepare_predictor();
    same_predictions(expected,
                     predictor.predict(dense_of(cases_masked, true, false)),
                     label + ": predictor prediction");
    require(text_model.attribute_usage(workspace, cases_text) ==
                text_model.attribute_usage(workspace,
                                           dense_of(cases_masked, true, false)),
            label + ": attribute usage differs");
}

void all_false_mask_equals_no_mask()
{
    table data = make_table(120, 99u, false, false);
    c50::context a, b;
    c50::options settings;
    settings.trials = 3;
    const auto without = c50::model::train(a, c50::model_kind::tree, names_text,
                                           dense_of(data, false, true), settings);
    const auto with = c50::model::train(b, c50::model_kind::tree, names_text,
                                        dense_of(data, true, true), settings);
    require(without_stamp(without.serialized_data()) ==
                without_stamp(with.serialized_data()),
            "an all-false mask changes the model");
    same_predictions(without.predict(a, dense_of(data, false, false)),
                     with.predict(a, dense_of(data, true, false)),
                     "all-false mask");
}

void mask_controls_nan_only_where_set()
{
    // NaN under a zero mask byte stays missing, as without a mask.
    table data = make_table(120, 5u, true, false);
    table nan_data = fill_masked(data, filler::nan);
    c50::context a, b;
    const auto masked = c50::model::train(
        a, c50::model_kind::tree, names_text, dense_of(nan_data, true, true));
    table no_na = nan_data;
    for (auto &byte : no_na.not_applicable) byte = 0;
    const auto missing = c50::model::train(
        b, c50::model_kind::tree, names_text, dense_of(no_na, true, true));
    require(without_stamp(masked.serialized_data()) !=
                without_stamp(missing.serialized_data()),
            "N/A and missing give the same model");
}

void borrowed_mask_with_row_stride()
{
    // The mask is contiguous even when values have a row stride.
    const table data = make_table(60, 31u, true, false);
    std::vector<double> padded;
    for (std::size_t row = 0; row < data.rows; ++row) {
        for (std::size_t f = 0; f < features; ++f)
            padded.push_back(data.values[row * features + f]);
        padded.push_back(-1);
        padded.push_back(-2);
    }
    c50::context a, b;
    const auto contiguous = c50::model::train(
        a, c50::model_kind::tree, names_text, dense_of(data, true, true));
    const c50::dense_dataset strided(padded.data(), data.rows, features,
                                     data.classes.data(), features + 2,
                                     data.not_applicable.data());
    const auto padded_model =
        c50::model::train(b, c50::model_kind::tree, names_text, strided);
    require(without_stamp(contiguous.serialized_data()) ==
                without_stamp(padded_model.serialized_data()),
            "a strided values array changes the masked model");
}

void member_assignment()
{
    const table data = make_table(10, 3u, true, false);
    c50::dense_dataset set;
    require(set.not_applicable == nullptr, "mask defaults to null");
    set.not_applicable = data.not_applicable.data();
    require(set.not_applicable == data.not_applicable.data(), "mask assigns");
}
} // namespace

int main()
{
    try {
        c50::options plain;
        c50::options boosted;
        boosted.trials = 4;
        c50::options subsets;
        subsets.subset_splits = true;
        c50::options winnowed;
        winnowed.winnow = true;
        c50::options everything;
        everything.trials = 3;
        everything.subset_splits = true;
        everything.winnow = true;
        everything.global_pruning = false;
        c50::options sampled;
        sampled.sample_fraction = 0.7;
        sampled.random_seed = 17;

        const char *costs = "mid, lo: 3\nhi, lo: 2\n";
        const scenario scenarios[] = {
            {"tree", c50::model_kind::tree, plain, "", true, true},
            {"rules", c50::model_kind::rules, plain, "", true, true},
            {"boosted tree", c50::model_kind::tree, boosted, "", true, true},
            {"boosted rules", c50::model_kind::rules, boosted, "", true, true},
            {"subset tree", c50::model_kind::tree, subsets, "", true, true},
            {"subset rules", c50::model_kind::rules, subsets, "", true, true},
            {"winnowed tree", c50::model_kind::tree, winnowed, "", true, true},
            {"winnowed rules", c50::model_kind::rules, winnowed, "", true, true},
            {"everything", c50::model_kind::tree, everything, "", true, true},
            {"sampled", c50::model_kind::tree, sampled, "", true, true},
            {"costs tree", c50::model_kind::tree, plain, costs, true, true},
            {"costs rules", c50::model_kind::rules, plain, costs, true, true},
            // Trained without N/A, classified with it.
            {"prediction only tree", c50::model_kind::tree, plain, "", false, true},
            {"prediction only rules", c50::model_kind::rules, plain, "", false, true},
            {"prediction only boosted", c50::model_kind::tree, boosted, "", false, true},
        };
        std::size_t na_models = 0;
        for (const auto &s : scenarios)
            for (filler kind : {filler::real, filler::nan, filler::invalid})
                run(s, kind, na_models);
        require(na_models > 20, "too few scenarios test N/A");
        all_false_mask_equals_no_mask();
        mask_controls_nan_only_where_set();
        borrowed_mask_with_row_stride();
        member_assignment();
    } catch (const std::exception &error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
    return 0;
}
