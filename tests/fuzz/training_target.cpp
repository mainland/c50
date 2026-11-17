/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "training_target.hpp"

#include <c50/c50.hpp>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// The decoder reads fields in a fixed order, one byte each unless noted, and
// reads zero past the end of the input. tests/fuzz/README.md documents the
// format.
class reader {
public:
    reader(const std::uint8_t *bytes, std::size_t size)
        : bytes_(bytes), size_(size)
    {
    }

    unsigned next()
    {
        return position_ < size_ ? bytes_[position_++] : 0u;
    }

    // The next byte reduced to [0, count).
    unsigned below(unsigned count) { return next() % count; }

private:
    const std::uint8_t *bytes_;
    std::size_t size_;
    std::size_t position_ = 0;
};

enum class state { known, missing, not_applicable };

struct problem {
    c50::model_kind kind = c50::model_kind::tree;
    c50::options settings;
    std::string names;
    std::string costs;
    std::string text_training;
    std::string text_cases;
    std::size_t rows = 0;
    std::size_t features = 0;
    std::vector<double> values;
    std::vector<unsigned char> not_applicable;
    std::vector<std::size_t> classes;
};

std::string number(double value)
{
    // Multiples of one half in [-4, 4] print exactly.
    const bool negative = value < 0;
    const double magnitude = negative ? -value : value;
    const auto whole = static_cast<unsigned>(magnitude);
    std::string text = negative ? "-" : "";
    text += std::to_string(whole);
    if (magnitude != static_cast<double>(whole)) text += ".5";
    return text;
}

problem decode(const std::uint8_t *bytes, std::size_t size)
{
    reader in(bytes, size);
    problem result;

    result.kind = in.below(2) ? c50::model_kind::rules : c50::model_kind::tree;
    const unsigned flags = in.next();
    result.settings.subset_splits = (flags & 1) != 0;
    result.settings.winnow = (flags & 2) != 0;
    result.settings.global_pruning = (flags & 4) != 0;
    result.settings.probabilistic_thresholds = (flags & 8) != 0;
    result.settings.trials = 1 + in.below(3);
    constexpr std::array<double, 3> minimum_cases{1.0, 2.0, 3.0};
    result.settings.minimum_cases = minimum_cases[in.below(3)];
    constexpr std::array<double, 3> confidence{0.05, 0.25, 0.5};
    result.settings.confidence_factor = confidence[in.below(3)];

    const unsigned class_count = 2 + in.below(3);
    const unsigned continuous_count = 1 + in.below(3);
    const unsigned categorical_count = in.below(3);
    const unsigned rows = 1 + in.below(24);

    const unsigned cost_choice = in.below(class_count * (class_count - 1) + 1);
    const unsigned cost = 2 + in.below(8);
    constexpr std::array<double, 3> fractions{0.0, 0.5, 0.75};
    const double fraction = fractions[in.below(3)];
    // Read the high byte first. The operands of `|` are unsequenced.
    const unsigned seed_high = in.next();
    const unsigned seed_low = in.next();
    const unsigned seed = (seed_high << 8 | seed_low) % 4096;
    if (rows >= 12 && fraction != 0) {
        result.settings.sample_fraction = fraction;
        result.settings.random_seed = seed;
    }

    std::vector<unsigned> level_counts;
    for (unsigned index = 0; index < categorical_count; ++index)
        level_counts.push_back(2 + in.below(3));

    std::vector<std::string> classes;
    for (unsigned index = 0; index < class_count; ++index)
        classes.push_back("class_" + std::to_string(index));
    for (unsigned index = 0; index < class_count; ++index)
        result.names += classes[index] + (index + 1 < class_count ? ", " : ".\n");
    result.names += "\n";
    for (unsigned index = 0; index < continuous_count; ++index)
        result.names += "feature_" + std::to_string(index) + ": continuous.\n";
    for (unsigned index = 0; index < categorical_count; ++index) {
        result.names +=
            "feature_" + std::to_string(continuous_count + index) + ": ";
        for (unsigned level = 0; level < level_counts[index]; ++level)
            result.names += "value_" + std::to_string(level) +
                            (level + 1 < level_counts[index] ? ", " : ".\n");
    }

    if (cost_choice != 0) {
        const unsigned pair = cost_choice - 1;
        const unsigned predicted = pair / (class_count - 1);
        const unsigned rest = pair % (class_count - 1);
        const unsigned actual = rest >= predicted ? rest + 1 : rest;
        result.costs = classes[predicted] + ", " + classes[actual] + ": " +
                       std::to_string(cost) + "\n";
    }

    result.rows = rows;
    result.features = continuous_count + categorical_count;
    for (unsigned row = 0; row < rows; ++row) {
        std::string fields;
        for (unsigned feature = 0; feature < result.features; ++feature) {
            const bool continuous = feature < continuous_count;
            const unsigned levels =
                continuous ? 0 : level_counts[feature - continuous_count];
            const unsigned code =
                in.below(continuous ? 19u : levels + 2);
            state kind = state::known;
            double value = 0;
            std::string text;
            if (code == 0) {
                kind = state::missing;
            } else if (code == 1) {
                kind = state::not_applicable;
            } else if (continuous) {
                value = (static_cast<int>(code) - 10) / 2.0;
                text = number(value);
            } else {
                value = code - 2;
                text = "value_" + std::to_string(code - 2);
            }
            switch (kind) {
            case state::known:
                result.values.push_back(value);
                result.not_applicable.push_back(0);
                break;
            case state::missing:
                text = "?";
                result.values.push_back(
                    std::numeric_limits<double>::quiet_NaN());
                result.not_applicable.push_back(0);
                break;
            case state::not_applicable:
                text = "N/A";
                // The library ignores the value under a set mask byte, so
                // alternate between NaN and a value outside every schema.
                result.values.push_back(
                    (row + feature) % 2 ? std::numeric_limits<double>::quiet_NaN()
                                        : 1e300);
                result.not_applicable.push_back(1);
                break;
            }
            fields += text + ", ";
        }
        const unsigned label = in.below(class_count);
        result.classes.push_back(label);
        result.text_training += fields + classes[label] + "\n";
        result.text_cases += fields + "?\n";
    }
    return result;
}

[[noreturn]] void violation(const std::string &message)
{
    throw std::runtime_error("property violated: " + message);
}

// Require identical labels, confidences, and class scores.
void same_predictions(const char *what, const c50::predictions &left,
                      const c50::predictions &right)
{
    const std::string prefix = std::string(what) + ": ";
    if (left.size() != right.size()) violation(prefix + "row counts differ");
    if (left.class_count() != right.class_count())
        violation(prefix + "class counts differ");
    for (std::size_t index = 0; index < left.class_count(); ++index)
        if (left.class_name(index) != right.class_name(index))
            violation(prefix + "class names differ");
    for (std::size_t row = 0; row < left.size(); ++row) {
        if (left.class_index(row) != right.class_index(row))
            violation(prefix + "labels differ at row " + std::to_string(row));
        if (!std::isfinite(left.confidence(row)) ||
            !std::isfinite(right.confidence(row)))
            violation(prefix + "confidence is not finite");
        if (left.confidence(row) != right.confidence(row))
            violation(prefix + "confidences differ at row " +
                      std::to_string(row));
        for (std::size_t index = 0; index < left.class_count(); ++index) {
            if (!std::isfinite(left.score(row, index)) ||
                !std::isfinite(right.score(row, index)))
                violation(prefix + "score is not finite");
            if (left.score(row, index) != right.score(row, index))
                violation(prefix + "scores differ at row " +
                          std::to_string(row));
        }
    }
}

// The first line of a serialized classifier names its creation date.
std::string stable_content(const c50::model &model)
{
    const std::string &text = model.serialized_data();
    const auto newline = text.find('\n');
    if (newline == std::string::npos)
        violation("serialized classifier has no first line");
    return text.substr(newline + 1);
}

}

namespace c50::test {

void exercise_training(const std::uint8_t *bytes, std::size_t size)
{
    if (size > max_training_input_size) return;
    const problem input = decode(bytes, size);

    const c50::dense_dataset training(
        input.values.data(), input.rows, input.features,
        input.classes.data(), 0, input.not_applicable.data());
    const c50::dense_dataset cases(
        input.values.data(), input.rows, input.features, nullptr, 0,
        input.not_applicable.data());

    c50::context context;
    const auto text_model = c50::model::train(
        context, input.kind, input.names, input.text_training, input.settings,
        input.costs);
    const auto dense_model = c50::model::train(
        context, input.kind, input.names, training, input.settings,
        input.costs);

    if (stable_content(dense_model) != stable_content(text_model))
        violation("the interfaces trained different classifiers");
    if (dense_model.costs_data() != text_model.costs_data())
        violation("the interfaces kept different costs");

    const auto text_predictions =
        text_model.predict(context, input.text_cases);
    const auto dense_predictions = dense_model.predict(context, cases);
    same_predictions("dense and text predictions", dense_predictions,
                     text_predictions);

    const auto loaded_text = c50::model::load(
        context, text_model.kind(), text_model.names_data(),
        text_model.serialized_data(), text_model.costs_data());
    const auto loaded_dense = c50::model::load(
        context, dense_model.kind(), dense_model.names_data(),
        dense_model.serialized_data(), dense_model.costs_data());
    same_predictions("text model after load",
                     loaded_text.predict(context, input.text_cases),
                     text_predictions);
    same_predictions("dense model after load",
                     loaded_dense.predict(context, cases), dense_predictions);
}

}
