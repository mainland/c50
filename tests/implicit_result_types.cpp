/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

// A defined attribute has a numeric or logical value. A definition whose
// value is a string has no representation as a case value, and training took
// the value from the address of the string. The definition must be rejected,
// and the date, time, and logical forms must still train the same model every
// time.

#include <c50/c50.hpp>

#include "model_text.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const std::string head =
    "class.\n"
    "color: red, green, blue.\n"
    "grade: [ordered] low, medium, high.\n"
    "d: date.\n"
    "t: time.\n"
    "dyn: discrete 5.\n"
    "size: continuous.\n";

const std::string tail = "class: yes, no.\nattributes included: x.\n";

std::string data()
{
    const char *colors[] = {"red", "green", "blue"};
    const char *grades[] = {"low", "medium", "high"};
    const char *dates[] = {"2020/01/01", "2021/06/15", "2022/12/31"};
    const char *times[] = {"01:00:00", "12:30:00", "23:59:59"};
    const char *dynamic[] = {"u", "v", "w"};
    std::string text;
    for (int row = 0; row < 90; ++row) {
        const int kind = row % 3;
        const double size = (row * 7 % 100) / 10.0;
        text += std::string(colors[kind]) + "," + grades[kind] + "," +
                dates[kind] + "," + times[kind] + "," + dynamic[kind] + "," +
                std::to_string(size) + "," + (kind == 1 ? "yes" : "no") + "\n";
    }
    return text;
}

std::string schema(const std::string &definition)
{
    return head + definition + "\n" + tail;
}

void rejects(c50::context &context, const std::string &definition)
{
    for (const auto kind : {c50::model_kind::tree, c50::model_kind::rules}) {
        try {
            c50::model::train(context, kind, schema(definition), data());
        } catch (const c50::exception &error) {
            if (error.code() != c50::error_code::parse_error) throw;
            const std::string message = error.what();
            if (message.find("value must be a number or a logical value") ==
                std::string::npos)
                throw std::runtime_error(definition + ": wrong message: " +
                                         message);
            continue;
        }
        throw std::runtime_error(definition + ": string definition accepted");
    }
}

// Two contexts that parse the same definition must agree, which the pointer
// bits of a string never did. The model must also predict the training
// cases it was built from.
void trains_alike(const std::string &definition)
{
    const std::string cases = data();
    for (const auto kind : {c50::model_kind::tree, c50::model_kind::rules}) {
        c50::context first;
        c50::context second;
        const c50::model one =
            c50::model::train(first, kind, schema(definition), cases);
        const c50::model two =
            c50::model::train(second, kind, schema(definition), cases);
        if (without_stamp(one.serialized_data()) !=
            without_stamp(two.serialized_data()))
            throw std::runtime_error(definition + ": models differ");
        const c50::model loaded = c50::model::load(
            second, kind, schema(definition), one.serialized_data());
        const c50::predictions expected = one.predict(first, cases);
        const c50::predictions actual = loaded.predict(second, cases);
        if (expected.size() != actual.size())
            throw std::runtime_error(definition + ": prediction count");
        for (std::size_t row = 0; row < expected.size(); ++row)
            if (expected.class_index(row) != actual.class_index(row))
                throw std::runtime_error(definition + ": predictions differ");
    }
}

} // namespace

int main()
{
    try {
        c50::context context;
        for (const char *definition : {"x := \"abc\".", "x := color.",
                                       "x := (color).", "x := grade.",
                                       "x := dyn.",
                                       "b := size > 5.\nx := b."})
            rejects(context, definition);
        for (const char *definition :
             {"x := d.", "x := t.", "x := color = \"green\".",
              "x := color <> grade.",
              "x := size.", "x := size > 5."})
            trains_alike(definition);
        // The context that rejected a definition trains the next model.
        trains_alike("x := size > 5.");
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
