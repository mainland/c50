/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "parser_target.hpp"

#include <c50/c50.hpp>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

constexpr std::string_view names =
    "no, yes.\nx: continuous.\ncolor: red, blue, green.\n";
constexpr std::string_view leaf =
    "entries=\"1\"\ntype=\"0\" class=\"no\" freq=\"2,0\"\n";
constexpr std::string_view data =
    "0, red, no\n1, blue, no\n2, green, yes\n3, red, yes\n";

bool expected(const c50::exception &error)
{
    return error.code() == c50::error_code::parse_error ||
           error.code() == c50::error_code::invalid_argument ||
           error.code() == c50::error_code::io_error ||
           error.code() == c50::error_code::unsupported;
}

}

namespace c50::test {

void exercise_parser(fuzz_kind kind, const std::uint8_t *bytes,
                     std::size_t size)
{
    if (size > 4096) return;
    const char *characters = size ? reinterpret_cast<const char *>(bytes) : "";
    const std::string_view input(characters, size);
    c50::context context;
    try {
        switch (kind) {
        case fuzz_kind::schema:
            c50::model::load(context, c50::model_kind::tree, input, leaf);
            break;
        case fuzz_kind::implicit: {
            const std::string schema =
                std::string(names) + "derived := " + std::string(input) + ".\n";
            c50::model::train(context, c50::model_kind::tree, schema, data);
            break;
        }
        case fuzz_kind::tree:
            c50::model::load(context, c50::model_kind::tree, names, input,
                             "no, yes: 2\n");
            break;
        case fuzz_kind::rules:
            c50::model::load(context, c50::model_kind::rules, names, input,
                             "no, yes: 2\n");
            break;
        case fuzz_kind::cases: {
            auto model = c50::model::load(
                context, c50::model_kind::tree, names, leaf);
            model.predict(context, input);
            break;
        }
        case fuzz_kind::costs:
            c50::model::train(context, c50::model_kind::tree, names, data, {},
                              input);
            break;
        default:
            throw std::invalid_argument("unknown parser fuzz target");
        }
    } catch (const c50::exception &error) {
        if (!expected(error)) throw;
    }

    // Acceptance and rejection must leave no parser state in the context.
    auto recovered =
        c50::model::load(context, c50::model_kind::tree, names, leaf);
    auto prediction = recovered.predict(context, "0, red, ?\n");
    if (prediction.size() != 1 || prediction.class_index(0) != 0)
        throw std::runtime_error("parser recovery changed prediction");
}

}
