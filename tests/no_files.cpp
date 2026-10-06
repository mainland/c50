/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <c50/c50.hpp>
#include <cstdio>
#include <iostream>
#include <string>

static int file_attempts = 0;
extern "C" FILE *__wrap_tmpfile()
{
    ++file_attempts;
    return nullptr;
}
extern "C" FILE *__wrap_fopen(const char *, const char *)
{
    ++file_attempts;
    return nullptr;
}

int main()
{
    try {
        const std::string names = "no, yes.\nx: continuous.\ny: a, b.\n";
        std::string training;
        for (int row = 0; row < 40; ++row)
            training += std::to_string(row) + (row % 2 ? ", b, " : ", a, ") +
                        (row < 20 ? "no\n" : "yes\n");
        const std::string cases = "0, a, ?\n39, b, ?\n";
        c50::context context;
        for (const auto kind : {c50::model_kind::tree, c50::model_kind::rules}) {
            c50::options options;
            options.winnow = true;
            options.trials = 3;
            const auto model = c50::model::train(context, kind, names, training,
                                                options, "no, yes: 2\n");
            const auto restored = c50::model::load(context, kind, names,
                model.serialized_data(), model.costs_data());
            const auto predicted = restored.predict(context, cases);
            const auto original = model.predict(context, cases);
            if (predicted.size() != 2 ||
                predicted.class_index(0) != original.class_index(0) ||
                predicted.class_index(1) != original.class_index(1) ||
                predicted.confidence(0) != original.confidence(0) ||
                predicted.confidence(1) != original.confidence(1) ||
                file_attempts != 0) {
                std::cerr << "in-memory model round trip failed; file attempts: "
                          << file_attempts << '\n';
                return 1;
            }
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
