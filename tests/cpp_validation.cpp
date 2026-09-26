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
    try { boosted_missing_values(); }
    catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
