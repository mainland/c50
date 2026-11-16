/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <c50/c50.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

static const std::string names = "no, yes.\nx: continuous.\n";
static const std::string data = "0, no\n1, no\n2, yes\n3, yes\n";

int main()
{
    try {
        c50::context context;
        auto large = c50::model::load(context, c50::model_kind::tree, names,
            "costs=\"1\"\nentries=\"1\"\ntype=\"0\" class=\"no\" freq=\"1,1\"\n",
            "no, yes: 3e38\nyes, no: 2.5e38\n");
        if (large.predict(context, "0, ?\n").class_index(0) != 1)
            throw std::runtime_error("large expected costs ignored");
        for (auto kind : {c50::model_kind::tree, c50::model_kind::rules}) {
            for (const auto &costs : {"no, yes: 0\nyes, no: 0\n", "no, yes: 0\n",
                "no, yes: 1e100\n", "no, yes: inf\n", "no, yes: nan\n",
                "no, yes: 2junk\n", "no, yes: -1\n"}) {
                try {
                    c50::model::train(context, kind, names, data, {}, costs);
                    throw std::runtime_error("invalid costs accepted");
                } catch (const c50::exception &error) {
                    if (error.code() != c50::error_code::parse_error) throw;
                }
            }
            // Positive binary costs still round-trip and predict.
            for (const auto &costs : {"no, yes: 2\n", "no, yes: 5\n"}) {
                auto model = c50::model::train(context, kind, names, data, {}, costs);
                auto result = model.predict(context, data);
                for (std::size_t row = 0; row < result.size(); ++row) {
                    if (result.class_index(row) >= 2 || !std::isfinite(result.confidence(row)))
                        throw std::runtime_error("nonfinite cost prediction");
                    for (std::size_t index = 0; index < 2; ++index)
                        if (!std::isfinite(result.score(row, index)))
                            throw std::runtime_error("nonfinite cost score");
                }
            }
            // A multiclass zero entry is valid when its column has positive cost.
            auto multi = c50::model::train(context, kind,
                "a, b, c.\nx: continuous.\n",
                "0, a\n1, a\n2, b\n3, b\n4, c\n5, c\n", {}, "a, b: 0\n");
            auto multi_result = multi.predict(context, "0, ?\n3, ?\n5, ?\n");
            for (std::size_t row = 0; row < multi_result.size(); ++row)
                if (multi_result.class_index(row) >= 3 || !std::isfinite(multi_result.confidence(row)))
                    throw std::runtime_error("invalid multiclass zero-cost prediction");
            // Every case in one of three classes leaves the rule cost
            // normalization at 0/0. Release 2.07 forms no rules and writes
            // that class as the default, so training must succeed.
            auto single = c50::model::train(context, kind,
                "a, b, c.\nx: continuous.\n", "0, b\n1, b\n2, b\n3, b\n",
                {}, "a, b: 2\nc, a: 3\n");
            auto single_result = single.predict(context, "0, ?\n9, ?\n");
            for (std::size_t row = 0; row < single_result.size(); ++row)
                if (single_result.class_index(row) != 1)
                    throw std::runtime_error("single-class costs prediction");
            if (kind == c50::model_kind::rules &&
                single.serialized_data().find("rules=\"0\" default=\"b\"") ==
                    std::string::npos)
                throw std::runtime_error("single-class costs formed rules");
            c50::options options;
            options.ignore_costs = true;
            auto ignored = c50::model::train(context, kind, names, data, options,
                                             "no, yes: 0\nyes, no: 0\n");
            auto ordinary = c50::model::train(context, kind, names, data);
            if (!ignored.costs_data().empty() ||
                ignored.serialized_data() != ordinary.serialized_data())
                throw std::runtime_error("ignored costs changed training");
        }
        // Two-class costs scale the pessimistic error estimate of a leaf by
        // the other class's weight multiplier. In both datasets the estimate
        // for the root's subtree then exceeds the number of cases, and neither
        // branch of the root is a subtree. Release 2.07 replaced the root with
        // a branch that does not exist and crashed: in the final tree for the
        // first dataset, and in winnowing's trial tree for the second. The
        // root's split must stay.
        c50::options pessimistic;
        pessimistic.confidence_factor = 0.05;
        for (auto kind : {c50::model_kind::tree, c50::model_kind::rules}) {
            auto model = c50::model::train(context, kind,
                "low, high.\nx: continuous.\n",
                "?, low\n-3.5, high\n?, low\n0.5, low\n0, low\n?, low\n",
                pessimistic, "low, high: 5\n");
            auto result = model.predict(context, "-3.5, ?\n0.5, ?\n");
            if (result.class_index(0) != 1 || result.class_index(1) != 0)
                throw std::runtime_error("costs pruned the root's split");
        }
        c50::options winnowing = pessimistic;
        winnowing.winnow = true;
        auto winnowed = c50::model::train(context, c50::model_kind::tree,
            "a, b.\nx: continuous.\n",
            "?, b\n?, b\n0, b\n?, b\n1, b\n-1, a\n?, a\n?, b\n2, b\n?, b\n",
            winnowing, "b, a: 3\n");
        auto winnowed_result = winnowed.predict(context, "-1, ?\n2, ?\n");
        if (winnowed_result.class_index(0) != 0 ||
            winnowed_result.class_index(1) != 1)
            throw std::runtime_error("winnowing with costs pruned the split");
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
