/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <c50/c50.hpp>
#include <iostream>
#include <stdexcept>
#include <string>

static const std::string names = "no, yes.\nx: continuous.\n";
static const std::string header = "id=\"See5/C5.0 2.07 GPL Edition\"\nentries=\"1\"\n";
static const std::string leaf = "type=\"0\" class=\"no\" freq=\"2,0\"\n";

template<class F> static void rejects(F operation)
{
    try { operation(); }
    catch (const c50::exception &error) {
        if (error.code() != c50::error_code::parse_error ||
            std::string(error.what()).empty()) throw;
        return;
    }
    throw std::runtime_error("malformed input accepted");
}

static void structure(c50::context &context)
{
    const auto kind = c50::model_kind::tree;
    for (const auto &bad : {
        "type=\"2\" class=\"no\" att=\"x\" forks=\"1\" cut=\"0\" freq=\"1,1\"\n",
        "type=\"0\" freq=\"1,1\"\n",
        "class=\"no\" freq=\"1,1\"\n",
        "type=\"999999999999999999999\" class=\"no\"\n",
        "type=\"9\" class=\"no\"\n",
        "type=\"0junk\" class=\"no\"\n",
        "type=\"0\" type=\"0\" class=\"no\"\n",
        "type=\"0\" class=\"no\" freq=\"1\"\n",
        "type=\"0\" class=\"no\" freq=\"1,1,1\"\n",
        "type=\"0\" class=\"no\" freq=\"nan,1\"\n",
        "type=\"0\" class=\"no\" freq=\"-1,1\"\n",
        "type=\"0\" class=\"no\" freq=\"0,0\"\n",
        "type=\"0\" class=\"no\" freq=\"1e-30,0\"\n",
        "type=\"0\" class=\"no\" freq=\"1e100,1\"\n",
        "type=\"2\" class=\"no\" att=\"x\" forks=\"-1\" cut=\"0\"\n",
        "type=\"2\" class=\"no\" att=\"x\" forks=\"3\" cut=\"inf\"\n",
        "type=\"2\" class=\"no\" att=\"x\" forks=\"3\" cut=\"1e100\"\n",
        "type=\"3\" class=\"no\" att=\"x\" forks=\"2\" elts=\"a\"\n"})
        rejects([&] { c50::model::load(context, kind, names, header + bad); });
    for (const auto &prefix : {"att=\"x\" elts=\"a\",\"b\"\n",
                               "elts=\"a\",\"b\"\n", "entries=\"-1\"\n",
                               "entries=\"1001\"\n", "entries=\"one\"\n"})
        rejects([&] { c50::model::load(context, kind, names, prefix + header + leaf); });
    const std::string dynamic_names = "no, yes.\nx: discrete 2.\n";
    rejects([&] { c50::model::load(context, kind, dynamic_names,
        "att=\"x\" elts=\"a\",\"b\",\"c\"\n" + header + leaf); });
    const std::string discrete_names = "no, yes.\nx: a, b.\n";
    for (const auto &subsets : {"elts=\"a\"", "elts=\"a\" elts=\"unknown\"",
                               "elts=\"a\" elts=\"b\" elts=\"a\""})
        rejects([&] { c50::model::load(context, kind, discrete_names,
            header + "type=\"3\" class=\"no\" att=\"x\" forks=\"2\" freq=\"2,0\" " + subsets + "\n"); });
    const auto rules = c50::model_kind::rules;
    for (const auto &bad : {"rules=\"-1\" default=\"no\"\n", "rules=\"1\"\n",
        "rules=\"1\" default=\"no\"\nconds=\"-1\"\n",
        "rules=\"1\" default=\"no\"\nconds=\"0\" cover=\"1\" ok=\"1\" class=\"no\"\n",
        "rules=\"1\" default=\"no\"\nconds=\"0\" cover=\"1\" ok=\"2\" lift=\"1\" class=\"no\"\n",
        "rules=\"1\" default=\"no\"\nconds=\"0\" cover=\"1\" ok=\"1\" lift=\"0\" class=\"no\"\n"})
        rejects([&] { c50::model::load(context, rules, names, header + bad); });
    rejects([&] { c50::model::load(context, rules, names,
        "entries=\"1\"\nrules=\"1\" default=\"yes\"\n"
        "conds=\"100000000\" cover=\"0\" ok=\"0\" lift=\"1\" class=\"no\"\n"); });
    rejects([&] { c50::model::load(context, rules, names,
        "entries=\"1\"\nrules=\"100000000\" default=\"yes\"\n"); });
    std::string growing_rule = header + "rules=\"1\" default=\"no\"\n"
        "conds=\"101\" cover=\"1\" ok=\"1\" lift=\"1\" class=\"no\"\n";
    for (int condition = 0; condition < 101; ++condition)
        growing_rule += "type=\"2\" att=\"x\" cut=\"" +
                        std::to_string(condition) + "\" result=\">\"\n";
    c50::model::load(context, rules, names, growing_rule);
    std::string growing_rules = header + "rules=\"101\" default=\"no\"\n";
    for (int rule_index = 0; rule_index < 101; ++rule_index)
        growing_rules +=
            "conds=\"0\" cover=\"1\" ok=\"1\" lift=\"1\" class=\"no\"\n";
    c50::model::load(context, rules, names, growing_rules);
    rejects([&] { c50::model::load(context, rules, names,
        growing_rules + "trailing"); });
    const std::string rule = header + "rules=\"1\" default=\"no\"\n"
        "conds=\"1\" cover=\"1\" ok=\"1\" lift=\"1\" class=\"no\"\n";
    for (const auto &condition : {"type=\"0\" att=\"x\"\n",
        "type=\"2\" att=\"x\" cut=\"0\" result=\"bad\"\n",
        "type=\"2\" val=\"N/A\" att=\"x\"\n",
        "type=\"2\" att=\"x\" val=\"bad\"\n",
        "type=\"3\" att=\"x\" elts=\"a\"\n"})
        rejects([&] { c50::model::load(context, rules, names, rule + condition); });
    const std::string duplicate_condition =
        "type=\"2\" att=\"x\" cut=\"0\" result=\">\"\n";
    rejects([&] { c50::model::load(context, rules, names,
        header + "rules=\"1\" default=\"no\"\n"
        "conds=\"2\" cover=\"1\" ok=\"1\" lift=\"1\" class=\"no\"\n" +
        duplicate_condition + duplicate_condition); });

    // Zero-case child nodes are valid legacy output and use their parent.
    auto empty_child = c50::model::load(context, kind, names, header +
        "type=\"2\" class=\"no\" att=\"x\" forks=\"3\" cut=\"1\" freq=\"2,0\"\n" +
        "type=\"0\" class=\"no\"\n" + leaf + "type=\"0\" class=\"no\"\n");
    if (empty_child.predict(context, "5, ?\n").class_index(0) != 0)
        throw std::runtime_error("empty child prediction changed");

    rejects([&] { c50::model::load(context, kind, names, header +
        "type=\"2\" class=\"no\" att=\"x\" cut=\"1\" forks=\"3\" freq=\"2,0\"\n" +
        "type=\"0\" class=\"no\"\n" +
        "type=\"2\" class=\"no\" att=\"x\" cut=\"0\" forks=\"3\"\n" +
        "type=\"0\" class=\"no\"\n" + "type=\"0\" class=\"no\"\n" +
        "type=\"0\" class=\"no\"\n" + leaf); });

    const auto zero_vote = c50::model::load(context, rules, names, header +
        "rules=\"1\" default=\"yes\"\n"
        "conds=\"0\" cover=\"10000\" ok=\"0\" lift=\"1\" class=\"no\"\n");
    const auto default_prediction = zero_vote.predict(context, "0, ?\n");
    if (default_prediction.class_index(0) != 1 ||
        default_prediction.confidence(0) != 0.5)
        throw std::runtime_error("zero-vote rules did not use the default class");
}

int main()
{
    try {
        c50::context context;
        const auto kind = c50::model_kind::tree;
        auto model = c50::model::load(context, kind, names, header + leaf);
        rejects([&] { c50::model::load(context, kind, names,
            header + leaf + "trailing"); });
        c50::model::load(context, kind, names, header + leaf + " \t\n");
        rejects([&] { c50::model::load(context, kind,
            "no, yes.\n" + std::string(2000, 'x') + ": continuous.\n", header + leaf); });
        rejects([&] { model.predict(context, std::string(2000, '1') + ", ?\n"); });
        rejects([&] { c50::model::load(context, kind, names,
            header + "type=\"0\" class=\"" + std::string(30000, 'x') + "\" freq=\"2,0\"\n"); });
        for (const auto &bad : {
            "type=\"0\" class=no freq=\"2,0\"\n",
            "type=\"0\" class=\"no\"junk freq=\"2,0\"\n",
            "type=\"0\" class=\"no\",\"yes\" freq=\"2,0\"\n",
            "type=\"0\" class=\"no\\",
            "type=\"0\" class=\"no\n"})
            rejects([&] { c50::model::load(context, kind, names, header + bad); });
        auto short_id = c50::model::load(context, kind, names,
                                         "id=\"x\"\nentries=\"1\"\n" + leaf);
        if (short_id.predict(context, "0, ?\n").class_index(0) != 0) return 1;
        for (const auto &capacity : {"-1", "1", "2147483647", "999999999999999999999", "2junk"})
            rejects([&] { c50::model::load(context, kind,
                std::string("no, yes.\nx: discrete ") + capacity + ".\n", header + leaf); });
        const std::string target(995, 'x');
        rejects([&] { c50::model::load(context, kind,
            target + ": 1, 2.\n" + target + ": continuous.\n", header + leaf); });
        c50::model::load(context, kind,
            "|" + std::string(9997, 'x') + "\n" + names, header + leaf);
        const auto long_line_model = c50::model::load(context, kind,
            "|" + std::string(20000, 'x') + "\n" + names, header + leaf);
        const auto long_line_predictions = long_line_model.predict(context,
            "0, ?\n" + std::string(20000, ' ') + "1, ?\n");
        if (long_line_predictions.size() != 2 ||
            long_line_predictions.class_index(0) != 0 ||
            long_line_predictions.class_index(1) != 0)
            throw std::runtime_error("long input lines changed predictions");
        for (const auto &threshold : {"nan", "inf", "-inf"})
            rejects([&] { c50::model::load(context, kind,
                "y: 1, " + std::string(threshold) + ".\ny: continuous.\nx: continuous.\n",
                header + leaf); });
        for (const auto &value : {"?", "N/A"})
            rejects([&] { c50::model::load(context, kind,
                "no, yes.\nx: continuous.\nderived := " + std::string(value) + ".\n",
                header + leaf); });
        const auto nested_expression = [](int depth) {
            std::string expression = "x";
            for (int level = 1; level < depth; ++level)
                expression = "1 + (" + expression + ")";
            return expression;
        };
        const auto boundary_model = c50::model::load(context, kind,
            "no, yes.\nx: continuous.\nderived := " + nested_expression(100) + ".\n",
            header + leaf);
        if (boundary_model.predict(context, "1, ?\n").class_index(0) != 0)
            throw std::runtime_error("expression depth boundary changed prediction");
        rejects([&] { c50::model::load(context, kind,
            "no, yes.\nx: continuous.\nderived := " + nested_expression(101) + ".\n",
            header + leaf); });
        const std::string boundary_expression =
            "color = \"" + std::string(38, 'a') + "\"";
        if (boundary_expression.size() != 48) return 1;
        c50::model::load(context, kind,
            "no, yes.\nx: continuous.\ncolor: red, blue, green.\nderived := " +
                boundary_expression + ".\n",
            header + leaf);
        c50::model::train(context, kind,
            "no, yes.\nx: continuous.\ncolor: red, blue, green.\n"
            "derived := x / 2e22.\n",
            "0, red, no\n1, blue, no\n2, green, yes\n3, red, yes\n");
        c50::model::train(context, kind,
            "no, yes.\nx: continuous.\nderived := x / (8 ^ 8 ^ 8).\n",
            "1, no\n2, yes\n3, no\n4, yes\n");
        rejects([&] { c50::model::load(context, kind,
            "no, yes.\nx: continuous.\nderived := 1e1000.\n", header + leaf); });
        c50::model::load(context, kind,
            "no, yes.\nx: discrete 472721496.\n", header + leaf);
        const std::string growing_names = "no, yes.\nx: discrete 101.\n";
        std::string growing_header = "att=\"x\" elts=";
        std::string growing_data;
        for (int value = 0; value < 101; ++value) {
            if (value) growing_header += ',';
            growing_header += "\"v" + std::to_string(value) + "\"";
            growing_data +=
                "v" + std::to_string(value) + (value < 51 ? ", no\n" : ", yes\n");
        }
        c50::model::load(context, kind, growing_names,
                         growing_header + "\n" + header + leaf);
        c50::model::train(context, kind, growing_names, growing_data);
        structure(context);
        if (model.predict(context, "0, ?\n").class_index(0) != 0) return 1;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
