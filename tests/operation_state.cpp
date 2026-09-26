/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */
/* A context's history must not change what it trains or how it names its
   inputs.  Each case runs the same operation on a reused context and on a
   fresh one and requires the same result. */
#include <c50/c50.hpp>
#include <iostream>
#include <stdexcept>
#include <string>

static void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

static const std::string stamp_names =
    "c.\nt: timestamp.\nc: early, late.\n";
static const std::string id_prefix = "id=\"See5/C5.0 2.07 GPL Edition ";

/* Twenty daily timestamps from 2025-03-01 whose class changes on the 11th. */
static std::string stamp_data()
{
    std::string data;
    for (int day = 1; day <= 20; ++day) {
        data += "2025/03/" + std::string(day < 10 ? "0" : "") +
                std::to_string(day) + " 12:00:00, " +
                (day <= 10 ? "early" : "late") + "\n";
    }
    return data;
}

/* Count the training cases that a classifier, parsed in a fresh context,
   predicts wrongly. */
static int stamp_errors(const c50::model &model)
{
    const std::string data = stamp_data();
    c50::context fresh;
    auto reloaded = c50::model::load(fresh, c50::model_kind::tree, stamp_names,
                                     model.serialized_data());
    auto result = reloaded.predict(fresh, data);
    int errors = 0;
    for (std::size_t row = 0; row < result.size(); ++row)
        errors += result.class_index(row) != (row < 10 ? 0u : 1u);
    return errors;
}

static c50::model train_stamps(c50::context &context)
{
    return c50::model::train(context, c50::model_kind::tree, stamp_names,
                             stamp_data());
}

/* A model whose id names another year. */
static std::string with_year(std::string text, const char *year)
{
    const auto at = text.find(id_prefix);
    require(at != std::string::npos, "model id");
    text.replace(at + id_prefix.size(), 4, year);
    return text;
}

static void stale_timestamp_base()
{
    c50::context fresh;
    const auto expected = train_stamps(fresh);
    require(stamp_errors(expected) == 0, "fresh context errors");

    c50::context reused;
    auto old = c50::model::load(reused, c50::model_kind::tree, stamp_names,
                                with_year(expected.serialized_data(), "2010"));
    (void) old;
    const auto trained = train_stamps(reused);
    require(trained.serialized_data() == expected.serialized_data(),
            "training after a load differs from a fresh context");
    require(stamp_errors(trained) == 0, "errors after a load of another year");
}

static const std::string cost_names = "low, high.\nsignal: continuous.\n";
static const std::string plain_tree = id_prefix + "2026-09-19\"\n"
    "entries=\"1\"\n"
    "type=\"0\" class=\"low\" freq=\"6,4\"\n";
static const std::string cost_tree = id_prefix + "2026-09-19\"\n"
    "costs=\"1\"\nentries=\"1\"\n"
    "type=\"0\" class=\"low\" freq=\"6,4\"\n";

/* Costs are retained only when the model reads them, so a caller cannot
   conclude that a model without costs is cost sensitive. */
static void unused_costs()
{
    c50::context context;
    const auto kind = c50::model_kind::tree;
    for (const char *costs : {"low, high: 5\n", "not a costs file\n"}) {
        auto plain = c50::model::load(context, kind, cost_names, plain_tree,
                                      costs);
        require(plain.costs_data().empty(), "costs kept for a plain model");
    }
    auto costly = c50::model::load(context, kind, cost_names, cost_tree,
                                   "low, high: 5\n");
    require(costly.costs_data() == "low, high: 5\n", "costs of a cost model");
    /* Costs text that sets no cost other than the default of 1 defines no
       matrix. It is as good as none, and must fail at load, not on every
       later use. The text holds nothing, only comments, only entries of
       cost 1, or only entries between a class and itself. */
    for (const char *costs : {"\n", "  \n\n", "\t \n", "| a comment\n",
                              "low, high: 1\n", "low, low: 5\n"}) {
        try {
            c50::model::load(context, kind, cost_names, cost_tree, costs);
        } catch (const c50::exception &error) {
            require(error.code() == c50::error_code::invalid_argument,
                    "missing costs must be an invalid argument");
            continue;
        }
        throw std::runtime_error("missing or blank costs accepted");
    }
    /* A comment beside a real entry still defines a matrix and keeps
       working. */
    auto commented_entry = c50::model::load(context, kind, cost_names,
                                            cost_tree,
                                            "| a comment\nlow, high: 5\n");
    commented_entry.predict(context, "1, ?\n");
}

/* The diagnostic of a failed operation, without the command-line program's
   decoration. */
template<class F> static std::string diagnostic(F operation)
{
    try {
        operation();
    } catch (const c50::exception &error) {
        const std::string text = error.what();
        require(!text.empty() && text.front() != '*' && text.front() != ' ' &&
                text.back() != '\n', "decorated diagnostic");
        return text;
    }
    throw std::runtime_error("invalid input accepted");
}

static bool cites(const std::string &text, const char *input)
{
    return text.find(std::string("`") + input + "'") != std::string::npos;
}

/* A diagnostic names the input being read, whatever the context did before. */
static void diagnostics_name_their_input()
{
    const auto kind = c50::model_kind::tree;
    const std::string cases = "1, low\nzzz, high\n";
    c50::context context;
    require(cites(diagnostic([&] {
        c50::model::load(context, kind, "low, high.\nsignal: bogus.\n",
                         plain_tree);
    }), "names"), "names");
    require(cites(diagnostic([&] {
        c50::model::train(context, kind, cost_names, cases);
    }), "training data"), "training data on a fresh context");
    auto model = c50::model::load(context, kind, cost_names, plain_tree);
    model.predict(context, "1, ?\n");
    require(cites(diagnostic([&] {
        c50::model::train(context, kind, cost_names, cases);
    }), "training data"), "training data after a prediction");
    require(cites(diagnostic([&] {
        model.predict(context, "zzz, ?\n");
    }), "cases"), "cases");
    require(cites(diagnostic([&] {
        c50::model::train(context, kind, cost_names, "1, low\n2, high\n", {},
                          "low, junk: 5\n");
    }), "costs"), "costs");
}

/* Run every case, so that one failure does not hide the others. */
template<class F> static int run(const char *name, F test)
{
    try {
        test();
    } catch (const std::exception &error) {
        std::cerr << name << ": " << error.what() << '\n';
        return 1;
    }
    return 0;
}

int main()
{
    int failures = 0;
    failures += run("stale timestamp base", stale_timestamp_base);
    failures += run("unused costs", unused_costs);
    failures += run("diagnostics", diagnostics_name_their_input);
    return failures ? 1 : 0;
}
