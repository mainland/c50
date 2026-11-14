/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "c50_api_internal.h"

#include <c50/c50.hpp>

#include <cstddef>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

const std::string names = "no, yes.\nx: continuous.\n";
const std::string header =
    "id=\"See5/C5.0 2.07 GPL Edition\"\nentries=\"1\"\n";
const std::string no_leaf = "type=\"0\" class=\"no\" freq=\"2,0\"\n";
const std::string yes_leaf = "type=\"0\" class=\"yes\" freq=\"0,2\"\n";
const std::string node =
    "type=\"2\" class=\"no\" att=\"x\" forks=\"3\" cut=\"0\" freq=\"1,1\"\n";

void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

/* A tree of the given number of levels. Every node tests x against 0. The
   branch for x <= 0 leads one level down, so that a case with x = -1 reaches
   the "yes" leaf on the deepest level, and a case with x = 5 stops at the
   "no" leaf under the root. */
std::string tree_text(std::size_t levels)
{
    const std::size_t internal = levels - 1;
    std::string text = header;
    for (std::size_t depth = 0; depth < internal; ++depth)
        text += node + no_leaf;
    text += yes_leaf;
    for (std::size_t depth = 0; depth < internal; ++depth) text += no_leaf;
    return text;
}

template <class F>
void throws_code(F operation, c50::error_code code, const char *message)
{
    try {
        operation();
    } catch (const c50::exception &error) {
        require(error.code() == code, message);
        return;
    }
    throw std::runtime_error(message);
}

std::size_t class_of(const c50::predictions &result)
{
    require(result.size() == 1, "one prediction");
    return result.class_index(0);
}

/* Every operation that parses the model's text again must succeed. */
void check_usable(c50::context &context, const c50::model &model,
                  std::size_t levels)
{
    require(class_of(model.predict(context, "-1, ?\n")) == 1,
            "a case at the deepest level reaches its leaf");
    require(class_of(model.predict(context, "5, ?\n")) == 0,
            "a case at the root reaches its leaf");
    const auto inspection = model.inspect(context);
    require(inspection.trees.size() == 1 &&
                inspection.trees[0].nodes.size() == 2 * levels - 1 + levels - 1,
            "inspection sees every node");
    auto predictor = model.prepare_predictor();
    require(class_of(predictor.predict("-1, ?\n")) == 1,
            "a prepared predictor reaches the deepest leaf");
}

void default_limit(c50::context &context)
{
    const auto tree = c50::model_kind::tree;
    require(c50::default_max_tree_depth == 1024, "the default limit changed");
    require(c50::load_options{}.max_tree_depth == c50::default_max_tree_depth,
            "load_options does not default to the constant");

    // The limit counts levels, so 1,024 levels load and 1,025 do not.
    const auto limit = c50::default_max_tree_depth;
    const auto at_limit =
        c50::model::load(context, tree, names, tree_text(limit));
    require(c50::detail::model_access::max_tree_depth(at_limit) == limit,
            "a loaded model keeps the default limit");
    require(class_of(at_limit.predict(context, "-1, ?\n")) == 1,
            "a tree at the limit predicts");
    throws_code(
        [&] {
            c50::model::load(context, tree, names, tree_text(limit + 1));
        },
        c50::error_code::parse_error, "a tree past the limit loads");
}

void explicit_limit(c50::context &context)
{
    const auto tree = c50::model_kind::tree;
    c50::load_options three;
    three.max_tree_depth = 3;
    const auto small = c50::model::load(context, tree, names, tree_text(3),
                                        {}, three);
    require(class_of(small.predict(context, "-1, ?\n")) == 1,
            "a tree at an explicit limit predicts");
    try {
        c50::model::load(context, tree, names, tree_text(4), {}, three);
        throw std::runtime_error("a tree past an explicit limit loads");
    } catch (const c50::exception &error) {
        require(error.code() == c50::error_code::parse_error,
                "a deep tree is a parse error");
        require(std::string(error.what()).find("max_tree_depth") !=
                    std::string::npos,
                "the message names the limit");
    }
}

void zero_limit(c50::context &context)
{
    c50::load_options zero;
    zero.max_tree_depth = 0;
    for (const auto kind : {c50::model_kind::tree, c50::model_kind::rules})
        throws_code(
            [&] {
                c50::model::load(context, kind, names, tree_text(1), {}, zero);
            },
            c50::error_code::invalid_argument, "a zero limit loads");
}

/* A tree past the default limit is usable once the caller removes it, and
   the model keeps its limit for every later parse. */
void no_limit(c50::context &context)
{
    const auto tree = c50::model_kind::tree;
    const std::size_t levels = c50::default_max_tree_depth + 76;
    const std::string text = tree_text(levels);
    throws_code([&] { c50::model::load(context, tree, names, text); },
                c50::error_code::parse_error, "the default accepts a deep tree");

    c50::load_options unlimited;
    unlimited.max_tree_depth = std::nullopt;
    auto model = c50::model::load(context, tree, names, text, {}, unlimited);
    require(!c50::detail::model_access::max_tree_depth(model),
            "an unlimited model keeps no limit");
    check_usable(context, model, levels);

    // Moving the model keeps its limit.
    c50::model moved = std::move(model);
    require(!c50::detail::model_access::max_tree_depth(moved),
            "a moved model keeps no limit");
    check_usable(context, moved, levels);

    c50::load_options larger;
    larger.max_tree_depth = levels;
    const auto sized = c50::model::load(context, tree, names, text, {}, larger);
    require(c50::detail::model_access::max_tree_depth(sized) == levels,
            "an explicit limit is the limit of every parse");
    check_usable(context, sized, levels);
    larger.max_tree_depth = levels - 1;
    throws_code(
        [&] { c50::model::load(context, tree, names, text, {}, larger); },
        c50::error_code::parse_error, "a limit one level short loads");

    // The model's own text loads again when the caller removes the limit.
    const auto copy = c50::model::load(context, tree, names,
                                       moved.serialized_data(), {}, unlimited);
    check_usable(context, copy, levels);
}

/* A model that train() returns parses its own text without a limit, and so
   does its prepared predictor. The depth where the limit would apply cannot be
   reached without minutes of training, so the test reads the limit directly. */
void trained_model(c50::context &context)
{
    const std::string data = "0, no\n1, yes\n2, no\n3, yes\n4, no\n5, yes\n";
    const std::string training_names = "cls.\nx: continuous.\ncls: no, yes.\n";
    auto model = c50::model::train(context, c50::model_kind::tree,
                                   training_names, data);
    require(!c50::detail::model_access::max_tree_depth(model),
            "a trained model has a depth limit");
    c50::model moved = std::move(model);
    require(!c50::detail::model_access::max_tree_depth(moved),
            "a moved trained model has a depth limit");
    moved.predict(context, "1, ?\n");
    moved.prepare_predictor();
}

} // namespace

int main()
{
    try {
        c50::context context;
        default_limit(context);
        explicit_limit(context);
        zero_limit(context);
        no_limit(context);
        trained_model(context);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
