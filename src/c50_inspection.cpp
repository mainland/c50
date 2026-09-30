/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <algorithm>
#include <utility>
#include <vector>

#include <c50/c50.hpp>
#include "c50_api_internal.h"
#include "defns.i"

namespace {

c50::split_condition condition(c50_context *Context, BranchType kind,
                                Attribute attribute, int outcome,
                                ContValue cut, Set subset)
{
    c50::split_condition result;
    result.feature = static_cast<std::size_t>(attribute - 1);
    if (kind == BrSubset) {
        result.kind = c50::condition_kind::in_subset;
        result.includes_not_applicable = In(1, subset);
        for (int value = 2; value <= Context->schema.max_attribute_value[attribute]; ++value)
            if (In(value, subset))
                result.values.emplace_back(Context->schema.attribute_value_names[attribute][value]);
    } else if (outcome == 1) {
        result.kind = c50::condition_kind::not_applicable;
    } else if (kind == BrThresh) {
        result.kind = outcome == 2 ? c50::condition_kind::less_equal :
                                    c50::condition_kind::greater;
        result.cut = cut;
    } else {
        result.kind = c50::condition_kind::equals;
        result.values.emplace_back(Context->schema.attribute_value_names[attribute][outcome]);
    }
    return result;
}

c50::tree_inspection inspect_tree(c50_context *Context, Tree root)
{
    struct pending_node {
        Tree source;
        std::size_t depth, parent, branch;
    };
    std::vector<pending_node> pending{{root, 0, 0, 0}};
    c50::tree_inspection result;
    result.feature_use.resize(Context->schema.max_attribute);
    while (!pending.empty()) {
        const auto item = pending.back();
        pending.pop_back();
        const auto source = item.source;
        const auto index = result.nodes.size();
        if (index) result.nodes[item.parent].branches[item.branch].child = index;
        result.depth = std::max(result.depth, item.depth);

        c50::tree_node node;
        node.predicted_class = static_cast<std::size_t>(source->Leaf - 1);
        node.case_weight = source->Cases;
        node.class_weights.assign(source->ClassDist + 1,
                                  source->ClassDist + Context->schema.max_class + 1);
        if (!source->NodeType) {
            ++result.leaf_count;
            if (source->Cases > 0) {
                ++result.supported_leaf_count;
                result.supported_depth = std::max(result.supported_depth, item.depth);
            }
        } else {
            node.feature = static_cast<std::size_t>(source->Tested - 1);
            ++result.feature_use[*node.feature];
            switch (source->NodeType) {
            case BrDiscr: node.kind = c50::node_kind::discrete; break;
            case BrSubset: node.kind = c50::node_kind::subset; break;
            case BrThresh:
                node.kind = c50::node_kind::threshold;
                node.threshold = c50::continuous_threshold{
                    source->Cut, source->Lower, source->Mid, source->Upper};
                break;
            default:
                throw c50::exception(c50::error_code::internal_error,
                                     "unknown inspected tree node kind");
            }
            for (int branch = 1; branch <= source->Forks; ++branch) {
                auto test = condition(Context, source->NodeType, source->Tested,
                                      branch, source->Cut,
                                      source->NodeType == BrSubset ? source->Subset[branch] : nullptr);
                node.branches.push_back({std::move(test), 0});
            }
            // Reverse insertion preserves native branch order in the preorder vector.
            for (int branch = source->Forks; branch >= 1; --branch)
                pending.push_back({source->Branch[branch], item.depth + 1, index,
                                   static_cast<std::size_t>(branch - 1)});
        }
        result.nodes.push_back(std::move(node));
    }
    return result;
}

c50::ruleset_inspection inspect_rules(c50_context *Context, CRuleSet source)
{
    c50::ruleset_inspection result;
    result.default_class = static_cast<std::size_t>(source->SDefault - 1);
    result.feature_use.resize(Context->schema.max_attribute);
    for (int index = 1; index <= source->SNRules; ++index) {
        const auto rule = source->SRule[index];
        c50::rule_inspection item;
        item.predicted_class = static_cast<std::size_t>(rule->Rhs - 1);
        item.cover = rule->Cover;
        item.correct = rule->Correct;
        item.prior = rule->Prior;
        item.vote = rule->Vote;
        for (int term = 1; term <= rule->Size; ++term) {
            const auto test = rule->Lhs[term];
            item.conditions.push_back(condition(Context, test->NodeType, test->Tested,
                                                test->TestValue, test->Cut, test->Subset));
            ++result.feature_use[static_cast<std::size_t>(test->Tested - 1)];
        }
        result.rules.push_back(std::move(item));
    }
    return result;
}

} // namespace

c50::model_inspection c50_build_inspection(c50_context *Context)
{
    c50::model_inspection result;
    result.kind = Context->options.rules ? c50::model_kind::rules : c50::model_kind::tree;
    for (int index = 1; index <= Context->schema.max_class; ++index)
        result.class_names.emplace_back(Context->schema.class_names[index]);
    for (int index = 1; index <= Context->schema.max_attribute; ++index)
        result.feature_names.emplace_back(Context->schema.attribute_names[index]);
    for (int trial = 0; trial < Context->options.trials; ++trial) {
        if (Context->options.rules)
            result.rulesets.push_back(inspect_rules(Context, Context->rules.sets[trial]));
        else
            result.trees.push_back(inspect_tree(Context, Context->trees.pruned[trial]));
    }
    return result;
}
