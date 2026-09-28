/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <c50/c50.hpp>
#include <future>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>

namespace {
void require(bool value, const char *message)
{
    if (!value) throw std::runtime_error(message);
}

const std::string names = "low, high.\nsignal: continuous.\ncolor: red, blue, green.\n";
const std::string soft_tree =
    "entries=\"1\"\n"
    "type=\"2\" class=\"low\" freq=\"3,3\" att=\"signal\" forks=\"3\" "
    "cut=\"1.5\" low=\"1\" mid=\"1.5\" high=\"2\"\n"
    "type=\"0\" class=\"low\"\n"
    "type=\"0\" class=\"low\" freq=\"3,0\"\n"
    "type=\"0\" class=\"high\" freq=\"0,3\"\n";

void tree_contract()
{
    c50::context context;
    auto model = c50::model::load(context, c50::model_kind::tree, names, soft_tree);
    const std::string cases = "0, red, ?\n3, blue, ?\n?, ?, ?\nN/A, green, ?\n";
    auto before = model.predict(context, cases);
    auto snapshot = model.inspect(context);
    require(snapshot.class_names == std::vector<std::string>{"low", "high"}, "class order");
    require(snapshot.feature_names == std::vector<std::string>{"signal", "color"}, "feature order");
    require(snapshot.trees.size() == 1, "tree count");
    const auto &tree = snapshot.trees[0];
    require(tree.nodes.size() == 4 && tree.leaf_count == 3 &&
            tree.supported_leaf_count == 2 && tree.depth == 1 && tree.supported_depth == 1,
            "tree statistics include empty N/A leaf");
    require(tree.feature_use == std::vector<std::size_t>{1, 0}, "feature use");
    const auto &root = tree.nodes[0];
    require(root.kind == c50::node_kind::threshold && root.feature == 0 &&
            root.predicted_class == 0 && root.case_weight == 6, "root metadata");
    require(root.class_weights == std::vector<double>{3, 3}, "class distribution");
    require(root.threshold && root.threshold->cut == 1.5 && root.threshold->lower == 1 &&
            root.threshold->midpoint == 1.5 && root.threshold->upper == 2, "soft thresholds");
    require(root.branches[0].condition.kind == c50::condition_kind::not_applicable &&
            root.branches[1].condition.kind == c50::condition_kind::less_equal &&
            root.branches[2].condition.kind == c50::condition_kind::greater, "threshold branch order");
    for (std::size_t index = 0; index < 3; ++index)
        require(root.branches[index].child == index + 1, "preorder child indices");
    require(!tree.nodes[1].feature && !tree.nodes[1].threshold &&
            tree.nodes[1].case_weight == 0, "empty leaf metadata");

    auto copy = snapshot;
    copy.class_names[0] = "changed";
    copy.trees[0].nodes[0].class_weights[0] = 999;
    require(snapshot.class_names[0] == "low" && snapshot.trees[0].nodes[0].class_weights[0] == 3,
            "snapshot copy ownership");
    auto after = model.predict(context, cases);
    for (std::size_t row = 0; row < before.size(); ++row) {
        require(before.class_index(row) == after.class_index(row), "inspection preserves predictions");
        require(before.confidence(row) == after.confidence(row), "inspection preserves confidence");
        for (std::size_t label = 0; label < before.class_count(); ++label)
            require(before.score(row, label) == after.score(row, label), "inspection preserves scores");
    }
    require(model.serialized_data() == soft_tree, "inspection preserves serialized bytes");
}

void categorical_contract()
{
    c50::context context;
    auto discrete = c50::model::load(context, c50::model_kind::tree, names,
        "entries=\"1\"\n"
        "type=\"1\" class=\"low\" freq=\"3,3\" att=\"color\" forks=\"4\"\n"
        "type=\"0\" class=\"low\"\n"
        "type=\"0\" class=\"low\" freq=\"3,0\"\n"
        "type=\"0\" class=\"high\" freq=\"0,2\"\n"
        "type=\"0\" class=\"high\" freq=\"0,1\"\n");
    auto snapshot = discrete.inspect(context);
    const auto &root = snapshot.trees[0].nodes[0];
    require(root.kind == c50::node_kind::discrete && root.feature == 1, "discrete node");
    require(root.branches[1].condition.values == std::vector<std::string>{"red"} &&
            root.branches[1].condition.kind == c50::condition_kind::equals, "category names");
    auto subset = c50::model::load(context, c50::model_kind::tree, names,
        "entries=\"1\"\n"
        "type=\"3\" class=\"low\" freq=\"3,3\" att=\"color\" forks=\"2\" "
        "elts=\"N/A\",\"red\" elts=\"blue\",\"green\"\n"
        "type=\"0\" class=\"low\" freq=\"3,0\"\n"
        "type=\"0\" class=\"high\" freq=\"0,3\"\n");
    snapshot = subset.inspect(context);
    const auto &branch = snapshot.trees[0].nodes[0].branches[0];
    require(branch.condition.kind == c50::condition_kind::in_subset &&
            branch.condition.includes_not_applicable &&
            branch.condition.values == std::vector<std::string>{"red"}, "subset N/A membership");
    require(snapshot.trees[0].nodes[0].branches[1].condition.values ==
            std::vector<std::string>{"blue", "green"}, "subset category order");
}

void ownership_and_training()
{
    auto snapshot = [] {
        c50::context context;
        auto model = c50::model::train(context, c50::model_kind::tree,
            "low, high.\nsignal: continuous.\n",
            "0, low\n1, low\n2, low\n3, high\n4, high\n5, high\n");
        auto first = model.inspect(context);
        auto reloaded = c50::model::load(context, model.kind(), model.names_data(), model.serialized_data());
        auto second = reloaded.inspect(context);
        require(first.trees[0].nodes.size() == second.trees[0].nodes.size(), "round-trip shape");
        require(first.trees[0].nodes[0].class_weights == second.trees[0].nodes[0].class_weights,
                "round-trip serialized precision");
        return first;
    }();
    require(snapshot.trees[0].supported_leaf_count == 2 && snapshot.class_names[1] == "high",
            "snapshot outlives owners");

    c50::context context;
    auto model = c50::model::load(context, c50::model_kind::tree, names, soft_tree);
    auto worker = [&model] {
        c50::context own;
        return model.inspect(own).trees[0].nodes.size();
    };
    auto first = std::async(std::launch::async, worker);
    auto second = std::async(std::launch::async, worker);
    require(first.get() == 4 && second.get() == 4, "concurrent inspection");
}
} // namespace

int main()
{
    try {
        tree_contract();
        categorical_contract();
        ownership_and_training();
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
