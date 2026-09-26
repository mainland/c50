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

int main()
{
    try {
        c50::context context;
        const auto kind = c50::model_kind::tree;
        auto model = c50::model::load(context, kind, names, header + leaf);
        rejects([&] { c50::model::load(context, kind,
            "no, yes.\n" + std::string(2000, 'x') + ": continuous.\n", header + leaf); });
        rejects([&] { model.predict(context, std::string(2000, '1') + ", ?\n"); });
        rejects([&] { c50::model::load(context, kind, names,
            header + "type=\"0\" class=\"" + std::string(30000, 'x') + "\" freq=\"2,0\"\n"); });
        if (model.predict(context, "0, ?\n").class_index(0) != 0) return 1;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
