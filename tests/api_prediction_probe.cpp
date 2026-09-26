/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <c50/c50.hpp>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>

static std::string read_file(const std::string &path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + path);
    return std::string(std::istreambuf_iterator<char>(input), {});
}

int main(int argc, char **argv)
{
    if (argc != 3 || (std::string(argv[2]) != "tree" && std::string(argv[2]) != "rules")) {
        std::fprintf(stderr, "Usage: api-prediction-probe <filestem> <tree|rules>\n");
        return 1;
    }
    try {
        const std::string stem = argv[1];
        const auto kind = std::string(argv[2]) == "rules" ?
            c50::model_kind::rules : c50::model_kind::tree;
        c50::context context;
        auto model = c50::model::load(context, kind, read_file(stem + ".names"),
                                    read_file(stem + "." + argv[2]));
        auto result = model.predict(context, read_file(stem + ".test"));
        std::printf("case,predicted,confidence");
        for (std::size_t index = 0; index < result.class_count(); ++index)
            std::printf(",score(%s)", result.class_name(index).c_str());
        std::putchar('\n');
        for (std::size_t row = 0; row < result.size(); ++row) {
            std::printf("%lu,%s,%.7g", static_cast<unsigned long>(row + 1),
                        result.class_name(result.class_index(row)).c_str(),
                        result.confidence(row));
            for (std::size_t index = 0; index < result.class_count(); ++index)
                std::printf(",%.7g", result.score(row, index));
            std::putchar('\n');
        }
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
