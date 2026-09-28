/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "parser_target.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using target = std::pair<c50::test::fuzz_kind, const char *>;

constexpr std::array<target, 6> targets{{
    {c50::test::fuzz_kind::schema, "schema"},
    {c50::test::fuzz_kind::implicit, "implicit"},
    {c50::test::fuzz_kind::tree, "tree"},
    {c50::test::fuzz_kind::rules, "rules"},
    {c50::test::fuzz_kind::cases, "cases"},
    {c50::test::fuzz_kind::costs, "costs"},
}};

std::vector<std::filesystem::path> files(const std::filesystem::path &directory)
{
    std::vector<std::filesystem::path> result;
    for (const auto &entry : std::filesystem::directory_iterator(directory))
        if (entry.is_regular_file()) result.push_back(entry.path());
    std::sort(result.begin(), result.end());
    return result;
}

std::vector<std::uint8_t> read(const std::filesystem::path &path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read " + path.string());
    return {std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
}

}

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "usage: c50_fuzz_seed_replay CORPUS-DIRECTORY\n";
        return 2;
    }
    try {
        for (const auto &[kind, name] : targets)
            for (const auto &path : files(std::filesystem::path(argv[1]) / name)) {
                const auto input = read(path);
                c50::test::exercise_parser(kind, input.data(), input.size());
            }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
