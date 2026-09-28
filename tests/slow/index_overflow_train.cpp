/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */
// Train through the C++ API from a names file and a data file, as the slow
// index-overflow check does.
#include <c50/c50.hpp>

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iterator>
#include <string>

static std::string slurp(const char *path)
{
    std::ifstream in(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in),
                       std::istreambuf_iterator<char>());
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s NAMES DATA TRIALS\n", argv[0]);
        return 2;
    }
    try {
        c50::options options;
        options.trials = static_cast<unsigned>(std::atoi(argv[3]));
        c50::context context;
        c50::model::train(context, c50::model_kind::tree, slurp(argv[1]),
                          slurp(argv[2]), options);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "exception: %s\n", error.what());
        return 1;
    }
    std::puts("trained");
    return 0;
}
