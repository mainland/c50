/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

// Feed pseudorandom inputs to the training target. Each input depends only on
// the seed and its index, so that a finding replays from those two numbers.
// The driver draws bytes from std::mt19937_64, whose sequence the standard
// fixes, and not from a std::uniform_*_distribution, whose results differ
// between standard libraries.

#include "training_target.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

std::uint64_t splitmix(std::uint64_t value)
{
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}

// The input of one index: a length from 1 to 512 bytes, then the bytes.
std::vector<std::uint8_t> generate(std::uint64_t seed, std::uint64_t index)
{
    std::mt19937_64 engine(splitmix(seed) ^ splitmix(index + 1));
    const std::size_t length = 1 + engine() % 512;
    std::vector<std::uint8_t> bytes(length);
    for (auto &byte : bytes) byte = static_cast<std::uint8_t>(engine() & 0xff);
    return bytes;
}

std::uint64_t parse(const char *text, const char *what)
{
    char *end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (*text == '\0' || *end != '\0') {
        std::cerr << "invalid " << what << ": " << text << '\n';
        std::exit(2);
    }
    return value;
}

}

int main(int argc, char **argv)
{
    if (argc < 4 || argc > 5) {
        std::cerr << "usage: c50_training_fuzz_driver SEED COUNT INPUT-FILE "
                     "[FIRST-INDEX]\n"
                     "Train inputs FIRST-INDEX to FIRST-INDEX + COUNT - 1 of "
                     "SEED, and write each to INPUT-FILE first.\n";
        return 2;
    }
    const std::uint64_t seed = parse(argv[1], "seed");
    const std::uint64_t count = parse(argv[2], "count");
    const std::string path = argv[3];
    const std::uint64_t first = argc == 5 ? parse(argv[4], "first index") : 0;

    const auto start = std::chrono::steady_clock::now();
    std::uint64_t index = first;
    try {
        for (; index - first < count; ++index) {
            const auto bytes = generate(seed, index);
            {
                std::ofstream file(path, std::ios::binary | std::ios::trunc);
                file.write(reinterpret_cast<const char *>(bytes.data()),
                           static_cast<std::streamsize>(bytes.size()));
                file.close();
                if (!file) {
                    std::cerr << "cannot write " << path << '\n';
                    return 2;
                }
            }
            c50::test::exercise_training(bytes.data(), bytes.size());
        }
    } catch (const std::exception &error) {
        std::cerr << "finding: seed " << seed << " index " << index << ": "
                  << error.what() << "\ninput file: " << path << '\n';
        return 1;
    }
    const auto seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << "trained " << count << " inputs of seed " << seed
              << " from index " << first << " in " << seconds << " s\n";
}
