/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "parser_target.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t *bytes, std::size_t size)
{
    try {
        c50::test::exercise_parser(
            static_cast<c50::test::fuzz_kind>(C50_FUZZ_KIND), bytes, size);
    } catch (...) {
        std::abort();
    }
    return 0;
}
