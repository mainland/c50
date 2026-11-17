/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "training_target.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t *bytes, std::size_t size)
{
    try {
        c50::test::exercise_training(bytes, size);
    } catch (...) {
        std::abort();
    }
    return 0;
}
