/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <cstddef>
#include <cstdint>

namespace c50::test {

enum class fuzz_kind {
    schema = 1,
    implicit,
    tree,
    rules,
    cases,
    costs
};

void exercise_parser(fuzz_kind kind, const std::uint8_t *bytes,
                     std::size_t size);

}
