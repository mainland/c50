/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <cstddef>
#include <cstdint>

namespace c50::test {

/**
 * Decode a small classification problem from bytes, train it through the text
 * and the dense interfaces, and check that the interfaces agree.
 *
 * Every byte string decodes to a problem, and bytes beyond the end read as
 * zero. Inputs longer than max_training_input_size are ignored. The function
 * throws when training fails or when a property is violated, and returns
 * normally when the problem trains and every property holds.
 *
 * @param bytes Input bytes, null only when size is zero.
 * @param size Input size in bytes.
 */
void exercise_training(const std::uint8_t *bytes, std::size_t size);

/** The longest input that exercise_training decodes. */
inline constexpr std::size_t max_training_input_size = 4096;

}
