/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <type_traits>

#include <c50/c50.h>

static_assert(std::is_enum<c50_status>::value, "c50_status must be an enum");
static_assert(std::is_enum<c50_model_kind>::value,
              "c50_model_kind must be an enum");

int main()
{
    c50_context *context = nullptr;
    c50_model *model = nullptr;

    return context || model;
}
