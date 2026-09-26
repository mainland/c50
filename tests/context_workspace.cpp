/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "c50_api_internal.h"
int main() {
    c50_context *first = nullptr, *second = nullptr;
    if (c50_context_create(&first) != C50_STATUS_OK ||
        c50_context_create(&second) != C50_STATUS_OK) return 1;
    if (!first || !second || first == second) return 1;
    if (c50_context_last_status(first) != C50_STATUS_OK ||
        c50_context_error_message(first)[0]) return 1;
    c50_context_destroy(first);
    c50_context_destroy(second);
    return 0;
}
