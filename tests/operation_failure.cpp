/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "c50_api_internal.h"
#include <cstdlib>
#include <stdexcept>
#include <string>

struct state { int destroyed = 0; int cleaned = 0; };
struct probe { state &s; ~probe() { ++s.destroyed; } };
static void fail(c50_context *, void *data)
{
    probe p{*static_cast<state *>(data)};
    throw std::runtime_error("original failure");
}
static void succeed(c50_context *, void *) {}
static void cleanup(c50_context *, void *data)
{
    ++static_cast<state *>(data)->cleaned;
}
static void bad_cleanup(c50_context *context, void *data)
{
    cleanup(context, data);
    throw std::logic_error("cleanup failure");
}
static void parser_error(c50_context *context, void *)
{
    c50_record_error(context, c50::error_code::parse_error, "first error");
    c50_record_error(context, c50::error_code::parse_error, "second error");
}
/* Leave allocations in members that only Cleanup frees, as an operation does
 * when it stops before its cleanup runs. */
static void leave_allocations(c50_context *context, void *)
{
    context->training.boost_vote_block =
        static_cast<float *>(malloc(16 * sizeof(float)));
    context->cases.some_missing =
        static_cast<unsigned char *>(malloc(16));
    context->evaluation.utility_errors =
        static_cast<int *>(malloc(16 * sizeof(int)));
    context->class_sum = static_cast<float *>(malloc(16 * sizeof(float)));
    context->ignored_values = static_cast<char *>(malloc(16));
}
int main()
{
    {
        /* The destructor releases what no cleanup freed. A leak checker
         * reports a failure here. */
        auto abandoned = c50_make_context();
        c50_run_operation(abandoned.get(), leave_allocations, nullptr, nullptr);
    }
    auto context = c50_make_context();
    state s;
    for (auto clean : {cleanup, bad_cleanup}) {
        try {
            c50_run_operation(context.get(), fail, clean, &s);
            return 1;
        } catch (const std::runtime_error &error) {
            if (std::string(error.what()) != "original failure") return 1;
        }
        if (context->operation_active) return 1;
        c50_run_operation(context.get(), succeed, nullptr, nullptr);
    }
    if (s.destroyed != 2 || s.cleaned != 2) return 1;
    try {
        c50_run_operation(context.get(), succeed, bad_cleanup, &s);
        return 1;
    } catch (const std::logic_error &) {}
    try {
        c50_run_operation(context.get(), parser_error, cleanup, &s);
        return 1;
    } catch (const c50::exception &error) {
        if (error.code() != c50::error_code::parse_error ||
            std::string(error.what()) != "first error") return 1;
    }
    c50_run_operation(context.get(), succeed, nullptr, nullptr);
    return s.cleaned == 4 && !context->operation_active ? 0 : 1;
}
