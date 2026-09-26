/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <utility>
#include "c50_api_internal.h"

c50::exception::exception(error_code code, const std::string &message)
    : std::runtime_error(message), code_(code) {}
c50::error_code c50::exception::code() const noexcept { return code_; }

c50::detail::context_state::~context_state()
{
    c50_clear_prediction_state(this);
    free(active_rules);
    free(ignored_values);
    free(property_value);
}

std::unique_ptr<c50_context> c50_make_context()
{
    auto context = std::make_unique<c50_context>();
    context->schema.max_discrete_value = 3;
    snprintf(context->schema.unknown_class_name,
             sizeof(context->schema.unknown_class_name), "%s", "?");
    snprintf(context->schema.other_attribute_value_name,
             sizeof(context->schema.other_attribute_value_name),
             "%s", "<other>");
    context->cases.max_case = -1;
    context->costs.unit_weights = 1;
    context->options.trials = 1;
    context->options.folds = 10;
    context->options.global_pruning = 1;
    context->options.minimum_cases = 2;
    context->options.confidence_factor = 0.25f;
    context->splits.sample_fraction = 1;
    context->io.file_stem = "undefined";
    context->io.option_index = 1;
    return context;
}

c50::context::context() : state_(c50_make_context()) {}
c50::context::~context() = default;
c50::context::context(context &&) noexcept = default;
c50::context &c50::context::operator=(context &&) noexcept = default;

void c50_clear_prediction_state(c50_context *context)
{
    free(context->class_sum);
    free(context->votes);
    free(context->trial_predictions);
    free(context->most_specific_rules);
    context->class_sum = nullptr;
    context->votes = nullptr;
    context->trial_predictions = nullptr;
    context->most_specific_rules = nullptr;
}

void c50_record_error(c50_context *context, c50::error_code code,
                      const char *message)
{
    if (context->error) return;
    context->error = code;
    snprintf(context->error_message, sizeof(context->error_message), "%s", message);
}

[[noreturn]] void c50_abort_operation(c50_context *context, int exit_status)
{
    if (context->error)
        throw c50::exception(*context->error, context->error_message);
    throw c50::exception(c50::error_code::internal_error,
                        exit_status ? "C5.0 operation failed" :
                                      "C5.0 operation terminated");
}

void c50_run_operation(c50_context *context, c50_operation_fn operation,
                       c50_operation_cleanup_fn cleanup, void *user_data)
{
    if (context->operation_active)
        throw c50::exception(c50::error_code::internal_error,
                            "a C5.0 operation is already active on this context");
    context->error.reset();
    context->error_message[0] = '\0';
    context->operation_active = 1;
    std::exception_ptr failure;
    try {
        operation(context, user_data);
        if (context->error) c50_abort_operation(context, 1);
    } catch (...) {
        failure = std::current_exception();
    }
    try {
        if (cleanup) cleanup(context, user_data);
    } catch (...) {
        if (!failure) failure = std::current_exception();
    }
    context->operation_active = 0;
    if (failure) std::rethrow_exception(failure);
}
