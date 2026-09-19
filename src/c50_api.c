/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <c50/c50.h>

#include "c50_api_internal.h"

static c50_context *ActiveContext;

static void SetError(c50_context *context, c50_status status,
                     const char *message)
{
    if ( ! context || context->status != C50_STATUS_OK ) return;

    context->status = status;
    if ( message )
    {
        snprintf(context->error_message, sizeof(context->error_message),
                 "%s", message);
    }
}

c50_status c50_context_create(c50_context **out_context)
{
    c50_context *context;

    if ( ! out_context )
    {
        return C50_STATUS_INVALID_ARGUMENT;
    }

    *out_context = NULL;
    context = calloc(1, sizeof(*context));
    if ( ! context )
    {
        return C50_STATUS_OUT_OF_MEMORY;
    }

    context->status = C50_STATUS_OK;
    context->schema.max_discrete_value = 3;
    *out_context = context;
    return C50_STATUS_OK;
}

void c50_context_destroy(c50_context *context)
{
    if ( ! context ) return;
    c50_clear_prediction_state(context);
    free(context->active_rules);
    free(context->ignored_values);
    free(context->property_value);
    free(context);
}

void c50_clear_prediction_state(c50_context *context)
{
    if ( ! context ) return;
    free(context->class_sum);
    free(context->votes);
    free(context->trial_predictions);
    free(context->most_specific_rules);
    context->class_sum = NULL;
    context->votes = NULL;
    context->trial_predictions = NULL;
    context->most_specific_rules = NULL;
}

c50_status c50_context_last_status(const c50_context *context)
{
    return context ? context->status : C50_STATUS_INVALID_ARGUMENT;
}

const char *c50_context_error_message(const c50_context *context)
{
    return context ? context->error_message :
                     c50_status_message(C50_STATUS_INVALID_ARGUMENT);
}

const char *c50_status_message(c50_status status)
{
    switch ( status )
    {
        case C50_STATUS_OK:
            return "success";
        case C50_STATUS_INVALID_ARGUMENT:
            return "invalid argument";
        case C50_STATUS_OUT_OF_MEMORY:
            return "out of memory";
        case C50_STATUS_IO_ERROR:
            return "I/O error";
        case C50_STATUS_PARSE_ERROR:
            return "parse error";
        case C50_STATUS_UNSUPPORTED:
            return "unsupported operation";
        case C50_STATUS_INTERNAL_ERROR:
            return "internal error";
        default:
            return "unknown C5.0 status";
    }
}

c50_status c50_run_operation(c50_context *context,
                             c50_operation_fn operation,
                             c50_operation_cleanup_fn cleanup,
                             void *user_data)
{
    if ( ! context || ! operation ) return C50_STATUS_INVALID_ARGUMENT;

    if ( ActiveContext )
    {
        SetError(context, C50_STATUS_INTERNAL_ERROR,
                 "a C5.0 operation is already active");
        return context->status;
    }

    context->status = C50_STATUS_OK;
    context->error_message[0] = '\0';

    if ( ! setjmp(context->exit_target) )
    {
        ActiveContext = context;
        operation(context, user_data);
    }

    ActiveContext = NULL;
    if ( cleanup ) cleanup(context, user_data);
    return context->status;
}

void c50_record_error(c50_status status, const char *message)
{
    SetError(ActiveContext, status, message);
}

c50_status c50_set_context_error(c50_context *context, c50_status status,
                                 const char *message)
{
    if ( ! context ) return C50_STATUS_INVALID_ARGUMENT;

    context->status = C50_STATUS_OK;
    context->error_message[0] = '\0';
    SetError(context, status, message);
    return status;
}

int c50_abort_active_operation(int exit_status)
{
    if ( ! ActiveContext ) return 0;

    if ( ActiveContext->status == C50_STATUS_OK )
    {
        SetError(ActiveContext, C50_STATUS_INTERNAL_ERROR,
                 exit_status ? "C5.0 operation failed" :
                               "C5.0 operation terminated");
    }
    longjmp(ActiveContext->exit_target, exit_status ? exit_status : 1);
}
