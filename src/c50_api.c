/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <stdlib.h>

#include <c50/c50.h>

#define C50_ERROR_MESSAGE_CAPACITY 1024

struct c50_context
{
    c50_status status;
    char error_message[C50_ERROR_MESSAGE_CAPACITY];
};

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
    *out_context = context;
    return C50_STATUS_OK;
}

void c50_context_destroy(c50_context *context)
{
    free(context);
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
