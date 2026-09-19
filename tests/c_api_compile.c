/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <string.h>

#include <c50/c50.h>

int main(void)
{
    c50_context *context = 0;

    if ( c50_context_create(0) != C50_STATUS_INVALID_ARGUMENT ) return 1;
    if ( c50_context_create(&context) != C50_STATUS_OK ) return 1;
    if ( ! context ) return 1;
    if ( c50_context_last_status(context) != C50_STATUS_OK ) return 1;
    if ( strcmp(c50_context_error_message(context), "") ) return 1;
    if ( strcmp(c50_status_message(C50_STATUS_PARSE_ERROR), "parse error") )
    {
        return 1;
    }
    if ( strcmp(c50_status_message((c50_status) 100),
                "unknown C5.0 status") )
    {
        return 1;
    }

    c50_context_destroy(context);
    c50_context_destroy(0);

    return c50_context_last_status(0) != C50_STATUS_INVALID_ARGUMENT ||
           strcmp(c50_context_error_message(0), "invalid argument");
}
