/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <stdio.h>
#include <string.h>

#include <c50/c50.h>

#include "c50_api_internal.h"
#include "defns.i"
#include "extern.i"

typedef struct operation_state
{
    int operation_calls;
    int cleanup_calls;
} operation_state;

static void FailWithModelError(void *user_data)
{
    operation_state *state = user_data;

    state->operation_calls++;
    Error(MODELFILE, E_MFATT, "unknown");
}

static void Succeed(void *user_data)
{
    operation_state *state = user_data;

    state->operation_calls++;
}

static void CleanupOperation(void *user_data)
{
    operation_state *state = user_data;

    state->cleanup_calls++;
    if ( Of )
    {
        fclose(Of);
        Of = NULL;
    }
}

int main(int argc, char *argv[])
{
    c50_context *context = NULL;
    operation_state state = {0, 0};
    c50_status status;

    (void) argc;
    (void) argv;

    if ( c50_context_create(&context) != C50_STATUS_OK ) return 1;

    Of = tmpfile();
    if ( ! Of ) return 1;
    snprintf(Fn, sizeof(Fn), "%s", "memory.tree");

    status = c50_run_operation(context, FailWithModelError,
                               CleanupOperation, &state);
    if ( status != C50_STATUS_PARSE_ERROR ) return 1;
    if ( c50_context_last_status(context) != C50_STATUS_PARSE_ERROR ) return 1;
    if ( ! strstr(c50_context_error_message(context), "unknown") ) return 1;
    if ( state.operation_calls != 1 || state.cleanup_calls != 1 ) return 1;
    if ( Of ) return 1;

    status = c50_run_operation(context, Succeed, CleanupOperation, &state);
    if ( status != C50_STATUS_OK ) return 1;
    if ( c50_context_error_message(context)[0] != '\0' ) return 1;
    if ( state.operation_calls != 2 || state.cleanup_calls != 2 ) return 1;

    c50_context_destroy(context);
    return 0;
}
