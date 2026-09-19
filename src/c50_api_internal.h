/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef C50_API_INTERNAL_H
#define C50_API_INTERNAL_H

#include <setjmp.h>

#include <c50/c50.h>

#define C50_ERROR_MESSAGE_CAPACITY 1024

struct c50_context
{
    c50_status status;
    char error_message[C50_ERROR_MESSAGE_CAPACITY];
    jmp_buf exit_target;
};

typedef void (*c50_operation_fn)(c50_context *context, void *user_data);
typedef void (*c50_operation_cleanup_fn)(c50_context *context,
                                         void *user_data);

/*
 * Run operation below a C-only failure boundary. Cleanup must not fail or call
 * C50Exit. The legacy core is not reentrant, so nested operations are rejected.
 */
c50_status c50_run_operation(c50_context *context,
                             c50_operation_fn operation,
                             c50_operation_cleanup_fn cleanup,
                             void *user_data);

/* Record the first error raised by the active operation. */
void c50_record_error(c50_status status, const char *message);

/* Replace the context result without starting an operation. */
c50_status c50_set_context_error(c50_context *context, c50_status status,
                                 const char *message);

/* Unwind the active operation, or return zero when no operation is active. */
int c50_abort_active_operation(int exit_status);

#endif
