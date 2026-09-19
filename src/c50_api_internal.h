/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef C50_API_INTERNAL_H
#define C50_API_INTERNAL_H

#include <c50/c50.h>

typedef void (*c50_operation_fn)(void *user_data);
typedef void (*c50_operation_cleanup_fn)(void *user_data);

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

/* Unwind the active operation, or return zero when no operation is active. */
int c50_abort_active_operation(int exit_status);

#endif
