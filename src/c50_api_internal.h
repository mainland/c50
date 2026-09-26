/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef C50_API_INTERNAL_H
#define C50_API_INTERNAL_H



typedef struct c50_context c50_context;
enum c50_status {
    C50_STATUS_OK, C50_STATUS_INVALID_ARGUMENT, C50_STATUS_OUT_OF_MEMORY,
    C50_STATUS_IO_ERROR, C50_STATUS_PARSE_ERROR, C50_STATUS_UNSUPPORTED,
    C50_STATUS_INTERNAL_ERROR
};
c50_status c50_context_create(c50_context **out_context);
void c50_context_destroy(c50_context *context);
c50_status c50_context_last_status(const c50_context *context);
const char *c50_context_error_message(const c50_context *context);
const char *c50_status_message(c50_status status);
struct c50_operation_abort { int status; };


#include "c50_input.h"

#define C50_ERROR_MESSAGE_CAPACITY 1024
#define C50_LINE_BUFFER_CAPACITY 10000

struct c50_implicit_state;

struct c50_context
{
    c50_status status;
    char error_message[C50_ERROR_MESSAGE_CAPACITY];
    int sample_from;
    int suppress_error_messages;
    int delimiter;
    char line_buffer[C50_LINE_BUFFER_CAPACITY];
    char *line_buffer_position;
    struct c50_implicit_state *implicit_state;
    c50_input classifier_input;
    const char *last_model_extension;
    int model_entry;
    char property_name[20];
    char *property_value;
    int property_value_size;
};

typedef void (*c50_operation_fn)(c50_context *context, void *user_data);
typedef void (*c50_operation_cleanup_fn)(c50_context *context,
                                         void *user_data);

/*
 * Run operation below a C++ exception boundary. Cleanup must not fail or call
 * C50Exit. The legacy core is not yet reentrant, so nested operations are rejected.
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
