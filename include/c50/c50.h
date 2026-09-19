/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef C50_C50_H
#define C50_C50_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Opaque handle reserved for operation state. The definition remains private
 * so that internal C5.0 data structures do not become part of the ABI.
 */
typedef struct c50_context c50_context;

/* Opaque handle reserved for a trained tree, ruleset, or boosted ensemble. */
typedef struct c50_model c50_model;

/* Status values returned by public API operations. */
typedef enum c50_status
{
    C50_STATUS_OK = 0,
    C50_STATUS_INVALID_ARGUMENT = 1,
    C50_STATUS_OUT_OF_MEMORY = 2,
    C50_STATUS_IO_ERROR = 3,
    C50_STATUS_PARSE_ERROR = 4,
    C50_STATUS_UNSUPPORTED = 5,
    C50_STATUS_INTERNAL_ERROR = 6
} c50_status;

/* Serialized classifier representations supported by C5.0. */
typedef enum c50_model_kind
{
    C50_MODEL_TREE = 0,
    C50_MODEL_RULES = 1
} c50_model_kind;

/*
 * Allocate a context and store it in *out_context. On success, the caller
 * owns the context and must release it with c50_context_destroy(). If
 * out_context is not NULL, this function sets *out_context to NULL before
 * reporting failure.
 */
c50_status c50_context_create(c50_context **out_context);

/* Release a context. A NULL context is allowed and has no effect. */
void c50_context_destroy(c50_context *context);

/*
 * Return the status of the last operation on context. A NULL context returns
 * C50_STATUS_INVALID_ARGUMENT.
 */
c50_status c50_context_last_status(const c50_context *context);

/*
 * Return the last error detail recorded by context. The returned pointer is
 * never NULL and remains valid until the next operation on the context or
 * until the context is destroyed. A new context returns an empty string. A
 * NULL context returns a static invalid-argument message.
 */
const char *c50_context_error_message(const c50_context *context);

/* Return a static, non-NULL description of status. */
const char *c50_status_message(c50_status status);

#ifdef __cplusplus
}
#endif

#endif
