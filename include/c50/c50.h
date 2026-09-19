/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef C50_C50_H
#define C50_C50_H

#include <stddef.h>

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

/* Opaque owner of one batch of prediction results. */
typedef struct c50_predictions c50_predictions;

/*
 * The imported core uses process-global mutable state. C5.0 operations must be
 * serialized across all contexts. Callers must also prevent concurrent access
 * to the same context or model handle.
 */

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

/*
 * Load and validate a serialized classifier. Inputs are copied, so they need
 * not remain valid after this function returns. costs_data may be NULL only
 * when costs_size is zero and the classifier does not require costs. On
 * failure, *out_model is NULL and context contains the diagnostic.
 */
c50_status c50_model_load(c50_context *context, c50_model_kind kind,
                          const char *names_data, size_t names_size,
                          const char *model_data, size_t model_size,
                          const char *costs_data, size_t costs_size,
                          c50_model **out_model);

/* Release a model. A NULL model is allowed and has no effect. */
void c50_model_destroy(c50_model *model);

/* Return the serialized representation kind. model must not be NULL. */
c50_model_kind c50_model_get_kind(const c50_model *model);

/*
 * Predict cases encoded in the C5.0 data-file syntax. Each row must contain
 * all input fields plus a final class field, which may be "?" when unknown.
 * cases_data may be NULL only when cases_size is zero. On success, the caller
 * owns *out_predictions and must release it with c50_predictions_destroy().
 * An empty input produces an empty result with the model's class metadata.
 */
c50_status c50_model_predict(c50_context *context, const c50_model *model,
                             const char *cases_data, size_t cases_size,
                             c50_predictions **out_predictions);

/* Release prediction results. A NULL pointer is allowed and has no effect. */
void c50_predictions_destroy(c50_predictions *predictions);

size_t c50_predictions_row_count(const c50_predictions *predictions);
size_t c50_predictions_class_count(const c50_predictions *predictions);

/* Return a borrowed class name, or NULL when class_index is out of range. */
const char *c50_predictions_class_name(const c50_predictions *predictions,
                                       size_t class_index);

/*
 * Access one row. Class indices are zero-based. Invalid row or class indices
 * return (size_t) -1 for the class index and zero for numeric values.
 */
size_t c50_predictions_class_index(const c50_predictions *predictions,
                                   size_t row);
double c50_predictions_confidence(const c50_predictions *predictions,
                                  size_t row);
double c50_predictions_score(const c50_predictions *predictions,
                             size_t row, size_t class_index);

#ifdef __cplusplus
}
#endif

#endif
