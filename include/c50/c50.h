/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file c50/c50.h
 * @brief Public C interface to the C5.0 GPL classifier core.
 *
 * The API accepts the contents of C5.0 names, data, costs, and serialized-model
 * files as length-delimited memory buffers. Training and loading copy all data
 * retained by a model. Prediction results own their class metadata and values.
 *
 * @par Thread safety
 * Independent contexts may be used concurrently. A context may have only one
 * active operation, so callers must serialize calls that use the same context.
 * Models are immutable after construction and may be shared by concurrent
 * operations that use different contexts. A model or result object must not be
 * destroyed while another thread is accessing it.
 */

#ifndef C50_C50_H
#define C50_C50_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @defgroup c50_c_api C API
 *  @brief Context-based C interface to C5.0 training and prediction.
 *  @{
 */

/**
 * @brief Opaque owner of mutable operation state.
 *
 * A context records the status and diagnostic for its most recent operation.
 * It does not own models returned by training or loading.
 */
typedef struct c50_context c50_context;

/** @brief Opaque, immutable tree, ruleset, or boosted ensemble. */
typedef struct c50_model c50_model;

/** @brief Opaque owner of one batch of prediction results. */
typedef struct c50_predictions c50_predictions;

/** @brief Status values returned by public API operations. */
typedef enum c50_status
{
    /** The operation completed successfully. */
    C50_STATUS_OK = 0,
    /** A pointer, size, enum value, or option value was invalid. */
    C50_STATUS_INVALID_ARGUMENT = 1,
    /** Memory allocation failed. */
    C50_STATUS_OUT_OF_MEMORY = 2,
    /** An internal stream operation or required costs input failed. */
    C50_STATUS_IO_ERROR = 3,
    /** A names, data, costs, or serialized-model input could not be parsed. */
    C50_STATUS_PARSE_ERROR = 4,
    /** The requested operation is not supported. */
    C50_STATUS_UNSUPPORTED = 5,
    /** An internal invariant or operation-state check failed. */
    C50_STATUS_INTERNAL_ERROR = 6
} c50_status;

/** @brief Serialized classifier representations supported by C5.0. */
typedef enum c50_model_kind
{
    /** A decision tree or boosted ensemble of decision trees. */
    C50_MODEL_TREE = 0,
    /** A ruleset or boosted ensemble of rulesets. */
    C50_MODEL_RULES = 1
} c50_model_kind;

/**
 * @brief Options that control classifier construction.
 *
 * Initialize the complete structure with c50_options_init() before changing
 * individual fields. Reserved fields must remain zero. c50_model_train()
 * validates every field before starting an operation.
 */
typedef struct c50_options
{
    /** Structure size used to detect incompatible API versions. */
    size_t struct_size;
    /** Number of classifiers to construct, in the range 1 through 1000. */
    unsigned int trials;
    /** Whether discrete attributes may be split into value subsets. */
    int subset_splits;
    /** Whether to remove attributes that appear unhelpful before training. */
    int winnow;
    /** Whether to perform global as well as local tree pruning. */
    int global_pruning;
    /** Whether to use probabilistic thresholds for continuous attributes. */
    int probabilistic_thresholds;
    /** Whether to ignore the supplied misclassification costs. */
    int ignore_costs;
    /** Minimum cases represented by two branches, from 1 through 1000000. */
    double minimum_cases;
    /** Pruning confidence factor expressed as a fraction in [0, 1]. */
    double confidence_factor;
    /** Training sample fraction in [0, 0.999], or zero for no sampling. */
    double sample_fraction;
    /** Sampling seed in the range 0 through 4095. */
    unsigned int random_seed;
    /** Reserved for ABI-compatible extension and required to remain zero. */
    unsigned int reserved[8];
} c50_options;

/**
 * @brief Initialize training options to deterministic single-tree defaults.
 *
 * The defaults are one trial, no subset splits, no winnowing, global pruning,
 * ordinary thresholds, costs enabled, two minimum cases, a confidence factor
 * of 0.25, no sampling, and random seed zero.
 *
 * @param[out] options Structure to initialize. A null pointer has no effect.
 */
void c50_options_init(c50_options *options);

/**
 * @brief Allocate an independent operation context.
 *
 * @param[out] out_context Receives the new context on success. When non-null,
 *                         it is set to null before allocation is attempted.
 * @retval C50_STATUS_OK A context was allocated.
 * @retval C50_STATUS_INVALID_ARGUMENT @p out_context was null.
 * @retval C50_STATUS_OUT_OF_MEMORY Allocation failed.
 *
 * The caller owns the returned context and must release it with
 * c50_context_destroy().
 */
c50_status c50_context_create(c50_context **out_context);

/**
 * @brief Release an operation context.
 *
 * @param[in] context Context to release. A null pointer has no effect.
 *
 * @pre No operation may be active on @p context.
 */
void c50_context_destroy(c50_context *context);

/**
 * @brief Return the status of the most recent operation on a context.
 *
 * A newly created context reports C50_STATUS_OK. A later operation resets the
 * previous status before doing work, so a successful operation clears an
 * earlier failure.
 *
 * @param[in] context Context to query.
 * @return The most recent status, or C50_STATUS_INVALID_ARGUMENT when
 *         @p context is null.
 */
c50_status c50_context_last_status(const c50_context *context);

/**
 * @brief Return diagnostic detail for the most recent context operation.
 *
 * @param[in] context Context to query.
 * @return A non-null, null-terminated message. The pointer remains valid until
 *         the next operation on @p context or until the context is destroyed.
 *         A new or successfully reused context returns an empty string. A null
 *         context returns a static invalid-argument message.
 */
const char *c50_context_error_message(const c50_context *context);

/**
 * @brief Return a short description of a status value.
 *
 * @param[in] status Status to describe.
 * @return A static, non-null string. Unknown values produce a generic message.
 */
const char *c50_status_message(c50_status status);

/**
 * @brief Load and validate a serialized classifier from memory.
 *
 * Required inputs must be non-null and nonempty. All text buffers reject an
 * embedded null byte. @p costs_data may be null only when @p costs_size is
 * zero. A classifier whose serialized header declares costs requires matching
 * costs input.
 *
 * On success, the model owns copies of all supplied inputs. On every failure,
 * @p *out_model is null and a non-null context records the diagnostic.
 *
 * @param[in,out] context Context used for parsing and diagnostics.
 * @param[in] kind Serialized representation in @p model_data.
 * @param[in] names_data C5.0 names-file contents.
 * @param[in] names_size Number of bytes in @p names_data.
 * @param[in] model_data Serialized tree or rules-file contents.
 * @param[in] model_size Number of bytes in @p model_data.
 * @param[in] costs_data Optional C5.0 costs-file contents.
 * @param[in] costs_size Number of bytes in @p costs_data.
 * @param[out] out_model Receives an owned model on success.
 * @return C50_STATUS_OK on success, or a status describing validation,
 *         allocation, input, or parsing failure.
 */
c50_status c50_model_load(c50_context *context, c50_model_kind kind,
                          const char *names_data, size_t names_size,
                          const char *model_data, size_t model_size,
                          const char *costs_data, size_t costs_size,
                          c50_model **out_model);

/**
 * @brief Train a classifier from C5.0 text held in memory.
 *
 * @p names_data and @p training_data must be non-null and nonempty. All text
 * buffers reject an embedded null byte. @p costs_data may be null only when
 * @p costs_size is zero. Passing null for @p options selects the defaults from
 * c50_options_init().
 *
 * On success, the model owns copies of its names input, serialized classifier,
 * and applicable costs input. On every failure, @p *out_model is null and a
 * non-null context records the diagnostic.
 *
 * @param[in,out] context Context used for training and diagnostics.
 * @param[in] kind Representation to construct.
 * @param[in] options Training options, or null for defaults.
 * @param[in] names_data C5.0 names-file contents.
 * @param[in] names_size Number of bytes in @p names_data.
 * @param[in] training_data C5.0 data-file contents containing training cases.
 * @param[in] training_size Number of bytes in @p training_data.
 * @param[in] costs_data Optional C5.0 costs-file contents.
 * @param[in] costs_size Number of bytes in @p costs_data.
 * @param[out] out_model Receives an owned model on success.
 * @return C50_STATUS_OK on success, or a status describing validation,
 *         allocation, input, or training failure.
 */
c50_status c50_model_train(c50_context *context, c50_model_kind kind,
                           const c50_options *options,
                           const char *names_data, size_t names_size,
                           const char *training_data, size_t training_size,
                           const char *costs_data, size_t costs_size,
                           c50_model **out_model);

/**
 * @brief Release a model.
 * @param[in] model Model to release. A null pointer has no effect.
 * @pre No operation may be accessing @p model.
 */
void c50_model_destroy(c50_model *model);

/**
 * @brief Return a model's serialized representation kind.
 * @param[in] model Model to query.
 * @return The model kind.
 * @pre @p model must not be null.
 */
c50_model_kind c50_model_get_kind(const c50_model *model);

/**
 * @brief Borrow the names-file contents retained by a model.
 * @param[in] model Model to query.
 * @param[out] size Optional destination for the buffer size in bytes.
 * @return The borrowed buffer, or null when @p model is null.
 *
 * The buffer is not necessarily null-terminated and remains valid until the
 * model is destroyed.
 */
const char *c50_model_names_data(const c50_model *model, size_t *size);

/**
 * @brief Borrow the serialized classifier retained by a model.
 * @param[in] model Model to query.
 * @param[out] size Optional destination for the buffer size in bytes.
 * @return The borrowed buffer, or null when @p model is null.
 *
 * The buffer is not necessarily null-terminated and remains valid until the
 * model is destroyed.
 */
const char *c50_model_serialized_data(const c50_model *model, size_t *size);

/**
 * @brief Borrow the costs-file contents retained by a model.
 * @param[in] model Model to query.
 * @param[out] size Optional destination for the buffer size in bytes.
 * @return The borrowed buffer. A model without retained costs, or a null model,
 *         returns null and reports size zero.
 *
 * The buffer is not necessarily null-terminated and remains valid until the
 * model is destroyed.
 */
const char *c50_model_costs_data(const c50_model *model, size_t *size);

/**
 * @brief Predict a batch of cases encoded in C5.0 data-file syntax.
 *
 * Each nonempty row must contain every input field followed by a class field,
 * which may be `?` when the class is unknown. @p cases_data may be null only
 * when @p cases_size is zero and rejects an embedded null byte otherwise. An
 * empty input produces an empty result that still contains class metadata.
 *
 * @param[in,out] context Context used for parsing, prediction, and diagnostics.
 * @param[in] model Immutable model used for prediction.
 * @param[in] cases_data C5.0 case data, or null for an empty batch.
 * @param[in] cases_size Number of bytes in @p cases_data.
 * @param[out] out_predictions Receives owned results on success.
 * @return C50_STATUS_OK on success, or a status describing validation,
 *         allocation, input, or prediction failure. On failure,
 *         @p *out_predictions is null.
 */
c50_status c50_model_predict(c50_context *context, const c50_model *model,
                             const char *cases_data, size_t cases_size,
                             c50_predictions **out_predictions);

/**
 * @brief Release prediction results.
 * @param[in] predictions Results to release. A null pointer has no effect.
 * @pre No thread may be accessing @p predictions.
 */
void c50_predictions_destroy(c50_predictions *predictions);

/**
 * @brief Return the number of predicted rows.
 * @param[in] predictions Results to query.
 * @return The row count, or zero when @p predictions is null.
 */
size_t c50_predictions_row_count(const c50_predictions *predictions);

/**
 * @brief Return the number of classes represented by each result row.
 * @param[in] predictions Results to query.
 * @return The class count, or zero when @p predictions is null.
 */
size_t c50_predictions_class_count(const c50_predictions *predictions);

/**
 * @brief Borrow a class name from prediction metadata.
 * @param[in] predictions Results to query.
 * @param[in] class_index Zero-based class index.
 * @return A null-terminated name owned by @p predictions, or null when the
 *         result pointer or class index is invalid.
 */
const char *c50_predictions_class_name(const c50_predictions *predictions,
                                       size_t class_index);

/**
 * @brief Return the predicted class for a row.
 * @param[in] predictions Results to query.
 * @param[in] row Zero-based row index.
 * @return A zero-based class index, or `(size_t) -1` when the result pointer or
 *         row index is invalid.
 */
size_t c50_predictions_class_index(const c50_predictions *predictions,
                                   size_t row);

/**
 * @brief Return C5.0's confidence in the predicted class for a row.
 * @param[in] predictions Results to query.
 * @param[in] row Zero-based row index.
 * @return The confidence value, or zero when the result pointer or row index is
 *         invalid.
 */
double c50_predictions_confidence(const c50_predictions *predictions,
                                  size_t row);

/**
 * @brief Return C5.0's score for one row and class.
 *
 * Scores form a row-major `row_count` by `class_count` matrix when retrieved
 * for every pair of indices.
 *
 * @param[in] predictions Results to query.
 * @param[in] row Zero-based row index.
 * @param[in] class_index Zero-based class index.
 * @return The class score, or zero when the result pointer or either index is
 *         invalid.
 */
double c50_predictions_score(const c50_predictions *predictions,
                             size_t row, size_t class_index);

/** @} */

#ifdef __cplusplus
}
#endif

#endif
