/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef C50_API_INTERNAL_H
#define C50_API_INTERNAL_H

#include <setjmp.h>

#include <c50/c50.h>

#include "c50_input.h"
#include "c50_rng.h"

#define C50_ERROR_MESSAGE_CAPACITY 1024
#define C50_LINE_BUFFER_CAPACITY 10000

struct c50_implicit_state;
struct _datablockrec;
struct _environment;
union _attribute_value;
struct _def_elt;
struct _rulerec;
struct _rulesetrec;
struct _treerec;

#ifdef USEDOUBLE
typedef double c50_continuous_value;
#else
typedef float c50_continuous_value;
#endif

typedef struct
{
    int class_attribute;
    int label_attribute;
    int case_weight_attribute;
    int max_attribute;
    int max_class;
    int max_discrete_value;
    char **class_names;
    char **attribute_names;
    char ***attribute_value_names;
    int *max_attribute_value;
    char *special_status;
    struct _def_elt **attribute_definitions;
    int **attribute_definition_uses;
    c50_continuous_value *class_thresholds;
} c50_schema_state;

typedef struct
{
    union _attribute_value **records;
    union _attribute_value **saved_records;
    int max_case;
    struct _datablockrec *memory_blocks;
    int block_size;
    unsigned char *some_missing;
    unsigned char *some_not_applicable;
} c50_case_state;

typedef struct
{
    struct _treerec **raw;
    struct _treerec **pruned;
    struct _treerec *winnow;
    int trial;
    int max_tree;
} c50_tree_state;

typedef struct
{
    struct _rulerec **rules;
    struct _rulesetrec **sets;
    int count;
    int capacity;
} c50_rule_state;

typedef struct
{
    float **matrix;
    float **normalized_matrix;
    float *weight_multipliers;
    unsigned char unit_weights;
    unsigned char weighted;
} c50_cost_state;

typedef struct
{
    double *class_frequencies;
    float *boost_vote_block;
    struct _environment *environment;
    float *attribute_importance;
    unsigned char *split_attributes;
    unsigned char *used_attributes;
} c50_training_state;

typedef struct
{
    int *utility_errors;
    int *utility_bands;
    double *utility_costs;
} c50_evaluation_state;

typedef struct
{
    int verbosity;
    int trials;
    int folds;
    int utility_bands;
    unsigned char subset_splits;
    unsigned char boosting;
    unsigned char probabilistic_thresholds;
    unsigned char rules;
    unsigned char cross_validation;
    unsigned char ignore_costs;
    unsigned char winnow;
    unsigned char global_pruning;
    float minimum_cases;
    float leaf_ratio;
    float confidence_factor;
    float sample_fraction;
} c50_options_state;

typedef struct
{
    double **discrete_frequencies;
    float *gain;
    float *information;
    float *estimated_max_gain_ratio;
    c50_continuous_value *thresholds;
    double base_information;
    double **bell_numbers;
    float sample_fraction;
    int *possible_cuts;
    unsigned char *tested_attributes;
    unsigned char ***subsets;
    int *subset_counts;
} c50_split_state;

typedef struct
{
    float attribute_test_bits;
    float *branch_bits;
    int *attribute_values;
} c50_rule_build_state;

struct c50_context
{
    c50_status status;
    char error_message[C50_ERROR_MESSAGE_CAPACITY];
    jmp_buf exit_target;
    int sample_from;
    int suppress_error_messages;
    int delimiter;
    int max_label;
    c50_schema_state schema;
    c50_case_state cases;
    c50_tree_state trees;
    c50_rule_state rules;
    c50_cost_state costs;
    c50_training_state training;
    c50_evaluation_state evaluation;
    c50_options_state options;
    c50_split_state splits;
    c50_rule_build_state rule_build;
    double average_case_weight;
    char *ignored_values;
    int ignored_values_size;
    int ignored_values_offset;
    int attributes_winnowed;
    char line_buffer[C50_LINE_BUFFER_CAPACITY];
    char *line_buffer_position;
    struct c50_implicit_state *implicit_state;
    c50_input classifier_input;
    const char *last_model_extension;
    int model_entry;
    char property_name[20];
    char *property_value;
    int property_value_size;
    int *active_rules;
    int active_rule_count;
    int active_rule_capacity;
    float confidence;
    float *class_sum;
    float *votes;
    int *trial_predictions;
    struct _rulerec **most_specific_rules;
    int default_class;
    KRState random;
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

/* Release context-owned prediction workspace. */
void c50_clear_prediction_state(c50_context *context);

#endif
