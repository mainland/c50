/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef C50_API_INTERNAL_H
#define C50_API_INTERNAL_H

#include <stdio.h>
#include <time.h>

#include <memory>

#include <c50/c50.hpp>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "c50_input.h"
#include "c50_locale.h"
#include "c50_output.h"
#include "c50_rng.h"

struct c50_implicit_state;
struct _datablockrec;
struct _split_workspace;
union _attribute_value;
struct _def_elt;
struct _condrec;
struct _rulerec;
struct _rulesetrec;
struct _treerec;

#ifdef USEDOUBLE
typedef double c50_continuous_value;
#else
typedef float c50_continuous_value;
#endif

/* Base of the context's state records. They hold raw pointers that the
 * context owns, so copying one would duplicate that ownership. */
struct c50_state_record
{
    c50_state_record() = default;
    c50_state_record(const c50_state_record &) = delete;
    c50_state_record &operator=(const c50_state_record &) = delete;
};

struct c50_schema_state : c50_state_record
{
    int class_attribute = 0;
    int label_attribute = 0;
    int case_weight_attribute = 0;
    int max_attribute = 0;
    int max_class = 0;
    int owned_class_names = 0;
    int max_discrete_value = 0;
    char unknown_class_name[2]{};
    char other_attribute_value_name[8]{};
    char **class_names = nullptr;
    char **attribute_names = nullptr;
    char ***attribute_value_names = nullptr;
    int *max_attribute_value = nullptr;
    char *special_status = nullptr;
    struct _def_elt **attribute_definitions = nullptr;
    int **attribute_definition_uses = nullptr;
    c50_continuous_value *class_thresholds = nullptr;
};

struct c50_case_state : c50_state_record
{
    union _attribute_value **records = nullptr;
    union _attribute_value **saved_records = nullptr;
    int max_case = 0;
    struct _datablockrec *memory_blocks = nullptr;
    int block_size = 0;
    unsigned char *some_missing = nullptr;
    unsigned char *some_not_applicable = nullptr;
};

struct c50_tree_state : c50_state_record
{
    struct _treerec **raw = nullptr;
    struct _treerec **pruned = nullptr;
    struct _treerec *winnow = nullptr;
    int trial = 0;
    int max_tree = 0;
    int printed_subtree_count = 0;
    int printed_subtree_capacity = 0;
    struct _treerec **printed_subtrees = nullptr;
    std::vector<unsigned char> last_branches;
};

struct c50_rule_state : c50_state_record
{
    struct _rulerec **rules = nullptr;
    struct _rulesetrec **sets = nullptr;
    int count = 0;
    int capacity = 0;
};

struct c50_cost_state : c50_state_record
{
    float **matrix = nullptr;
    float **normalized_matrix = nullptr;
    float *weight_multipliers = nullptr;
    unsigned char unit_weights = 0;
    unsigned char weighted = 0;
};

/* Frees partially constructed split scratch without consulting the context. */
struct c50_split_workspace_deleter
{
    void operator()(struct _split_workspace *workspace) const noexcept;
};

struct c50_training_state : c50_state_record
{
    double *class_frequencies = nullptr;
    float *boost_vote_block = nullptr;
    std::unique_ptr<struct _split_workspace, c50_split_workspace_deleter> environment;
    float *attribute_importance = nullptr;
    unsigned char *split_attributes = nullptr;
    unsigned char *used_attributes = nullptr;
    int *wrong_predictions = nullptr;
};

struct c50_evaluation_state : c50_state_record
{
    int *utility_errors = nullptr;
    int *utility_bands = nullptr;
    double *utility_costs = nullptr;
};

struct c50_options_state : c50_state_record
{
    int verbosity = 0;
    int trials = 0;
    int folds = 0;
    int utility_bands = 0;
    unsigned char subset_splits = 0;
    unsigned char boosting = 0;
    unsigned char probabilistic_thresholds = 0;
    unsigned char rules = 0;
    unsigned char cross_validation = 0;
    unsigned char ignore_costs = 0;
    unsigned char winnow = 0;
    unsigned char global_pruning = 0;
    unsigned char stable_ties = 0;
    float minimum_cases = 0;
    float leaf_ratio = 0;
    float confidence_factor = 0;
    float sample_fraction = 0;
};

struct c50_split_state : c50_state_record
{
    double **discrete_frequencies = nullptr;
    float *gain = nullptr;
    float *information = nullptr;
    float *estimated_max_gain_ratio = nullptr;
    c50_continuous_value *thresholds = nullptr;
    double base_information = 0;
    double **bell_numbers = nullptr;
    float sample_fraction = 0;
    int *possible_cuts = nullptr;
    unsigned char *tested_attributes = nullptr;
    unsigned char ***subsets = nullptr;
    int *subset_counts = nullptr;
    unsigned char all_attributes_multi_valued = 0;
    unsigned char use_subsampling = 0;
    float average_gain_weight = 0;
    float mdl_weight = 0;
    int *discrete_attributes = nullptr;
    int discrete_attribute_count = 0;
    int max_leaves = 0;
    float value_threshold = 0;
    unsigned char sampled = 0;
    int *waiting_attributes = nullptr;
    int waiting_count = 0;
};

struct c50_rule_build_state : c50_state_record
{
    float attribute_test_bits = 0;
    float *branch_bits = nullptr;
    int *attribute_values = nullptr;
    unsigned char **fires = nullptr;
    unsigned char *compression_buffer = nullptr;
    int *coverage_counts = nullptr;
    int *list = nullptr;
    double *log_case_count = nullptr;
    double *log_factorial = nullptr;
    double *condition_errors = nullptr;
    double *condition_totals = nullptr;
    float *pessimistic_errors = nullptr;
    float *condition_costs = nullptr;
    unsigned char **condition_failed_by = nullptr;
    unsigned char *deleted_conditions = nullptr;
    struct _condrec **condition_stack = nullptr;
    int max_rule_depth = 0;
    int condition_count = 0;
    int best_condition = 0;
    int target_class = 0;
    short *failure_count = nullptr;
    short *local_failure_count = nullptr;
    int fail_zero = 0;
    int fail_one = 0;
    int fail_many = 0;
    int *successors = nullptr;
};

struct c50_rule_tree_state : c50_state_record
{
    struct _condrec **tests = nullptr;
    int test_count = 0;
    int test_capacity = 0;
    int *test_occurrences = nullptr;
    int *rule_conditions_satisfied = nullptr;
    unsigned char *tests_used = nullptr;
};

struct c50_rule_selection_state : c50_state_record
{
    float *delta_errors = nullptr;
    float *rule_bits = nullptr;
    float error_bits = 0;
    float correct_bits = 0;
    int **total_votes = nullptr;
    int *top_classes = nullptr;
    int *alternate_classes = nullptr;
    unsigned char *rules_included = nullptr;
    unsigned char *covered_cases = nullptr;
    unsigned char *coverage_block = nullptr;
    unsigned char **coverage_pointers = nullptr;
    int *last_covering_rule = nullptr;
};

struct c50_pruning_state : c50_state_record
{
    unsigned char **possible_values = nullptr;
    double maximum_extra_errors = 0;
    double total_extra_errors = 0;
    struct _treerec **minimum_cost_subtrees = nullptr;
    int minimum_cost_subtree_count = 0;
    float minimum_cost_complexity = 0;
    unsigned char recalculate_errors = 0;
    float confidence_coefficient = 0;
};

struct c50_cross_validation_state : c50_state_record
{
    union _attribute_value **blocked_cases = nullptr;
    float **results = nullptr;
    int *confusion_matrix = nullptr;
    int saved_folds = 0;
};

struct c50_progress_state : c50_state_record
{
    FILE *update_file = nullptr;
    float total = 0;
    float current = 0;
    int twentieth = 0;
    int last_stage = 0;
    int stage = 0;
};

struct c50_io_state : c50_state_record
{
    int line_number = 0;
    int error_count = 0;
    int attribute_exclusions = 0;
    int timestamp_base = 0;
    /* The single clock reading of the operation, from which the timestamp
       base year and the model's id date are both derived. */
    time_t operation_clock = 0;
    int random_initial_seed = 0;
    const char *file_stem = nullptr;
    std::string file_name;
    FILE *output = nullptr;
    char *option_argument = nullptr;
    char *option = nullptr;
    int option_index = 0;
    std::string label_buffer;
};

using c50_context = c50::detail::context_state;

/* Test access to a model's private state. Library code reads the state
   directly. */
struct c50::detail::model_access
{
    /* The tree depth limit that every parse of the model applies. */
    static std::optional<std::size_t> max_tree_depth(const model &source);
};

struct c50::detail::context_state
{
    context_state() = default;
    /* Releases everything that an interrupted operation left allocated. */
    ~context_state();
    context_state(const context_state &) = delete;
    context_state &operator=(const context_state &) = delete;

    std::optional<c50::error_code> error;
    std::string error_message;
    int operation_active = 0;
    /* True once the operation has printed its own diagnostic for the failure
     * that ends it, so that a front end need not print the message again. */
    int failure_reported = 0;
    unsigned int split_worker_count = 0;
    /* The most levels that the model-text parser accepts in a tree, or no
     * limit. ParseModel sets it from the model before every parse. */
    std::optional<std::size_t> max_serialized_tree_depth =
        c50::default_max_tree_depth;
    int sample_from = 0;
    int suppress_error_messages = 0;
    int delimiter = 0;
    int max_label = 0;
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
    c50_rule_tree_state rule_tree;
    c50_rule_selection_state rule_selection;
    c50_pruning_state pruning;
    c50_cross_validation_state cross_validation;
    c50_progress_state progress;
    c50_io_state io;
    /* The C locale that every operation and worker thread of this context
     * selects, so that results do not depend on the host locale. */
    c50_c_locale c_locale;
    double average_case_weight = 0;
    char *ignored_values = nullptr;
    int ignored_values_size = 0;
    int ignored_values_offset = 0;
    int attributes_winnowed = 0;
    std::string line_buffer;
    size_t line_buffer_position = 0;
    struct c50_implicit_state *implicit_state = nullptr;
    c50_output classifier_output;
    int classifier_output_active = 0;
    const char *last_model_extension = nullptr;
    int model_entry = 0;
    std::string property_name;
    char *property_value = nullptr;
    int property_value_size = 0;
    int *active_rules = nullptr;
    int active_rule_count = 0;
    int active_rule_capacity = 0;
    float confidence = 0;
    float *class_sum = nullptr;
    float *votes = nullptr;
    int *trial_predictions = nullptr;
    struct _rulerec **most_specific_rules = nullptr;
    int default_class = 0;
    KRState random{};
    Drand48State held_out_random{};
};

typedef void (*c50_operation_fn)(c50_context *context, void *user_data);
typedef void (*c50_operation_cleanup_fn)(c50_context *context,
                                         void *user_data);

/* Run an operation and always clean up. Preserve the original exception when
 * both the operation and its cleanup fail. Nested operations are rejected. */
void c50_run_operation(c50_context *context, c50_operation_fn operation,
                       c50_operation_cleanup_fn cleanup, void *user_data);

/* Record the first parser error, which may be reported after parsing resumes. */
void c50_record_error(c50_context *context, c50::error_code code,
                      const char *message);

/* Exit legacy control flow by unwinding C++ frames. Never terminate the host. */
[[noreturn]] void c50_abort_operation(c50_context *context, int exit_status);

/* Frees a vector of byte-set pointers that is indexed from 1 to Last, with the
 * sets it holds, as FreeVector does. A null vector is allowed. */
struct c50_set_vector_deleter
{
    int Last;
    void operator()(unsigned char **Vector) const noexcept;
};

/* Release every allocation that the context still holds. Cleanup, defined in
 * utility.cpp, frees what operations leave behind and is safe to repeat. */
void Cleanup(c50_context *Context);
void c50_clear_prediction_state(c50_context *context);
std::unique_ptr<c50_context> c50_make_context();

c50::model_inspection c50_build_inspection(c50_context *Context);

/* A null diagnostic stream discards output for in-memory library calls. */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 2, 3)))
#endif
int c50_diagnostic_printf(FILE *stream, const char *format, ...);
inline int c50_diagnostic_putc(int character, FILE *stream)
{
    return stream ? fputc(character, stream) : character;
}

#endif
