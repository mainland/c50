/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <c50/c50.h>

#include "c50_api_internal.h"
#include "defns.i"
#include "extern.i"

struct c50_model
{
    c50_model_kind kind;
    char *names_data;
    size_t names_size;
    char *model_data;
    size_t model_size;
    char *costs_data;
    size_t costs_size;
};

struct c50_predictions
{
    size_t row_count;
    size_t class_count;
    char **class_names;
    size_t *class_indices;
    double *confidences;
    double *scores;
};

typedef struct c50_model_load_state
{
    c50_model_kind kind;
    const char *names_data;
    size_t names_size;
    const char *model_data;
    size_t model_size;
    const char *costs_data;
    size_t costs_size;
    c50_model *model;
} c50_model_load_state;

typedef struct c50_train_state
{
    c50_model_kind kind;
    c50_options options;
    const char *names_data;
    size_t names_size;
    const char *training_data;
    size_t training_size;
    const char *costs_data;
    size_t costs_size;
    FILE *diagnostics;
    c50_model *model;
} c50_train_state;

typedef struct c50_predict_state
{
    const c50_model *model;
    const char *cases_data;
    size_t cases_size;
    c50_predictions *predictions;
} c50_predict_state;

static char *CopyInput(c50_context *Context, const char *data, size_t size)
{
    char *copy;

    if ( ! size ) return NULL;
    copy = Pmalloc(Context, size);
    memcpy(copy, data, size);
    return copy;
}

void c50_options_init(c50_options *options)
{
    if ( ! options ) return;
    memset(options, 0, sizeof(*options));
    options->struct_size = sizeof(*options);
    options->trials = 1;
    options->global_pruning = 1;
    options->minimum_cases = 2;
    options->confidence_factor = 0.25;
}

void c50_model_destroy(c50_model *model)
{
    if ( ! model ) return;

    free(model->names_data);
    free(model->model_data);
    free(model->costs_data);
    free(model);
}

c50_model_kind c50_model_get_kind(const c50_model *model)
{
    return model ? model->kind : C50_MODEL_TREE;
}

const char *c50_model_names_data(const c50_model *model, size_t *size)
{
    if ( size ) *size = model ? model->names_size : 0;
    return model ? model->names_data : NULL;
}

const char *c50_model_serialized_data(const c50_model *model, size_t *size)
{
    if ( size ) *size = model ? model->model_size : 0;
    return model ? model->model_data : NULL;
}

const char *c50_model_costs_data(const c50_model *model, size_t *size)
{
    if ( size ) *size = model ? model->costs_size : 0;
    return model ? model->costs_data : NULL;
}

void c50_predictions_destroy(c50_predictions *predictions)
{
    size_t class_index;

    if ( ! predictions ) return;
    if ( predictions->class_names )
    {
        for ( class_index = 0; class_index < predictions->class_count;
              class_index++ )
        {
            free(predictions->class_names[class_index]);
        }
    }
    free(predictions->class_names);
    free(predictions->class_indices);
    free(predictions->confidences);
    free(predictions->scores);
    free(predictions);
}

size_t c50_predictions_row_count(const c50_predictions *predictions)
{
    return predictions ? predictions->row_count : 0;
}

size_t c50_predictions_class_count(const c50_predictions *predictions)
{
    return predictions ? predictions->class_count : 0;
}

const char *c50_predictions_class_name(const c50_predictions *predictions,
                                       size_t class_index)
{
    if ( ! predictions || class_index >= predictions->class_count ) return NULL;
    return predictions->class_names[class_index];
}

size_t c50_predictions_class_index(const c50_predictions *predictions,
                                   size_t row)
{
    if ( ! predictions || row >= predictions->row_count ) return (size_t) -1;
    return predictions->class_indices[row];
}

double c50_predictions_confidence(const c50_predictions *predictions,
                                  size_t row)
{
    if ( ! predictions || row >= predictions->row_count ) return 0;
    return predictions->confidences[row];
}

double c50_predictions_score(const c50_predictions *predictions,
                             size_t row, size_t class_index)
{
    if ( ! predictions || row >= predictions->row_count ||
         class_index >= predictions->class_count ) return 0;
    return predictions->scores[row * predictions->class_count + class_index];
}

static void ParseModel(c50_context *Context, const c50_model *model)
{
    c50_input names_input, model_input, costs_input, *costs = NULL;

    Context->io.output = NULL;
    Context->io.file_stem = "memory";
    snprintf(Context->io.file_name, sizeof(Context->io.file_name), "%s", "memory.model");
    Context->options.rules = model->kind == C50_MODEL_RULES;
    Context->options.trials = 1;
    Context->trees.max_tree = -1;
    Context->options.sample_fraction = 0;

    c50_input_init_memory(&names_input, model->names_data, model->names_size);
    GetNames(Context, &names_input);

    c50_input_init_memory(&model_input, model->model_data, model->model_size);
    if ( model->costs_size )
    {
        c50_input_init_memory(&costs_input, model->costs_data,
                              model->costs_size);
        costs = &costs_input;
    }
    ReadHeaderMemory(Context, &model_input, costs);
    if ( Context->options.trials < 1 )
    {
        c50_record_error(Context, C50_STATUS_PARSE_ERROR,
                         "model contains no classifier entries");
        C50Exit(Context, 1);
    }

    Context->trees.max_tree = Context->options.trials - 1;
    if ( Context->options.rules )
    {
        Context->rules.sets = AllocZero(Context->options.trials + 1, CRuleSet);
        ForEach(Context->trees.trial, 0, Context->options.trials - 1)
        {
            InRulesAt(Context, &model_input, &Context->rules.sets[Context->trees.trial]);
        }
    }
    else
    {
        Context->trees.pruned = AllocZero(Context->options.trials + 1, Tree);
        ForEach(Context->trees.trial, 0, Context->options.trials - 1)
        {
            InTreeAt(Context, &model_input, &Context->trees.pruned[Context->trees.trial]);
        }
    }
}

static void LoadModel(c50_context *Context, void *user_data)
{
    c50_model_load_state *state = user_data;

    state->model = AllocZero(1, c50_model);
    state->model->kind = state->kind;
    state->model->names_size = state->names_size;
    state->model->names_data = CopyInput(Context, state->names_data,
                                         state->names_size);
    state->model->model_size = state->model_size;
    state->model->model_data = CopyInput(Context, state->model_data,
                                         state->model_size);
    state->model->costs_size = state->costs_size;
    state->model->costs_data = CopyInput(Context, state->costs_data,
                                         state->costs_size);

    ParseModel(Context, state->model);
}

static void CleanupModelLoad(c50_context *Context, void *user_data)
{
    c50_model_load_state *state = user_data;

    Cleanup(Context);
    Context->io.output = NULL;
    if ( c50_context_last_status(Context) != C50_STATUS_OK )
    {
        c50_model_destroy(state->model);
        state->model = NULL;
    }
}

static c50_status InvalidArgument(c50_context *Context, const char *message)
{
    return c50_set_context_error(Context, C50_STATUS_INVALID_ARGUMENT,
                                 message);
}

static int IsBoolean(int value)
{
    return value == 0 || value == 1;
}

static const char *ValidateOptions(const c50_options *options)
{
    if ( options->struct_size != sizeof(*options) )
    {
        return "options has an incompatible struct_size";
    }
    if ( options->trials < 1 || options->trials > 1000 )
    {
        return "options.trials must be between 1 and 1000";
    }
    if ( ! IsBoolean(options->subset_splits) ||
         ! IsBoolean(options->winnow) ||
         ! IsBoolean(options->global_pruning) ||
         ! IsBoolean(options->probabilistic_thresholds) ||
         ! IsBoolean(options->ignore_costs) )
    {
        return "boolean options must be zero or one";
    }
    if ( ! isfinite(options->minimum_cases) ||
         options->minimum_cases < 1 || options->minimum_cases > 1000000 )
    {
        return "options.minimum_cases must be between 1 and 1000000";
    }
    if ( ! isfinite(options->confidence_factor) ||
         options->confidence_factor < 0 || options->confidence_factor > 1 )
    {
        return "options.confidence_factor must be between 0 and 1";
    }
    if ( ! isfinite(options->sample_fraction) ||
         options->sample_fraction < 0 || options->sample_fraction > 0.999 )
    {
        return "options.sample_fraction must be between 0 and 0.999";
    }
    if ( options->random_seed > 4095 )
    {
        return "options.random_seed must be between 0 and 4095";
    }
    return NULL;
}

static void TrainModel(c50_context *Context, void *user_data)
{
    c50_train_state *state = user_data;
    c50_input names_input, training_input, costs_input;
    unsigned char *serialized;
    size_t serialized_size;

    state->diagnostics = tmpfile();
    if ( ! state->diagnostics )
    {
        c50_record_error(Context, C50_STATUS_IO_ERROR,
                         "could not create training diagnostics stream");
        C50Exit(Context, 1);
    }

    Context->io.output = state->diagnostics;
    Context->progress.update_file = state->diagnostics;
    Context->io.file_stem = "memory";
    Context->io.attribute_exclusions = 0;
    Context->io.random_initial_seed = (int) state->options.random_seed;
    Context->schema.max_discrete_value = 3;
    Context->cases.max_case = -1;
    Context->trees.max_tree = -1;
    Context->attributes_winnowed = false;
    Context->costs.unit_weights = true;
    Context->costs.weighted = false;
    Context->options.verbosity = 0;
    Context->options.trials = (int) state->options.trials;
    Context->options.folds = 10;
    Context->options.utility_bands = 0;
    Context->options.subset_splits = state->options.subset_splits;
    Context->options.boosting = state->options.trials > 1;
    Context->options.probabilistic_thresholds =
        state->options.probabilistic_thresholds;
    Context->options.rules = state->kind == C50_MODEL_RULES;
    Context->options.cross_validation = false;
    Context->options.ignore_costs = state->options.ignore_costs;
    Context->options.winnow = state->options.winnow;
    Context->options.global_pruning = state->options.global_pruning;
    Context->options.minimum_cases = (float) state->options.minimum_cases;
    Context->options.leaf_ratio = 0;
    Context->options.confidence_factor =
        (float) state->options.confidence_factor;
    Context->options.sample_fraction =
        (float) state->options.sample_fraction;
    Context->splits.sample_fraction = 1;
    Context->last_model_extension = NULL;

    c50_output_init_memory(&Context->classifier_output);
    Context->classifier_output_active = true;

    c50_input_init_memory(&names_input, state->names_data, state->names_size);
    GetNames(Context, &names_input);

    Context->cases.some_missing =
        AllocZero(Context->schema.max_attribute + 1, Boolean);
    Context->cases.some_not_applicable =
        AllocZero(Context->schema.max_attribute + 1, Boolean);

    c50_input_init_memory(&training_input, state->training_data,
                          state->training_size);
    GetDataInput(Context, &training_input, true, false);
    if ( Context->cases.max_case < 0 )
    {
        c50_record_error(Context, C50_STATUS_PARSE_ERROR,
                         "training data contains no cases");
        C50Exit(Context, 1);
    }

    if ( ! Context->options.ignore_costs && state->costs_size )
    {
        c50_input_init_memory(&costs_input, state->costs_data,
                              state->costs_size);
        GetMCostsInput(Context, &costs_input);
    }

    InitialiseTreeData(Context);
    if ( Context->options.rules )
    {
        Context->rules.sets =
            AllocZero(Context->options.trials + 1, CRuleSet);
    }
    if ( Context->options.winnow )
    {
        NotifyStage(Context, WINNOWATTS);
        Progress(Context, -Context->schema.max_attribute);
        WinnowAtts(Context);
    }
    ConstructClassifiers(Context);

    state->model = AllocZero(1, c50_model);
    serialized = c50_output_take_memory(&Context->classifier_output,
                                        &serialized_size);
    Context->classifier_output_active = false;
    if ( ! serialized )
    {
        c50_record_error(Context, C50_STATUS_OUT_OF_MEMORY,
                         "could not finalize serialized classifier");
        C50Exit(Context, 1);
    }

    state->model->kind = state->kind;
    state->model->names_size = state->names_size;
    state->model->names_data =
        CopyInput(Context, state->names_data, state->names_size);
    state->model->model_size = serialized_size;
    state->model->model_data = (char *) serialized;
    if ( ! state->options.ignore_costs )
    {
        state->model->costs_size = state->costs_size;
        state->model->costs_data =
            CopyInput(Context, state->costs_data, state->costs_size);
    }
}

static void CleanupTraining(c50_context *Context, void *user_data)
{
    c50_train_state *state = user_data;

    Context->progress.update_file = NULL;
    Cleanup(Context);
    c50_clear_prediction_state(Context);
    Context->io.output = NULL;
    if ( state->diagnostics ) fclose(state->diagnostics);
    state->diagnostics = NULL;
    if ( c50_context_last_status(Context) != C50_STATUS_OK )
    {
        c50_model_destroy(state->model);
        state->model = NULL;
    }
}

c50_status c50_model_train(c50_context *Context, c50_model_kind kind,
                           const c50_options *options,
                           const char *names_data, size_t names_size,
                           const char *training_data, size_t training_size,
                           const char *costs_data, size_t costs_size,
                           c50_model **out_model)
{
    c50_train_state state;
    const char *error;
    c50_status status;

    if ( out_model ) *out_model = NULL;
    if ( ! Context ) return C50_STATUS_INVALID_ARGUMENT;
    if ( ! out_model ) return InvalidArgument(Context, "out_model is NULL");
    if ( kind != C50_MODEL_TREE && kind != C50_MODEL_RULES )
    {
        return InvalidArgument(Context, "invalid model kind");
    }
    if ( ! names_data || ! names_size || ! training_data || ! training_size )
    {
        return InvalidArgument(Context,
                               "names and training inputs are required");
    }
    if ( ! costs_data && costs_size )
    {
        return InvalidArgument(Context, "costs_data is NULL");
    }
    if ( memchr(names_data, '\0', names_size) ||
         memchr(training_data, '\0', training_size) ||
         (costs_size && memchr(costs_data, '\0', costs_size)) )
    {
        return InvalidArgument(Context, "text inputs contain a NUL byte");
    }

    memset(&state, 0, sizeof(state));
    state.kind = kind;
    c50_options_init(&state.options);
    if ( options )
    {
        if ( options->struct_size != sizeof(*options) )
        {
            return InvalidArgument(Context,
                                   "options has an incompatible struct_size");
        }
        state.options = *options;
    }
    if ( (error = ValidateOptions(&state.options)) )
    {
        return InvalidArgument(Context, error);
    }
    state.names_data = names_data;
    state.names_size = names_size;
    state.training_data = training_data;
    state.training_size = training_size;
    state.costs_data = costs_data;
    state.costs_size = costs_size;

    status = c50_run_operation(Context, TrainModel, CleanupTraining, &state);
    if ( status == C50_STATUS_OK ) *out_model = state.model;
    return status;
}

c50_status c50_model_load(c50_context *Context, c50_model_kind kind,
                          const char *names_data, size_t names_size,
                          const char *model_data, size_t model_size,
                          const char *costs_data, size_t costs_size,
                          c50_model **out_model)
{
    c50_model_load_state state;
    c50_status status;

    if ( out_model ) *out_model = NULL;
    if ( ! Context ) return C50_STATUS_INVALID_ARGUMENT;
    if ( ! out_model ) return InvalidArgument(Context, "out_model is NULL");
    if ( kind != C50_MODEL_TREE && kind != C50_MODEL_RULES )
    {
        return InvalidArgument(Context, "invalid model kind");
    }
    if ( ! names_data || ! names_size || ! model_data || ! model_size )
    {
        return InvalidArgument(Context, "names and model inputs are required");
    }
    if ( ! costs_data && costs_size )
    {
        return InvalidArgument(Context, "costs_data is NULL");
    }
    if ( memchr(names_data, '\0', names_size) ||
         memchr(model_data, '\0', model_size) ||
         (costs_size && memchr(costs_data, '\0', costs_size)) )
    {
        return InvalidArgument(Context, "text inputs contain a NUL byte");
    }

    memset(&state, 0, sizeof(state));
    state.kind = kind;
    state.names_data = names_data;
    state.names_size = names_size;
    state.model_data = model_data;
    state.model_size = model_size;
    state.costs_data = costs_data;
    state.costs_size = costs_size;

    status = c50_run_operation(Context, LoadModel, CleanupModelLoad, &state);
    if ( status == C50_STATUS_OK ) *out_model = state.model;
    return status;
}

static void PredictModel(c50_context *Context, void *user_data)
{
    c50_predict_state *state = user_data;
    c50_input cases_input;
    c50_predictions *predictions;
    CaseNo row;
    ClassNo class_number, predicted;

    ParseModel(Context, state->model);

    Context->cases.some_missing = AllocZero(Context->schema.max_attribute + 1, Boolean);
    Context->cases.some_not_applicable = AllocZero(Context->schema.max_attribute + 1, Boolean);
    if ( Context->options.rules ) Context->most_specific_rules = Alloc(Context->schema.max_class + 1, CRule);
    Context->default_class =
        ( Context->options.rules ? Context->rules.sets[0]->SDefault : Context->trees.pruned[0]->Leaf );
    Context->class_sum = AllocZero(Context->schema.max_class + 1, float);
    Context->votes = AllocZero(Context->schema.max_class + 1, float);
    Context->trial_predictions = AllocZero(Context->options.trials, ClassNo);

    c50_input_init_memory(&cases_input, state->cases_data, state->cases_size);
    GetDataInput(Context, &cases_input, false, true);

    predictions = AllocZero(1, c50_predictions);
    state->predictions = predictions;
    predictions->class_count = Context->schema.max_class;
    predictions->class_names = AllocZero(Context->schema.max_class, char *);
    ForEach(class_number, 1, Context->schema.max_class)
    {
        predictions->class_names[class_number - 1] =
            CopyInput(Context, Context->schema.class_names[class_number],
                      strlen(Context->schema.class_names[class_number]) + 1);
    }

    predictions->row_count = Context->cases.max_case + 1;
    if ( predictions->row_count )
    {
        predictions->class_indices =
            AllocZero(predictions->row_count, size_t);
        predictions->confidences =
            AllocZero(predictions->row_count, double);
        predictions->scores =
            AllocZero(predictions->row_count * predictions->class_count, double);
    }

    ForEach(row, 0, Context->cases.max_case)
    {
        predicted = Classify(Context, Context->cases.records[row]);
        predictions->class_indices[row] = predicted - 1;
        predictions->confidences[row] = Context->confidence;
        ForEach(class_number, 1, Context->schema.max_class)
        {
            predictions->scores[
                row * predictions->class_count + class_number - 1] =
                Context->class_sum[class_number];
        }
    }
}

static void CleanupPrediction(c50_context *Context, void *user_data)
{
    c50_predict_state *state = user_data;

    Cleanup(Context);
    c50_clear_prediction_state(Context);
    Context->io.output = NULL;
    if ( c50_context_last_status(Context) != C50_STATUS_OK )
    {
        c50_predictions_destroy(state->predictions);
        state->predictions = NULL;
    }
}

c50_status c50_model_predict(c50_context *Context, const c50_model *model,
                             const char *cases_data, size_t cases_size,
                             c50_predictions **out_predictions)
{
    c50_predict_state state;
    c50_status status;

    if ( out_predictions ) *out_predictions = NULL;
    if ( ! Context ) return C50_STATUS_INVALID_ARGUMENT;
    if ( ! out_predictions )
    {
        return InvalidArgument(Context, "out_predictions is NULL");
    }
    if ( ! model ) return InvalidArgument(Context, "model is NULL");
    if ( ! cases_data && cases_size )
    {
        return InvalidArgument(Context, "cases_data is NULL");
    }
    if ( cases_size && memchr(cases_data, '\0', cases_size) )
    {
        return InvalidArgument(Context, "cases input contains a NUL byte");
    }

    memset(&state, 0, sizeof(state));
    state.model = model;
    state.cases_data = cases_data;
    state.cases_size = cases_size;

    status = c50_run_operation(Context, PredictModel, CleanupPrediction, &state);
    if ( status == C50_STATUS_OK ) *out_predictions = state.predictions;
    return status;
}
