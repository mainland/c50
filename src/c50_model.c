/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

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

typedef struct c50_predict_state
{
    const c50_model *model;
    const char *cases_data;
    size_t cases_size;
    c50_predictions *predictions;
} c50_predict_state;

static char *CopyInput(const char *data, size_t size)
{
    char *copy;

    if ( ! size ) return NULL;
    copy = Pmalloc(size);
    memcpy(copy, data, size);
    return copy;
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

static void ParseModel(c50_context *context, const c50_model *model)
{
    c50_input names_input, model_input, costs_input, *costs = NULL;

    Of = NULL;
    FileStem = "memory";
    snprintf(Fn, sizeof(Fn), "%s", "memory.model");
    RULES = model->kind == C50_MODEL_RULES;
    TRIALS = 1;
    MaxTree = -1;
    SAMPLE = 0;

    c50_input_init_memory(&names_input, model->names_data, model->names_size);
    GetNames(context, &names_input);

    c50_input_init_memory(&model_input, model->model_data, model->model_size);
    if ( model->costs_size )
    {
        c50_input_init_memory(&costs_input, model->costs_data,
                              model->costs_size);
        costs = &costs_input;
    }
    ReadHeaderMemory(context, &model_input, costs);
    if ( TRIALS < 1 )
    {
        c50_record_error(C50_STATUS_PARSE_ERROR,
                         "model contains no classifier entries");
        C50Exit(1);
    }

    MaxTree = TRIALS - 1;
    if ( RULES )
    {
        RuleSet = AllocZero(TRIALS + 1, CRuleSet);
        ForEach(Trial, 0, TRIALS - 1)
        {
            InRulesAt(context, &model_input, &RuleSet[Trial]);
        }
    }
    else
    {
        Pruned = AllocZero(TRIALS + 1, Tree);
        ForEach(Trial, 0, TRIALS - 1)
        {
            InTreeAt(context, &model_input, &Pruned[Trial]);
        }
    }
}

static void LoadModel(c50_context *context, void *user_data)
{
    c50_model_load_state *state = user_data;

    state->model = AllocZero(1, c50_model);
    state->model->kind = state->kind;
    state->model->names_size = state->names_size;
    state->model->names_data = CopyInput(state->names_data, state->names_size);
    state->model->model_size = state->model_size;
    state->model->model_data = CopyInput(state->model_data, state->model_size);
    state->model->costs_size = state->costs_size;
    state->model->costs_data = CopyInput(state->costs_data, state->costs_size);

    ParseModel(context, state->model);
}

static void CleanupModelLoad(c50_context *context, void *user_data)
{
    c50_model_load_state *state = user_data;

    Cleanup(context);
    Of = NULL;
    if ( c50_context_last_status(context) != C50_STATUS_OK )
    {
        c50_model_destroy(state->model);
        state->model = NULL;
    }
}

static c50_status InvalidArgument(c50_context *context, const char *message)
{
    return c50_set_context_error(context, C50_STATUS_INVALID_ARGUMENT,
                                 message);
}

c50_status c50_model_load(c50_context *context, c50_model_kind kind,
                          const char *names_data, size_t names_size,
                          const char *model_data, size_t model_size,
                          const char *costs_data, size_t costs_size,
                          c50_model **out_model)
{
    c50_model_load_state state;
    c50_status status;

    if ( out_model ) *out_model = NULL;
    if ( ! context ) return C50_STATUS_INVALID_ARGUMENT;
    if ( ! out_model ) return InvalidArgument(context, "out_model is NULL");
    if ( kind != C50_MODEL_TREE && kind != C50_MODEL_RULES )
    {
        return InvalidArgument(context, "invalid model kind");
    }
    if ( ! names_data || ! names_size || ! model_data || ! model_size )
    {
        return InvalidArgument(context, "names and model inputs are required");
    }
    if ( ! costs_data && costs_size )
    {
        return InvalidArgument(context, "costs_data is NULL");
    }
    if ( memchr(names_data, '\0', names_size) ||
         memchr(model_data, '\0', model_size) ||
         (costs_size && memchr(costs_data, '\0', costs_size)) )
    {
        return InvalidArgument(context, "text inputs contain a NUL byte");
    }

    memset(&state, 0, sizeof(state));
    state.kind = kind;
    state.names_data = names_data;
    state.names_size = names_size;
    state.model_data = model_data;
    state.model_size = model_size;
    state.costs_data = costs_data;
    state.costs_size = costs_size;

    status = c50_run_operation(context, LoadModel, CleanupModelLoad, &state);
    if ( status == C50_STATUS_OK ) *out_model = state.model;
    return status;
}

static void PredictModel(c50_context *context, void *user_data)
{
    c50_predict_state *state = user_data;
    c50_input cases_input;
    c50_predictions *predictions;
    CaseNo row;
    ClassNo class_number, predicted;

    ParseModel(context, state->model);

    SomeMiss = AllocZero(MaxAtt + 1, Boolean);
    SomeNA = AllocZero(MaxAtt + 1, Boolean);
    if ( RULES ) context->most_specific_rules = Alloc(MaxClass + 1, CRule);
    context->default_class =
        ( RULES ? RuleSet[0]->SDefault : Pruned[0]->Leaf );
    context->class_sum = AllocZero(MaxClass + 1, float);
    context->votes = AllocZero(MaxClass + 1, float);
    context->trial_predictions = AllocZero(TRIALS, ClassNo);

    c50_input_init_memory(&cases_input, state->cases_data, state->cases_size);
    GetDataInput(context, &cases_input, false, true);

    predictions = AllocZero(1, c50_predictions);
    state->predictions = predictions;
    predictions->class_count = MaxClass;
    predictions->class_names = AllocZero(MaxClass, char *);
    ForEach(class_number, 1, MaxClass)
    {
        predictions->class_names[class_number - 1] =
            CopyInput(ClassName[class_number], strlen(ClassName[class_number]) + 1);
    }

    predictions->row_count = MaxCase + 1;
    if ( predictions->row_count )
    {
        predictions->class_indices =
            AllocZero(predictions->row_count, size_t);
        predictions->confidences =
            AllocZero(predictions->row_count, double);
        predictions->scores =
            AllocZero(predictions->row_count * predictions->class_count, double);
    }

    ForEach(row, 0, MaxCase)
    {
        predicted = Classify(context, Case[row]);
        predictions->class_indices[row] = predicted - 1;
        predictions->confidences[row] = context->confidence;
        ForEach(class_number, 1, MaxClass)
        {
            predictions->scores[
                row * predictions->class_count + class_number - 1] =
                context->class_sum[class_number];
        }
    }
}

static void CleanupPrediction(c50_context *context, void *user_data)
{
    c50_predict_state *state = user_data;

    Cleanup(context);
    c50_clear_prediction_state(context);
    Of = NULL;
    if ( c50_context_last_status(context) != C50_STATUS_OK )
    {
        c50_predictions_destroy(state->predictions);
        state->predictions = NULL;
    }
}

c50_status c50_model_predict(c50_context *context, const c50_model *model,
                             const char *cases_data, size_t cases_size,
                             c50_predictions **out_predictions)
{
    c50_predict_state state;
    c50_status status;

    if ( out_predictions ) *out_predictions = NULL;
    if ( ! context ) return C50_STATUS_INVALID_ARGUMENT;
    if ( ! out_predictions )
    {
        return InvalidArgument(context, "out_predictions is NULL");
    }
    if ( ! model ) return InvalidArgument(context, "model is NULL");
    if ( ! cases_data && cases_size )
    {
        return InvalidArgument(context, "cases_data is NULL");
    }
    if ( cases_size && memchr(cases_data, '\0', cases_size) )
    {
        return InvalidArgument(context, "cases input contains a NUL byte");
    }

    memset(&state, 0, sizeof(state));
    state.model = model;
    state.cases_data = cases_data;
    state.cases_size = cases_size;

    status = c50_run_operation(context, PredictModel, CleanupPrediction, &state);
    if ( status == C50_STATUS_OK ) *out_predictions = state.predictions;
    return status;
}
