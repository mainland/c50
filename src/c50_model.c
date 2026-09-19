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

typedef struct c50_model_load_state
{
    c50_context *context;
    c50_model_kind kind;
    const char *names_data;
    size_t names_size;
    const char *model_data;
    size_t model_size;
    const char *costs_data;
    size_t costs_size;
    c50_model *model;
} c50_model_load_state;

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

static void LoadModel(void *user_data)
{
    c50_model_load_state *state = user_data;
    c50_input names_input, model_input, costs_input, *costs = NULL;

    state->model = AllocZero(1, c50_model);
    state->model->kind = state->kind;
    state->model->names_size = state->names_size;
    state->model->names_data = CopyInput(state->names_data, state->names_size);
    state->model->model_size = state->model_size;
    state->model->model_data = CopyInput(state->model_data, state->model_size);
    state->model->costs_size = state->costs_size;
    state->model->costs_data = CopyInput(state->costs_data, state->costs_size);

    Of = NULL;
    FileStem = "memory";
    snprintf(Fn, sizeof(Fn), "%s", "memory.model");
    RULES = state->kind == C50_MODEL_RULES;
    TRIALS = 1;
    MaxTree = -1;
    SAMPLE = 0;

    c50_input_init_memory(&names_input, state->names_data, state->names_size);
    GetNames(&names_input);

    c50_input_init_memory(&model_input, state->model_data, state->model_size);
    if ( state->costs_size )
    {
        c50_input_init_memory(&costs_input, state->costs_data,
                              state->costs_size);
        costs = &costs_input;
    }
    ReadHeaderMemory(&model_input, costs);
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
            InRulesAt(&model_input, &RuleSet[Trial]);
        }
    }
    else
    {
        Pruned = AllocZero(TRIALS + 1, Tree);
        ForEach(Trial, 0, TRIALS - 1)
        {
            InTreeAt(&model_input, &Pruned[Trial]);
        }
    }
}

static void CleanupModelLoad(void *user_data)
{
    c50_model_load_state *state = user_data;

    Cleanup();
    Of = NULL;
    if ( c50_context_last_status(state->context) != C50_STATUS_OK )
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
    state.context = context;
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
