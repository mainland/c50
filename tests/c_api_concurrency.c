/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <pthread.h>
#include <stddef.h>
#include <string.h>

#include <c50/c50.h>

typedef struct start_gate
{
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    int ready;
    int open;
} start_gate;

typedef struct worker_state
{
    start_gate *gate;
    c50_context *context;
    const c50_model *model;
    const char *cases;
    size_t cases_size;
    size_t class_count;
    size_t class_index;
    const char *class_name;
    int failed;
} worker_state;

static void *RunPredictions(void *argument)
{
    worker_state *state = argument;
    int iteration;

    pthread_mutex_lock(&state->gate->mutex);
    state->gate->ready++;
    pthread_cond_broadcast(&state->gate->condition);
    while ( ! state->gate->open )
    {
        pthread_cond_wait(&state->gate->condition, &state->gate->mutex);
    }
    pthread_mutex_unlock(&state->gate->mutex);

    for ( iteration = 0; iteration < 200 && ! state->failed; iteration++ )
    {
        c50_predictions *predictions = NULL;
        c50_status status = c50_model_predict(state->context, state->model,
                                              state->cases,
                                              state->cases_size,
                                              &predictions);

        state->failed =
            status != C50_STATUS_OK || ! predictions ||
            c50_predictions_row_count(predictions) != 1 ||
            c50_predictions_class_count(predictions) != state->class_count ||
            c50_predictions_class_index(predictions, 0) != state->class_index ||
            strcmp(c50_predictions_class_name(predictions,
                                              state->class_index),
                   state->class_name);
        c50_predictions_destroy(predictions);
    }

    return NULL;
}

static int RunWorkers(start_gate *gate, worker_state workers[2])
{
    pthread_t threads[2];

    pthread_mutex_lock(&gate->mutex);
    gate->ready = 0;
    gate->open = 0;
    pthread_mutex_unlock(&gate->mutex);

    if ( pthread_create(&threads[0], NULL, RunPredictions, &workers[0]) )
    {
        return 1;
    }
    if ( pthread_create(&threads[1], NULL, RunPredictions, &workers[1]) )
    {
        pthread_mutex_lock(&gate->mutex);
        gate->open = 1;
        pthread_cond_broadcast(&gate->condition);
        pthread_mutex_unlock(&gate->mutex);
        pthread_join(threads[0], NULL);
        return 1;
    }

    pthread_mutex_lock(&gate->mutex);
    while ( gate->ready != 2 )
    {
        pthread_cond_wait(&gate->condition, &gate->mutex);
    }
    gate->open = 1;
    pthread_cond_broadcast(&gate->condition);
    pthread_mutex_unlock(&gate->mutex);

    pthread_join(threads[0], NULL);
    pthread_join(threads[1], NULL);
    return workers[0].failed || workers[1].failed;
}

int main(void)
{
    static const char names_a[] =
        "low, high.\n\n"
        "signal: continuous.\n";
    static const char tree_a[] =
        "id=\"See5/C5.0 2.07 GPL Edition 2026-09-19\"\n"
        "entries=\"1\"\n"
        "type=\"0\" class=\"low\" freq=\"2,0\"\n";
    static const char cases_a[] = "1, ?\n";
    static const char names_b[] =
        "first, second, third.\n\n"
        "value: continuous.\n";
    static const char tree_b[] =
        "id=\"See5/C5.0 2.07 GPL Edition 2026-09-19\"\n"
        "entries=\"1\"\n"
        "type=\"0\" class=\"third\" freq=\"0,0,4\"\n";
    static const char cases_b[] = "2, ?\n";
    start_gate gate = {
        PTHREAD_MUTEX_INITIALIZER,
        PTHREAD_COND_INITIALIZER,
        0,
        0
    };
    c50_context *contexts[2] = {NULL, NULL};
    c50_model *models[2] = {NULL, NULL};
    worker_state workers[2];
    int result = 1;

    if ( c50_context_create(&contexts[0]) != C50_STATUS_OK ||
         c50_context_create(&contexts[1]) != C50_STATUS_OK ) goto cleanup;
    if ( c50_model_load(contexts[0], C50_MODEL_TREE,
                        names_a, sizeof(names_a) - 1,
                        tree_a, sizeof(tree_a) - 1,
                        NULL, 0, &models[0]) != C50_STATUS_OK ) goto cleanup;
    if ( c50_model_load(contexts[1], C50_MODEL_TREE,
                        names_b, sizeof(names_b) - 1,
                        tree_b, sizeof(tree_b) - 1,
                        NULL, 0, &models[1]) != C50_STATUS_OK ) goto cleanup;

    workers[0] = (worker_state) {
        &gate, contexts[0], models[0], cases_a, sizeof(cases_a) - 1,
        2, 0, "low", 0
    };
    workers[1] = (worker_state) {
        &gate, contexts[1], models[1], cases_b, sizeof(cases_b) - 1,
        3, 2, "third", 0
    };

    if ( RunWorkers(&gate, workers) ) goto cleanup;

    workers[0].failed = 0;
    workers[1] = (worker_state) {
        &gate, contexts[1], models[0], cases_a, sizeof(cases_a) - 1,
        2, 0, "low", 0
    };
    if ( RunWorkers(&gate, workers) ) goto cleanup;
    result = 0;

cleanup:
    c50_model_destroy(models[1]);
    c50_model_destroy(models[0]);
    c50_context_destroy(contexts[1]);
    c50_context_destroy(contexts[0]);
    pthread_cond_destroy(&gate.condition);
    pthread_mutex_destroy(&gate.mutex);
    return result;
}
