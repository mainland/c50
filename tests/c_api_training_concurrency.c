/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <pthread.h>
#include <stddef.h>

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
    const char *names;
    size_t names_size;
    const char *training_data;
    size_t training_size;
    const char *cases;
    size_t cases_size;
    size_t expected_class;
    int failed;
} worker_state;

static void *TrainModels(void *argument)
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

    for ( iteration = 0; iteration < 20 && ! state->failed; iteration++ )
    {
        c50_model *model = NULL;
        c50_predictions *predictions = NULL;

        state->failed =
            c50_model_train(state->context, C50_MODEL_TREE, NULL,
                            state->names, state->names_size,
                            state->training_data, state->training_size,
                            NULL, 0, &model) != C50_STATUS_OK ||
            ! model ||
            c50_model_predict(state->context, model,
                              state->cases, state->cases_size,
                              &predictions) != C50_STATUS_OK ||
            ! predictions ||
            c50_predictions_row_count(predictions) != 1 ||
            c50_predictions_class_index(predictions, 0) !=
                state->expected_class;
        c50_predictions_destroy(predictions);
        c50_model_destroy(model);
    }

    return NULL;
}

int main(void)
{
    static const char names[] =
        "low, high.\n\n"
        "signal: continuous.\n";
    static const char ascending[] =
        "0, low\n1, low\n2, low\n3, high\n4, high\n5, high\n";
    static const char descending[] =
        "0, high\n1, high\n2, high\n3, low\n4, low\n5, low\n";
    static const char low_case[] = "0, ?\n";
    start_gate gate = {
        PTHREAD_MUTEX_INITIALIZER,
        PTHREAD_COND_INITIALIZER,
        0,
        0
    };
    c50_context *contexts[2] = {NULL, NULL};
    pthread_t threads[2];
    worker_state workers[2];
    int result = 1;

    if ( c50_context_create(&contexts[0]) != C50_STATUS_OK ||
         c50_context_create(&contexts[1]) != C50_STATUS_OK ) goto cleanup;

    workers[0] = (worker_state) {
        &gate, contexts[0], names, sizeof(names) - 1,
        ascending, sizeof(ascending) - 1,
        low_case, sizeof(low_case) - 1, 0, 0
    };
    workers[1] = (worker_state) {
        &gate, contexts[1], names, sizeof(names) - 1,
        descending, sizeof(descending) - 1,
        low_case, sizeof(low_case) - 1, 1, 0
    };

    if ( pthread_create(&threads[0], NULL, TrainModels, &workers[0]) )
    {
        goto cleanup;
    }
    if ( pthread_create(&threads[1], NULL, TrainModels, &workers[1]) )
    {
        pthread_mutex_lock(&gate.mutex);
        gate.open = 1;
        pthread_cond_broadcast(&gate.condition);
        pthread_mutex_unlock(&gate.mutex);
        pthread_join(threads[0], NULL);
        goto cleanup;
    }

    pthread_mutex_lock(&gate.mutex);
    while ( gate.ready != 2 )
    {
        pthread_cond_wait(&gate.condition, &gate.mutex);
    }
    gate.open = 1;
    pthread_cond_broadcast(&gate.condition);
    pthread_mutex_unlock(&gate.mutex);

    pthread_join(threads[0], NULL);
    pthread_join(threads[1], NULL);
    result = workers[0].failed || workers[1].failed;

cleanup:
    c50_context_destroy(contexts[1]);
    c50_context_destroy(contexts[0]);
    pthread_cond_destroy(&gate.condition);
    pthread_mutex_destroy(&gate.mutex);
    return result;
}
