/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <stddef.h>
#include <string.h>

#include <c50/c50.h>

static int CheckPredictions(c50_context *context, const c50_model *model)
{
    static const char cases[] =
        "0.5, alpha, ?\n"
        "3.5, beta, ?\n";
    c50_predictions *predictions = NULL;

    if ( c50_model_predict(context, model, cases, sizeof(cases) - 1,
                           &predictions) != C50_STATUS_OK )
    {
        return 0;
    }
    if ( c50_predictions_row_count(predictions) != 2 ||
         c50_predictions_class_index(predictions, 0) != 0 ||
         c50_predictions_class_index(predictions, 1) != 1 )
    {
        c50_predictions_destroy(predictions);
        return 0;
    }
    c50_predictions_destroy(predictions);
    return 1;
}

int main(void)
{
    static const char names[] =
        "low, high.\n\n"
        "signal: continuous.\n"
        "group: alpha, beta.\n";
    static const char data[] =
        "0, alpha, low\n"
        "0.5, beta, low\n"
        "1, alpha, low\n"
        "1.5, beta, low\n"
        "2.5, alpha, high\n"
        "3, beta, high\n"
        "3.5, alpha, high\n"
        "4, beta, high\n";
    static const char costs[] = "low, high: 2\n";
    c50_context *context = NULL;
    c50_model *model = NULL, *loaded = NULL;
    c50_options options;
    const char *serialized;
    size_t names_size, model_size, costs_size;

    if ( c50_context_create(&context) != C50_STATUS_OK ) return 1;

    c50_options_init(&options);
    if ( options.struct_size != sizeof(options) || options.trials != 1 ||
         options.subset_splits || options.winnow || ! options.global_pruning ||
         options.probabilistic_thresholds || options.ignore_costs ||
         options.minimum_cases != 2 || options.confidence_factor != 0.25 ||
         options.sample_fraction != 0 || options.random_seed != 0 ) return 1;

    if ( c50_model_train(context, C50_MODEL_TREE, NULL,
                         names, sizeof(names) - 1,
                         data, sizeof(data) - 1,
                         costs, sizeof(costs) - 1,
                         &model) != C50_STATUS_OK ) return 1;
    if ( ! model || c50_model_get_kind(model) != C50_MODEL_TREE ) return 1;
    if ( c50_model_names_data(model, &names_size) == NULL ||
         names_size != sizeof(names) - 1 ) return 1;
    serialized = c50_model_serialized_data(model, &model_size);
    if ( ! serialized || ! model_size ||
         ! strstr(serialized, "entries=\"1\"") ) return 1;
    if ( c50_model_costs_data(model, &costs_size) == NULL ||
         costs_size != sizeof(costs) - 1 ) return 1;
    if ( ! CheckPredictions(context, model) ) return 1;

    if ( c50_model_load(context, C50_MODEL_TREE,
                        c50_model_names_data(model, &names_size), names_size,
                        serialized, model_size,
                        c50_model_costs_data(model, &costs_size), costs_size,
                        &loaded) != C50_STATUS_OK ) return 1;
    if ( ! CheckPredictions(context, loaded) ) return 1;
    c50_model_destroy(loaded);
    loaded = NULL;
    c50_model_destroy(model);
    model = NULL;

    c50_options_init(&options);
    options.subset_splits = 1;
    options.winnow = 1;
    if ( c50_model_train(context, C50_MODEL_RULES, &options,
                         names, sizeof(names) - 1,
                         data, sizeof(data) - 1,
                         NULL, 0, &model) != C50_STATUS_OK ) return 1;
    serialized = c50_model_serialized_data(model, &model_size);
    if ( ! serialized || ! strstr(serialized, "rules=\"") ) return 1;
    if ( c50_model_costs_data(model, &costs_size) != NULL || costs_size ) return 1;
    if ( ! CheckPredictions(context, model) ) return 1;
    c50_model_destroy(model);
    model = NULL;

    c50_options_init(&options);
    options.trials = 0;
    if ( c50_model_train(context, C50_MODEL_TREE, &options,
                         names, sizeof(names) - 1,
                         data, sizeof(data) - 1,
                         NULL, 0, &model) != C50_STATUS_INVALID_ARGUMENT )
    {
        return 1;
    }
    if ( model ) return 1;

    c50_options_init(&options);
    options.struct_size--;
    if ( c50_model_train(context, C50_MODEL_TREE, &options,
                         names, sizeof(names) - 1,
                         data, sizeof(data) - 1,
                         NULL, 0, &model) != C50_STATUS_INVALID_ARGUMENT )
    {
        return 1;
    }
    if ( c50_model_names_data(NULL, &names_size) != NULL || names_size ) return 1;
    if ( c50_model_serialized_data(NULL, &model_size) != NULL || model_size ) return 1;
    if ( c50_model_costs_data(NULL, &costs_size) != NULL || costs_size ) return 1;

    c50_model_destroy(NULL);
    c50_context_destroy(context);
    return 0;
}
