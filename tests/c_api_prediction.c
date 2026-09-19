/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <string.h>

#include <c50/c50.h>

int main(void)
{
    static const char names[] =
        "low, high.\n\n"
        "signal: continuous.\n"
        "group: alpha, beta.\n";
    static const char tree[] =
        "id=\"See5/C5.0 2.07 GPL Edition 2026-09-19\"\n"
        "entries=\"1\"\n"
        "type=\"2\" class=\"low\" att=\"signal\" forks=\"3\" cut=\"1\" "
        "freq=\"1,1\"\n"
        "type=\"0\" class=\"low\" freq=\"1,1\"\n"
        "type=\"0\" class=\"low\" freq=\"1,0\"\n"
        "type=\"0\" class=\"high\" freq=\"0,1\"\n";
    static const char cases[] =
        "0.5, alpha, ?\n"
        "2, beta, ?\n"
        "?, alpha, ?\n"
        "N/A, beta, ?\n";
    static const char malformed[] = "0.5";
    static const char unknown_value[] = "0.5, gamma, ?\n";
    static const char cases_with_nul[] = "0.5, alpha, ?\0\n";
    c50_context *context = NULL;
    c50_model *model = NULL;
    c50_predictions *predictions = NULL;
    c50_status status;

    if ( c50_context_create(&context) != C50_STATUS_OK ) return 1;

    predictions = (c50_predictions *) context;
    status = c50_model_predict(NULL, NULL, cases, sizeof(cases) - 1,
                               &predictions);
    if ( status != C50_STATUS_INVALID_ARGUMENT || predictions ) return 1;
    status = c50_model_predict(context, NULL, cases, sizeof(cases) - 1,
                               &predictions);
    if ( status != C50_STATUS_INVALID_ARGUMENT || predictions ) return 1;
    status = c50_model_load(context, C50_MODEL_TREE,
                            names, sizeof(names) - 1,
                            tree, sizeof(tree) - 1,
                            NULL, 0, &model);
    if ( status != C50_STATUS_OK || ! model ) return 1;

    status = c50_model_predict(context, model, cases, sizeof(cases) - 1,
                               &predictions);
    if ( status != C50_STATUS_OK || ! predictions ) return 1;
    if ( c50_predictions_row_count(predictions) != 4 ) return 1;
    if ( c50_predictions_class_count(predictions) != 2 ) return 1;
    if ( strcmp(c50_predictions_class_name(predictions, 0), "low") ) return 1;
    if ( strcmp(c50_predictions_class_name(predictions, 1), "high") ) return 1;
    if ( c50_predictions_class_name(predictions, 2) ) return 1;
    if ( c50_predictions_class_index(predictions, 0) != 0 ) return 1;
    if ( c50_predictions_class_index(predictions, 1) != 1 ) return 1;
    if ( c50_predictions_class_index(predictions, 2) != 0 ) return 1;
    if ( c50_predictions_class_index(predictions, 3) != 0 ) return 1;
    if ( c50_predictions_confidence(predictions, 0) != 1 ) return 1;
    if ( c50_predictions_confidence(predictions, 1) != 1 ) return 1;
    if ( c50_predictions_confidence(predictions, 2) != 1 ) return 1;
    if ( c50_predictions_confidence(predictions, 3) != 0.5 ) return 1;
    if ( c50_predictions_score(predictions, 0, 0) != 1 ) return 1;
    if ( c50_predictions_score(predictions, 0, 1) != 0 ) return 1;
    if ( c50_predictions_score(predictions, 1, 0) != 0 ) return 1;
    if ( c50_predictions_score(predictions, 1, 1) != 1 ) return 1;
    if ( c50_predictions_score(predictions, 2, 0) != 1 ) return 1;
    if ( c50_predictions_score(predictions, 2, 1) != 1 ) return 1;
    if ( c50_predictions_class_index(predictions, 4) != (size_t) -1 ) return 1;
    if ( c50_predictions_confidence(predictions, 4) != 0 ) return 1;
    if ( c50_predictions_score(predictions, 4, 0) != 0 ) return 1;
    c50_predictions_destroy(predictions);
    predictions = NULL;

    status = c50_model_predict(context, model, NULL, 0, &predictions);
    if ( status != C50_STATUS_OK || ! predictions ) return 1;
    if ( c50_predictions_row_count(predictions) != 0 ) return 1;
    if ( c50_predictions_class_count(predictions) != 2 ) return 1;
    c50_predictions_destroy(predictions);
    predictions = NULL;

    status = c50_model_predict(context, model,
                               malformed, sizeof(malformed) - 1,
                               &predictions);
    if ( status != C50_STATUS_PARSE_ERROR || predictions ) return 1;

    status = c50_model_predict(context, model,
                               unknown_value, sizeof(unknown_value) - 1,
                               &predictions);
    if ( status != C50_STATUS_PARSE_ERROR || predictions ) return 1;

    status = c50_model_predict(context, model, NULL, 1, &predictions);
    if ( status != C50_STATUS_INVALID_ARGUMENT || predictions ) return 1;

    status = c50_model_predict(context, model,
                               cases_with_nul, sizeof(cases_with_nul) - 1,
                               &predictions);
    if ( status != C50_STATUS_INVALID_ARGUMENT || predictions ) return 1;

    status = c50_model_predict(context, model, cases, sizeof(cases) - 1,
                               NULL);
    if ( status != C50_STATUS_INVALID_ARGUMENT ) return 1;

    status = c50_model_predict(context, model, cases, sizeof(cases) - 1,
                               &predictions);
    if ( status != C50_STATUS_OK || ! predictions ) return 1;

    c50_model_destroy(model);
    c50_context_destroy(context);
    if ( strcmp(c50_predictions_class_name(predictions, 0), "low") ) return 1;
    if ( c50_predictions_row_count(predictions) != 4 ) return 1;
    c50_predictions_destroy(predictions);
    c50_predictions_destroy(NULL);
    if ( c50_predictions_row_count(NULL) != 0 ) return 1;
    if ( c50_predictions_class_count(NULL) != 0 ) return 1;
    if ( c50_predictions_class_name(NULL, 0) ) return 1;
    if ( c50_predictions_class_index(NULL, 0) != (size_t) -1 ) return 1;
    return 0;
}
