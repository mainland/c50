/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <stdio.h>
#include <string.h>

#include <c50/c50.h>

static int CheckPrediction(c50_context *context, const c50_model *model,
                           const char *cases, size_t cases_size,
                           size_t class_count, size_t class_index,
                           const char *class_name)
{
    c50_predictions *predictions = NULL;
    c50_status status;
    int ok;

    status = c50_model_predict(context, model, cases, cases_size,
                               &predictions);
    ok = status == C50_STATUS_OK && predictions &&
         c50_predictions_row_count(predictions) == 1 &&
         c50_predictions_class_count(predictions) == class_count &&
         c50_predictions_class_index(predictions, 0) == class_index &&
         ! strcmp(c50_predictions_class_name(predictions, class_index),
                  class_name);
    c50_predictions_destroy(predictions);
    return ok;
}

#define REQUIRE(Condition)                                                    \
    do                                                                        \
    {                                                                         \
        if ( ! (Condition) )                                                  \
        {                                                                     \
            fprintf(stderr, "context isolation check failed at line %d\n",  \
                    __LINE__);                                                \
            result = 1;                                                       \
            goto cleanup;                                                     \
        }                                                                     \
    } while ( 0 )

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
    static const char tree_cost[] =
        "id=\"See5/C5.0 2.07 GPL Edition 2026-09-19\"\n"
        "costs=\"1\"\n"
        "entries=\"1\"\n"
        "type=\"0\" class=\"low\" freq=\"6,4\"\n";
    static const char costs[] = "low, high: 5\n";
    static const char malformed_cases[] = "1";
    c50_context *context_a = NULL;
    c50_context *context_b = NULL;
    c50_context *context_cost = NULL;
    c50_context *temporary_context = NULL;
    c50_model *model_a = NULL;
    c50_model *model_b = NULL;
    c50_model *model_cost = NULL;
    c50_predictions *predictions = NULL;
    c50_status status;
    int iteration;
    int result = 0;

    REQUIRE(c50_context_create(&context_a) == C50_STATUS_OK);
    REQUIRE(c50_context_create(&context_b) == C50_STATUS_OK);
    REQUIRE(c50_context_create(&context_cost) == C50_STATUS_OK);

    status = c50_model_load(context_a, C50_MODEL_TREE,
                            names_a, sizeof(names_a) - 1,
                            tree_a, sizeof(tree_a) - 1,
                            NULL, 0, &model_a);
    REQUIRE(status == C50_STATUS_OK && model_a);
    status = c50_model_load(context_b, C50_MODEL_TREE,
                            names_b, sizeof(names_b) - 1,
                            tree_b, sizeof(tree_b) - 1,
                            NULL, 0, &model_b);
    REQUIRE(status == C50_STATUS_OK && model_b);
    status = c50_model_load(context_cost, C50_MODEL_TREE,
                            names_a, sizeof(names_a) - 1,
                            tree_cost, sizeof(tree_cost) - 1,
                            costs, sizeof(costs) - 1, &model_cost);
    REQUIRE(status == C50_STATUS_OK && model_cost);

    REQUIRE(CheckPrediction(context_a, model_a,
                            cases_a, sizeof(cases_a) - 1,
                            2, 0, "low"));
    REQUIRE(CheckPrediction(context_cost, model_cost,
                            cases_a, sizeof(cases_a) - 1,
                            2, 1, "high"));
    REQUIRE(CheckPrediction(context_a, model_a,
                            cases_a, sizeof(cases_a) - 1,
                            2, 0, "low"));
    REQUIRE(CheckPrediction(context_b, model_b,
                            cases_b, sizeof(cases_b) - 1,
                            3, 2, "third"));
    REQUIRE(CheckPrediction(context_a, model_a,
                            cases_a, sizeof(cases_a) - 1,
                            2, 0, "low"));

    status = c50_model_predict(context_a, model_a,
                               malformed_cases,
                               sizeof(malformed_cases) - 1,
                               &predictions);
    REQUIRE(status == C50_STATUS_PARSE_ERROR && ! predictions);
    REQUIRE(c50_context_error_message(context_a)[0] != '\0');
    REQUIRE(c50_context_last_status(context_b) == C50_STATUS_OK);
    REQUIRE(c50_context_error_message(context_b)[0] == '\0');

    REQUIRE(CheckPrediction(context_b, model_b,
                            cases_b, sizeof(cases_b) - 1,
                            3, 2, "third"));
    REQUIRE(c50_context_last_status(context_a) == C50_STATUS_PARSE_ERROR);
    REQUIRE(CheckPrediction(context_a, model_a,
                            cases_a, sizeof(cases_a) - 1,
                            2, 0, "low"));
    REQUIRE(c50_context_last_status(context_a) == C50_STATUS_OK);

    REQUIRE(CheckPrediction(context_b, model_a,
                            cases_a, sizeof(cases_a) - 1,
                            2, 0, "low"));
    c50_context_destroy(context_a);
    context_a = NULL;
    REQUIRE(CheckPrediction(context_b, model_a,
                            cases_a, sizeof(cases_a) - 1,
                            2, 0, "low"));

    for ( iteration = 0; iteration < 16; iteration++ )
    {
        REQUIRE(c50_context_create(&temporary_context) == C50_STATUS_OK);
        REQUIRE(CheckPrediction(temporary_context, model_b,
                                cases_b, sizeof(cases_b) - 1,
                                3, 2, "third"));
        c50_context_destroy(temporary_context);
        temporary_context = NULL;
    }

cleanup:
    c50_predictions_destroy(predictions);
    c50_model_destroy(model_cost);
    c50_model_destroy(model_b);
    c50_model_destroy(model_a);
    c50_context_destroy(temporary_context);
    c50_context_destroy(context_b);
    c50_context_destroy(context_cost);
    c50_context_destroy(context_a);
    return result;
}
