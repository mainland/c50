/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <string.h>

#include <c50/c50.h>

int main(void)
{
    static const char names[] =
        "low, high.\n\n"
        "signal: continuous.\n";
    static const char tree[] =
        "id=\"See5/C5.0 2.07 GPL Edition 2026-09-19\"\n"
        "costs=\"1\"\n"
        "entries=\"1\"\n"
        "type=\"0\" class=\"low\" freq=\"1,0\"\n";
    static const char rules[] =
        "id=\"See5/C5.0 2.07 GPL Edition 2026-09-19\"\n"
        "entries=\"1\"\n"
        "rules=\"1\" default=\"low\"\n"
        "conds=\"0\" cover=\"1\" ok=\"1\" lift=\"1\" class=\"low\"\n";
    static const char costs[] = "low, high: 5\n";
    static const char malformed[] = "id=\"truncated\"\n";
    static const char malformed_tree[] =
        "id=\"See5/C5.0 2.07 GPL Edition 2026-09-19\"\n"
        "entries=\"1\"\n"
        "type=\"2\" class=\"low\" att=\"signal\" forks=\"3\" cut=\"1\"\n"
        "type=\"0\" class=\"low\"\n"
        "type=\"0\" class=\"low\"";
    static const char malformed_rules[] =
        "id=\"See5/C5.0 2.07 GPL Edition 2026-09-19\"\n"
        "entries=\"1\"\n"
        "rules=\"1\" default=\"low\"\n"
        "conds=\"1\" cover=\"1\" ok=\"1\" lift=\"1\" class=\"low\"\n"
        "type=\"2\" att=\"signal\"";
    static const char no_entries[] =
        "id=\"See5/C5.0 2.07 GPL Edition 2026-09-19\"\n"
        "entries=\"0\"\n";
    static const char names_with_nul[] = "low,\0high.\n";
    static const char model_with_nul[] = "id=\"bad\"\0\n";
    static const char costs_with_nul[] = "low, high:\0 5\n";
    c50_context *context = NULL;
    c50_model *model = NULL;
    c50_status status;

    if ( c50_context_create(&context) != C50_STATUS_OK ) return 1;

    model = (c50_model *) context;
    status = c50_model_load(NULL, C50_MODEL_TREE,
                            names, sizeof(names) - 1,
                            tree, sizeof(tree) - 1,
                            costs, sizeof(costs) - 1, &model);
    if ( status != C50_STATUS_INVALID_ARGUMENT || model ) return 1;

    status = c50_model_load(context, C50_MODEL_TREE,
                            names, sizeof(names) - 1,
                            tree, sizeof(tree) - 1,
                            costs, sizeof(costs) - 1, NULL);
    if ( status != C50_STATUS_INVALID_ARGUMENT ) return 1;

    status = c50_model_load(context, (c50_model_kind) 100,
                            names, sizeof(names) - 1,
                            tree, sizeof(tree) - 1,
                            costs, sizeof(costs) - 1, &model);
    if ( status != C50_STATUS_INVALID_ARGUMENT || model ) return 1;

    status = c50_model_load(context, C50_MODEL_TREE,
                            NULL, sizeof(names) - 1,
                            tree, sizeof(tree) - 1,
                            costs, sizeof(costs) - 1, &model);
    if ( status != C50_STATUS_INVALID_ARGUMENT || model ) return 1;

    status = c50_model_load(context, C50_MODEL_TREE,
                            names, 0, tree, sizeof(tree) - 1,
                            costs, sizeof(costs) - 1, &model);
    if ( status != C50_STATUS_INVALID_ARGUMENT || model ) return 1;

    status = c50_model_load(context, C50_MODEL_TREE,
                            names, sizeof(names) - 1, NULL, sizeof(tree) - 1,
                            costs, sizeof(costs) - 1, &model);
    if ( status != C50_STATUS_INVALID_ARGUMENT || model ) return 1;

    status = c50_model_load(context, C50_MODEL_TREE,
                            names, sizeof(names) - 1, tree, 0,
                            costs, sizeof(costs) - 1, &model);
    if ( status != C50_STATUS_INVALID_ARGUMENT || model ) return 1;

    status = c50_model_load(context, C50_MODEL_TREE,
                            names, sizeof(names) - 1,
                            tree, sizeof(tree) - 1,
                            NULL, 1, &model);
    if ( status != C50_STATUS_INVALID_ARGUMENT || model ) return 1;

    status = c50_model_load(context, C50_MODEL_TREE,
                            names_with_nul, sizeof(names_with_nul) - 1,
                            tree, sizeof(tree) - 1,
                            costs, sizeof(costs) - 1, &model);
    if ( status != C50_STATUS_INVALID_ARGUMENT || model ) return 1;

    status = c50_model_load(context, C50_MODEL_TREE,
                            names, sizeof(names) - 1,
                            model_with_nul, sizeof(model_with_nul) - 1,
                            costs, sizeof(costs) - 1, &model);
    if ( status != C50_STATUS_INVALID_ARGUMENT || model ) return 1;

    status = c50_model_load(context, C50_MODEL_TREE,
                            names, sizeof(names) - 1,
                            tree, sizeof(tree) - 1,
                            costs_with_nul, sizeof(costs_with_nul) - 1, &model);
    if ( status != C50_STATUS_INVALID_ARGUMENT || model ) return 1;

    status = c50_model_load(context, C50_MODEL_TREE,
                            names, sizeof(names) - 1,
                            no_entries, sizeof(no_entries) - 1,
                            NULL, 0, &model);
    if ( status != C50_STATUS_PARSE_ERROR || model ) return 1;

    status = c50_model_load(context, C50_MODEL_TREE,
                            names, sizeof(names) - 1,
                            tree, sizeof(tree) - 1,
                            costs, sizeof(costs) - 1, &model);
    if ( status != C50_STATUS_OK || ! model ) return 1;
    if ( c50_model_get_kind(model) != C50_MODEL_TREE ) return 1;
    c50_model_destroy(model);
    model = NULL;

    status = c50_model_load(context, C50_MODEL_RULES,
                            names, sizeof(names) - 1,
                            rules, sizeof(rules) - 1,
                            NULL, 0, &model);
    if ( status != C50_STATUS_OK || ! model ) return 1;
    if ( c50_model_get_kind(model) != C50_MODEL_RULES ) return 1;
    c50_model_destroy(model);
    model = NULL;

    status = c50_model_load(context, C50_MODEL_TREE,
                            names, sizeof(names) - 1,
                            tree, sizeof(tree) - 1,
                            NULL, 0, &model);
    if ( status != C50_STATUS_IO_ERROR || model ) return 1;
    if ( ! strstr(c50_context_error_message(context), "costs") ) return 1;

    status = c50_model_load(context, C50_MODEL_TREE,
                            names, sizeof(names) - 1,
                            malformed, sizeof(malformed) - 1,
                            NULL, 0, &model);
    if ( status != C50_STATUS_PARSE_ERROR || model ) return 1;

    status = c50_model_load(context, C50_MODEL_TREE,
                            names, sizeof(names) - 1,
                            malformed_tree, sizeof(malformed_tree) - 1,
                            NULL, 0, &model);
    if ( status != C50_STATUS_PARSE_ERROR || model ) return 1;

    status = c50_model_load(context, C50_MODEL_RULES,
                            names, sizeof(names) - 1,
                            malformed_rules, sizeof(malformed_rules) - 1,
                            NULL, 0, &model);
    if ( status != C50_STATUS_PARSE_ERROR || model ) return 1;

    status = c50_model_load(context, C50_MODEL_RULES,
                            names, sizeof(names) - 1,
                            rules, sizeof(rules) - 1,
                            NULL, 0, &model);
    if ( status != C50_STATUS_OK || ! model ) return 1;

    c50_model_destroy(model);
    c50_context_destroy(context);
    return 0;
}
