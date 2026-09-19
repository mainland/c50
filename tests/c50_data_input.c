/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "defns.i"
#include "extern.i"
#include "c50_api_internal.h"

int main(int argc, char *argv[])
{
    static const char names[] =
        "low, high.\n\n"
        "identifier: label.\n"
        "signal: continuous.\n"
        "case weight: continuous.\n";
    static const char data[] =
        "first, 0.25, 2, low\n"
        "second, 1.5, 4, high\n";
    static const char trailing_comment[] =
        "0.25, low\n"
        "| trailing comment";
    c50_input names_input, data_input, comment_input;
    c50_context *context = NULL;

    (void) argc;
    (void) argv;

    if ( c50_context_create(&context) != C50_STATUS_OK ) return 1;

    c50_input_init_memory(&names_input, names, sizeof(names) - 1);
    GetNames(context, &names_input);
    if ( context->schema.class_attribute != 0 ||
         context->schema.case_weight_attribute != 3 ) return 1;

    c50_input_init_memory(&data_input, data, sizeof(data) - 1);
    if ( CountDataInput(&data_input) != 2 ) return 1;
    if ( data_input.position != 0 ) return 1;

    c50_input_init_memory(&comment_input, trailing_comment,
                          sizeof(trailing_comment) - 1);
    if ( CountDataInput(&comment_input) != 1 ) return 1;
    if ( comment_input.position != 0 ) return 1;

    GetDataInput(context, &data_input, false, false);
    if ( context->cases.max_case != 1 ) return 1;
    if ( CVal(context->cases.records[0], 2) != 0.25 || Class(context->cases.records[0]) != 1 ) return 1;
    if ( CVal(context->cases.records[1], 2) != 1.5 || Class(context->cases.records[1]) != 2 ) return 1;
    if ( strcmp(CaseLabel(context, 0), "first") ) return 1;
    if ( strcmp(CaseLabel(context, 1), "second") ) return 1;
    SetAvCWt(context);
    if ( context->average_case_weight != 3 ) return 1;

    Cleanup(context);
    c50_context_destroy(context);
    return 0;
}
