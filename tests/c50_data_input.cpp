/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <climits>

#include "defns.i"
#include "extern.i"
#include "c50_api_internal.h"

using namespace c50::core;

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
    auto owner = c50_make_context();
    c50_context *context = owner.get();

    (void) argc;
    (void) argv;


    c50_input_init_memory(&names_input, names, sizeof(names) - 1);
    GetNames(context, &names_input);
    if ( context->schema.class_attribute != 0 ||
         context->schema.case_weight_attribute != 3 ) return 1;

    c50_input_init_memory(&data_input, data, sizeof(data) - 1);
    if ( CountDataInput(context, &data_input) != 2 ) return 1;
    if ( data_input.next != data_input.data ) return 1;

    c50_input_init_memory(&comment_input, trailing_comment,
                          sizeof(trailing_comment) - 1);
    if ( CountDataInput(context, &comment_input) != 1 ) return 1;
    if ( comment_input.next != comment_input.data ) return 1;

    GetDataInput(context, &data_input, false, false);
    if ( context->cases.max_case != 1 ) return 1;
    if ( CVal(context->cases.records[0], 2) != 0.25 || Class(context->cases.records[0]) != 1 ) return 1;
    if ( CVal(context->cases.records[1], 2) != 1.5 || Class(context->cases.records[1]) != 2 ) return 1;
    if ( strcmp(CaseLabel(context, 0), "first") ) return 1;
    if ( strcmp(CaseLabel(context, 1), "second") ) return 1;
    SetAvCWt(context);
    if ( context->average_case_weight != 3 ) return 1;

    // The line counter holds more than INT_MAX lines, and a diagnostic prints
    // the line number that the counter holds.
    static const char two_lines[] = "first\nsecond\n";
    c50_input line_input;
    c50_input_init_memory(&line_input, two_lines, sizeof(two_lines) - 1);
    context->io.line_number = INT_MAX;
    if ( InChar(context, &line_input) != 'f' ) return 1;
    if ( context->io.line_number != 2147483648LL ) return 1;
    context->error.reset();
    Error(context, UNKNOWNATT, "missing", Nil);
    if ( context->error_message.find("line 2147483648 of ") != 0 ) return 1;

    Cleanup(context);

    return 0;
}
