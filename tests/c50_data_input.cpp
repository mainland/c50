/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "defns.i"
#include "extern.i"

int main(int argc, char *argv[])
{
    static const char names[] =
        "low, high.\n\n"
        "signal: continuous.\n";
    static const char data[] =
        "0.25, low\n"
        "1.5, high\n";
    static const char trailing_comment[] =
        "0.25, low\n"
        "| trailing comment";
    c50_input names_input, data_input, comment_input;

    (void) argc;
    (void) argv;

    c50_input_init_memory(&names_input, names, sizeof(names) - 1);
    GetNames(&names_input);

    c50_input_init_memory(&data_input, data, sizeof(data) - 1);
    if ( CountDataInput(&data_input) != 2 ) return 1;
    if ( data_input.position != 0 ) return 1;

    c50_input_init_memory(&comment_input, trailing_comment,
                          sizeof(trailing_comment) - 1);
    if ( CountDataInput(&comment_input) != 1 ) return 1;
    if ( comment_input.position != 0 ) return 1;

    GetDataInput(&data_input, false, false);
    if ( MaxCase != 1 ) return 1;
    if ( CVal(Case[0], 1) != 0.25 || Class(Case[0]) != 1 ) return 1;
    if ( CVal(Case[1], 1) != 1.5 || Class(Case[1]) != 2 ) return 1;

    Cleanup();
    return 0;
}
