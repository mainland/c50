/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <string.h>

#include "defns.i"
#include "extern.i"
#include "c50_input.h"

int main(int argc, char *argv[])
{
    static const unsigned char text[] = "first\nsecond";
    static const unsigned char byte[] = {0xff};
    static const unsigned char names[] =
        "low, high.\n\n"
        "signal: continuous.\n"
        "group: alpha, beta.\n";
    static const unsigned char tree[] =
        "id=\"See5/C5.0 2.07 GPL Edition 2026-09-19\"\n"
        "costs=\"1\"\n"
        "entries=\"1\"\n"
        "type=\"0\" class=\"low\" freq=\"1,0\"\n";
    static const unsigned char costs[] = "low, high: 5\n";
    char line[16];
    c50_input costs_input, input;

    (void) argc;
    (void) argv;

    c50_input_init_memory(&input, text, sizeof(text) - 1);
    if ( c50_input_getc(&input) != 'f' ) return 1;

    c50_input_rewind(&input);
    if ( ! c50_input_gets(line, sizeof(line), &input) ) return 1;
    if ( strcmp(line, "first\n") ) return 1;
    if ( ! c50_input_gets(line, sizeof(line), &input) ) return 1;
    if ( strcmp(line, "second") ) return 1;
    if ( c50_input_gets(line, sizeof(line), &input) ) return 1;

    c50_input_init_memory(&input, byte, sizeof(byte));
    if ( c50_input_getc(&input) != 0xff ) return 1;
    if ( c50_input_getc(&input) != EOF ) return 1;
    if ( c50_input_gets(line, 0, &input) ) return 1;

    Of = stderr;
    c50_input_init_memory(&input, names, sizeof(names) - 1);
    GetNames(&input);
    if ( MaxClass != 2 || MaxAtt != 2 ) return 1;
    if ( strcmp(ClassName[1], "low") || strcmp(ClassName[2], "high") )
    {
        return 1;
    }
    if ( strcmp(AttName[1], "signal") || strcmp(AttName[2], "group") )
    {
        return 1;
    }
    c50_input_init_memory(&input, tree, sizeof(tree) - 1);
    c50_input_init_memory(&costs_input, costs, sizeof(costs) - 1);
    ReadHeaderMemory(&input, &costs_input);
    if ( TRIALS != 1 ) return 1;
    if ( ! MCost || MCost[1][2] != 5 ) return 1;

    RULES = false;
    MaxTree = 0;
    Pruned = AllocZero(2, Tree);
    Pruned[0] = InTree(&input);
    if ( ! Pruned[0] || Pruned[0]->NodeType != 0 ||
         Pruned[0]->Leaf != 1 )
    {
        return 1;
    }
    Cleanup();

    return 0;
}
