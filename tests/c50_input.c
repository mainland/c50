/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <string.h>

#include "defns.i"
#include "extern.i"
#include "c50_input.h"
#include "c50_api_internal.h"

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
    c50_context *context = NULL;

    (void) argc;
    (void) argv;

    if ( c50_context_create(&context) != C50_STATUS_OK ) return 1;

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

    context->io.output = stderr;
    c50_input_init_memory(&input, names, sizeof(names) - 1);
    GetNames(context, &input);
    if ( context->schema.max_class != 2 || context->schema.max_attribute != 2 ) return 1;
    if ( strcmp(context->schema.class_names[1], "low") || strcmp(context->schema.class_names[2], "high") )
    {
        return 1;
    }
    if ( strcmp(context->schema.attribute_names[1], "signal") || strcmp(context->schema.attribute_names[2], "group") )
    {
        return 1;
    }
    c50_input_init_memory(&input, tree, sizeof(tree) - 1);
    c50_input_init_memory(&costs_input, costs, sizeof(costs) - 1);
    ReadHeaderMemory(context, &input, &costs_input);
    if ( context->options.trials != 1 ) return 1;
    if ( ! context->costs.matrix || context->costs.matrix[1][2] != 5 ) return 1;

    context->options.rules = false;
    context->trees.max_tree = 0;
    context->trees.pruned = Pcalloc(context, 2, sizeof(Tree));
    context->trees.pruned[0] = InTree(context, &input);
    if ( ! context->trees.pruned[0] || context->trees.pruned[0]->NodeType != 0 ||
         context->trees.pruned[0]->Leaf != 1 )
    {
        return 1;
    }
    Cleanup(context);
    c50_context_destroy(context);

    return 0;
}
