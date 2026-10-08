/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>

#include "defns.i"
#include "extern.i"
#include "c50_input.h"
#include "c50_api_internal.h"

static std::string read_bytes(c50_input *input)
{
    std::string bytes;
    int c;

    while ( (c = c50_input_getc(input)) != EOF ) bytes += static_cast<char>(c);
    return bytes;
}

static std::vector<std::string> read_lines(c50_input *input)
{
    std::vector<std::string> lines;
    std::string line;

    while ( c50_input_read_line(input, line) )
    {
        lines.push_back(line);
        line.clear();
    }
    return lines;
}

/* A file read in blocks must return the same bytes and lines as the same
   text in memory, including lines that cross a block boundary. */
static int check_blocks(void)
{
    static const char path[] = "c50-input-blocks.data";
    c50_input file_input, memory_input;
    std::string text;
    std::vector<std::string> expected;
    FILE *file;
    int status = 1;

    for ( size_t i = 0; text.size() < 2 * c50_input_block_size + 1000; i++ )
    {
        std::string line(i % 97, static_cast<char>('a' + i % 26));
        line += '\n';
        expected.push_back(line);
        text += line;
    }
    expected.push_back(std::string("nul\0byte\xff\n", 10));
    expected.push_back("final line without a newline");
    text += expected[expected.size() - 2] + expected.back();

    if ( ! (file = fopen(path, "w+b")) ) return 1;
    if ( fwrite(text.data(), 1, text.size(), file) == text.size() )
    {
        rewind(file);
        c50_input_init_file(&file_input, file);
        c50_input_init_memory(&memory_input, text.data(), text.size());

        bool ok = read_lines(&file_input) == expected &&
                  read_lines(&memory_input) == expected &&
                  c50_input_getc(&file_input) == EOF &&
                  c50_input_getc(&file_input) == EOF;

        /* Rewind mid-read, then mix character and line reads. */
        c50_input_rewind(&file_input);
        std::string line;
        ok = ok && c50_input_getc(&file_input) == '\n' &&
             c50_input_read_line(&file_input, line) && line == expected[1];
        c50_input_rewind(&file_input);
        ok = ok && read_bytes(&file_input) == text;
        c50_input_rewind(&memory_input);
        ok = ok && read_bytes(&memory_input) == text;

        c50_input_init_memory(&memory_input, NULL, 0);
        line.clear();
        ok = ok && c50_input_getc(&memory_input) == EOF &&
             ! c50_input_read_line(&memory_input, line) && line.empty();
        if ( ok ) status = 0;
    }
    fclose(file);
    remove(path);
    return status;
}

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
    auto owner = c50_make_context();
    c50_context *context = owner.get();

    (void) argc;
    (void) argv;

    if ( check_blocks() ) return 1;

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
    context->trees.pruned =
        static_cast<Tree *>(Pcalloc(context, 2, sizeof(Tree)));
    context->trees.pruned[0] = InTree(context, &input);
    if ( ! context->trees.pruned[0] || context->trees.pruned[0]->NodeType != 0 ||
         context->trees.pruned[0]->Leaf != 1 )
    {
        return 1;
    }
    Cleanup(context);


    return 0;
}
