/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "c50_input.h"

void c50_input_init_file(c50_input *input, FILE *file)
{
    input->kind = C50_INPUT_FILE;
    input->file = file;
    input->data = NULL;
    input->size = 0;
    input->position = 0;
}

void c50_input_init_memory(c50_input *input, const void *data, size_t size)
{
    input->kind = C50_INPUT_MEMORY;
    input->file = NULL;
    input->data = data;
    input->size = size;
    input->position = 0;
}

int c50_input_getc(c50_input *input)
{
    if ( input->kind == C50_INPUT_FILE ) return fgetc(input->file);
    if ( input->position == input->size ) return EOF;

    return input->data[input->position++];
}

char *c50_input_gets(char *buffer, size_t capacity, c50_input *input)
{
    int c = EOF;
    size_t length = 0;

    if ( ! capacity ) return NULL;

    while ( length < capacity - 1 &&
            (c = c50_input_getc(input)) != EOF )
    {
        buffer[length++] = c;
        if ( c == '\n' ) break;
    }

    if ( ! length ) return NULL;

    buffer[length] = '\0';
    return buffer;
}

void c50_input_rewind(c50_input *input)
{
    if ( input->kind == C50_INPUT_FILE )
    {
        rewind(input->file);
    }
    else
    {
        input->position = 0;
    }
}
