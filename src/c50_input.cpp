/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <string.h>

#include "c50_input.h"

void c50_input_init_file(c50_input *input, FILE *file)
{
    input->buffer.resize(c50_input_block_size);
    input->file = file;
    input->data = NULL;
    input->size = 0;
    input->next = input->buffer.data();
    input->end = input->buffer.data();
}

void c50_input_init_memory(c50_input *input, const void *data, size_t size)
{
    input->file = NULL;
    input->data = static_cast<const unsigned char *>(data);
    input->size = size;
    input->next = input->data;
    input->end = input->data + size;
}

bool c50_input_fill(c50_input *input)
{
    if ( ! input->file ) return false;

    /* A short count means the end of the file or a read error. Both end the
       input, as EOF from fgetc() did. */
    size_t count = fread(input->buffer.data(), 1, input->buffer.size(), input->file);
    input->next = input->buffer.data();
    input->end = input->buffer.data() + count;
    return count > 0;
}

bool c50_input_read_line(c50_input *input, std::string &line)
{
    bool read = false;

    while ( input->next != input->end || c50_input_fill(input) )
    {
        const void *newline =
            memchr(input->next, '\n', static_cast<size_t>(input->end - input->next));
        const unsigned char *stop =
            newline ? static_cast<const unsigned char *>(newline) + 1 : input->end;

        line.append(reinterpret_cast<const char *>(input->next),
                    static_cast<size_t>(stop - input->next));
        input->next = stop;
        read = true;
        if ( newline ) break;
    }

    return read;
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
    if ( input->file )
    {
        rewind(input->file);
        input->next = input->buffer.data();
        input->end = input->buffer.data();
    }
    else
    {
        input->next = input->data;
        input->end = input->data + input->size;
    }
}
