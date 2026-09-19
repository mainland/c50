/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#include "c50_output.h"

static int Reserve(c50_output *output, size_t additional)
{
    size_t required, capacity;
    unsigned char *data;

    if ( additional > (size_t) -1 - output->size - 1 ) return 0;
    required = output->size + additional + 1;
    if ( required <= output->capacity ) return 1;

    capacity = output->capacity ? output->capacity : 256;
    while ( capacity < required )
    {
        if ( capacity > (size_t) -1 / 2 )
        {
            capacity = required;
            break;
        }
        capacity *= 2;
    }

    data = realloc(output->data, capacity);
    if ( ! data ) return 0;
    output->data = data;
    output->capacity = capacity;
    return 1;
}

void c50_output_init_file(c50_output *output, FILE *file, int owns_file)
{
    memset(output, 0, sizeof(*output));
    output->kind = C50_OUTPUT_FILE;
    output->file = file;
    output->owns_file = owns_file;
}

void c50_output_init_memory(c50_output *output)
{
    memset(output, 0, sizeof(*output));
    output->kind = C50_OUTPUT_MEMORY;
}

int c50_output_printf(c50_output *output, const char *format, ...)
{
    int result;
    va_list arguments;

    va_start(arguments, format);
    result = c50_output_vprintf(output, format, arguments);
    va_end(arguments);
    return result;
}

int c50_output_vprintf(c50_output *output, const char *format,
                       va_list arguments)
{
    int length, result;
    va_list copy;

    if ( output->kind == C50_OUTPUT_FILE )
    {
        return vfprintf(output->file, format, arguments);
    }

    va_copy(copy, arguments);
    length = vsnprintf(NULL, 0, format, copy);
    va_end(copy);
    if ( length < 0 || ! Reserve(output, (size_t) length) )
    {
        return -1;
    }

    result = vsnprintf((char *) output->data + output->size,
                       output->capacity - output->size, format, arguments);
    if ( result < 0 ) return -1;
    output->size += (size_t) result;
    return result;
}

int c50_output_putc(int c, c50_output *output)
{
    if ( output->kind == C50_OUTPUT_FILE ) return fputc(c, output->file);
    if ( ! Reserve(output, 1) ) return EOF;
    output->data[output->size++] = (unsigned char) c;
    output->data[output->size] = '\0';
    return (unsigned char) c;
}

unsigned char *c50_output_take_memory(c50_output *output, size_t *size)
{
    unsigned char *data;

    if ( output->kind != C50_OUTPUT_MEMORY ) return NULL;
    if ( ! Reserve(output, 0) ) return NULL;

    data = output->data;
    if ( size ) *size = output->size;
    output->data = NULL;
    output->size = 0;
    output->capacity = 0;
    return data;
}

int c50_output_close(c50_output *output)
{
    int result = 0;

    if ( output->kind == C50_OUTPUT_FILE && output->file &&
         output->owns_file )
    {
        result = fclose(output->file);
    }
    free(output->data);
    memset(output, 0, sizeof(*output));
    return result;
}
