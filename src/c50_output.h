/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef C50_OUTPUT_H
#define C50_OUTPUT_H

#include <stddef.h>
#include <stdio.h>

typedef enum c50_output_kind
{
    C50_OUTPUT_FILE,
    C50_OUTPUT_MEMORY
} c50_output_kind;

typedef struct c50_output
{
    c50_output_kind kind;
    FILE *file;
    unsigned char *data;
    size_t size;
    size_t capacity;
    int owns_file;
} c50_output;

/* Initialize an output that writes to file. */
void c50_output_init_file(c50_output *output, FILE *file, int owns_file);

/* Initialize an output that accumulates bytes in memory. */
void c50_output_init_memory(c50_output *output);

int c50_output_printf(c50_output *output, const char *format, ...);
int c50_output_putc(int c, c50_output *output);

/*
 * Transfer ownership of a memory output's buffer to the caller. The returned
 * buffer has a trailing NUL byte that is not included in *size.
 */
unsigned char *c50_output_take_memory(c50_output *output, size_t *size);

/* Close an owned file or release a memory buffer. */
int c50_output_close(c50_output *output);

#endif
