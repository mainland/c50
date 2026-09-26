/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef C50_INPUT_H
#define C50_INPUT_H

#include <stddef.h>
#include <stdio.h>

typedef enum c50_input_kind
{
    C50_INPUT_FILE,
    C50_INPUT_MEMORY
} c50_input_kind;

typedef struct c50_input
{
    c50_input_kind kind;
    FILE *file;
    const unsigned char *data;
    size_t size;
    size_t position;
} c50_input;

/* Initialize a non-owning view of file. */
void c50_input_init_file(c50_input *input, FILE *file);

/* Initialize a non-owning view of data[0..size). */
void c50_input_init_memory(c50_input *input, const void *data, size_t size);

int c50_input_getc(c50_input *input);
char *c50_input_gets(char *buffer, size_t capacity, c50_input *input);
void c50_input_rewind(c50_input *input);

#endif
