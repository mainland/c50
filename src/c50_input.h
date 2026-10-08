/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef C50_INPUT_H
#define C50_INPUT_H

#include <stddef.h>
#include <stdio.h>

#include <string>
#include <vector>

/* The number of bytes that one refill reads from a file. */
constexpr size_t c50_input_block_size = 65536;

/* A byte source for the parsers: an open file or text held in memory.
   Bytes are read from a window, [next, end), so that reading one byte is a
   comparison and an increment. Text in memory is a single window over the
   caller's bytes. A file is read in blocks into a heap buffer, which
   c50_input_fill() refills, so that inputs stay small on the stack. An input
   therefore reads ahead of the bytes it returns, so a file must be read only
   through one input until it is closed or rewound. An input cannot be
   copied, because a copy's window would point into the original's buffer. */
typedef struct c50_input
{
    c50_input() = default;
    c50_input(const c50_input &) = delete;
    c50_input &operator=(const c50_input &) = delete;

    const unsigned char *next = nullptr;
    const unsigned char *end = nullptr;
    FILE *file = nullptr;
    const unsigned char *data = nullptr;
    size_t size = 0;
    std::vector<unsigned char> buffer;
} c50_input;

/* Initialize a non-owning view of file. */
void c50_input_init_file(c50_input *input, FILE *file);

/* Initialize a non-owning view of data[0..size). */
void c50_input_init_memory(c50_input *input, const void *data, size_t size);

/* Refill an empty window with the next block of a file. Return false at the
   end of the input. */
bool c50_input_fill(c50_input *input);

inline int c50_input_getc(c50_input *input)
{
    if ( input->next == input->end && ! c50_input_fill(input) ) return EOF;
    return *input->next++;
}

/* Append the bytes through the next newline, or through the end of the
   input, to line. Return false if no bytes remain. */
bool c50_input_read_line(c50_input *input, std::string &line);

char *c50_input_gets(char *buffer, size_t capacity, c50_input *input);
void c50_input_rewind(c50_input *input);

#endif
