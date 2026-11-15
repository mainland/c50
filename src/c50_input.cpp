/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <errno.h>
#include <string.h>

#include <string>
#include <system_error>

#include "defns.i"
#include "extern.i"
#include "c50_input.h"
#include "c50_api_internal.h"

namespace c50::core {

void c50_input_init_file(c50_input *input, FILE *file, c50_context *context)
{
    input->buffer.resize(c50_input_block_size);
    input->file = file;
    input->context = context;
    input->data = NULL;
    input->size = 0;
    input->next = input->buffer.data();
    input->end = input->buffer.data();
}

void c50_input_init_memory(c50_input *input, const void *data, size_t size)
{
    input->file = NULL;
    input->context = NULL;
    input->data = static_cast<const unsigned char *>(data);
    input->size = size;
    input->next = input->data;
    input->end = input->data + size;
}

bool c50_input_fill(c50_input *input)
{
    if ( ! input->file ) return false;

    size_t count = fread(input->buffer.data(), 1, input->buffer.size(), input->file);

    /* A short count means the end of the file or a read error. Release 2.07
       treated both as the end of the input, so a read error truncated the
       data silently. This project reports it. */
    if ( count < input->buffer.size() && ferror(input->file) )
    {
        const std::string reason =
            std::generic_category().message(errno);
        c50_record_error(input->context, c50::error_code::io_error,
                         ("could not read " + input->context->io.file_name +
                          ": " + reason).c_str());
        C50Exit(input->context, 1);
    }
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

}  // namespace c50::core
