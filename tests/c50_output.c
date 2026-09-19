/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "c50_output.h"

int main(void)
{
    static const char expected[] = "value=42\nxxxxxxxxxxxxxxxx";
    c50_output output;
    unsigned char *data;
    size_t size;
    FILE *file;
    char buffer[16];

    c50_output_init_memory(&output);
    if ( c50_output_printf(&output, "value=%d\n", 42) != 9 ) return 1;
    if ( c50_output_printf(&output, "%s", "xxxxxxxxxxxxxxxx") != 16 ) return 1;
    data = c50_output_take_memory(&output, &size);
    if ( ! data || size != sizeof(expected) - 1 ) return 1;
    if ( memcmp(data, expected, sizeof(expected)) ) return 1;
    free(data);
    if ( c50_output_close(&output) ) return 1;

    file = tmpfile();
    if ( ! file ) return 1;
    c50_output_init_file(&output, file, 0);
    if ( c50_output_printf(&output, "%s", "file") != 4 ) return 1;
    if ( c50_output_putc('\n', &output) != '\n' ) return 1;
    if ( c50_output_close(&output) ) return 1;
    rewind(file);
    if ( ! fgets(buffer, sizeof(buffer), file) ) return 1;
    if ( strcmp(buffer, "file\n") ) return 1;
    fclose(file);

    return 0;
}
