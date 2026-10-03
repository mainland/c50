/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

/*
 * Run a command and write its resource usage as JSON.
 *
 * Usage: measure OUTPUT COMMAND [ARGUMENT...]
 *
 * On Linux, the peak resident set size that wait4 reports for a child
 * includes the memory of the process that started it, at the time of the
 * fork. The benchmark driver can grow to gigabytes while it prepares
 * datasets, so it starts each measured command through this small program
 * instead of directly.
 */

#include <stdio.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>

static double seconds(struct timeval value)
{
    return (double) value.tv_sec + (double) value.tv_usec / 1e6;
}

int main(int argc, char **argv)
{
    pid_t pid;
    int status, exit_status;
    struct rusage usage;
    FILE *output;

    if ( argc < 3 )
    {
        fprintf(stderr, "usage: measure OUTPUT COMMAND [ARGUMENT...]\n");
        return 2;
    }

    pid = fork();
    if ( pid < 0 )
    {
        perror("fork");
        return 2;
    }
    if ( pid == 0 )
    {
        execvp(argv[2], argv + 2);
        perror(argv[2]);
        _exit(127);
    }

    if ( wait4(pid, &status, 0, &usage) < 0 )
    {
        perror("wait4");
        return 2;
    }
    exit_status = WIFEXITED(status) ? WEXITSTATUS(status)
                                    : 128 + WTERMSIG(status);

    if ( ! (output = fopen(argv[1], "w")) )
    {
        perror(argv[1]);
        return 2;
    }
    fprintf(output,
            "{\"status\": %d, \"user_seconds\": %.6f, "
            "\"system_seconds\": %.6f, \"maxrss\": %ld}\n",
            exit_status, seconds(usage.ru_utime), seconds(usage.ru_stime),
            (long) usage.ru_maxrss);
    if ( fclose(output) != 0 )
    {
        perror(argv[1]);
        return 2;
    }
    return exit_status;
}
