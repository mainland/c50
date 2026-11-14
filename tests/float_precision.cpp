/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

// Pin the values of math expressions that Release 2.07's C evaluates in
// double precision, where a C++ overload of log, exp, or rint takes a float
// and rounds each step. Every expected value comes from Release 2.07's source
// compiled as C with GCC.

#include <stdio.h>

#include "defns.i"

static int failures;

static void expect_int(const char *label, int actual, int expected)
{
    if ( actual != expected )
    {
	fprintf(stderr, "%s: got %d, expected %d\n", label, actual, expected);
	++failures;
    }
}

int main(int argc, char **argv)
{
    (void) argc;
    (void) argv;

    /*  rint(Val * Mult) / Mult - Val subtracts in double, not in float  */

    expect_int("Denominator(0.498761)", Denominator(0.498761f), 1000000);
    expect_int("Denominator(493.769)", Denominator(493.769f), 1000);

    return failures == 0 ? 0 : 1;
}
