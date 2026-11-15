/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

// Pin the values of math expressions that Release 2.07's C evaluates in
// double precision, where a C++ overload of log, exp, or rint takes a float
// and rounds each step. Every expected value comes from Release 2.07's source
// compiled as C with GCC.

#include <stdio.h>

#include <cstring>

#include "defns.i"
#include "extern.i"
#include "c50_api_internal.h"

using namespace c50::core;

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
    auto owner = c50_make_context();
    c50_context *context = owner.get();

    /*  The float Val - rint(Val * Mult) / Mult must subtract in double  */

    expect_int("Denominator(0.498761)", Denominator(0.498761f), 1000000);
    expect_int("Denominator(493.769)", Denominator(493.769f), 1000);

    /*  N * (1 - exp(log(CF) / N)) cancels for large N, and its float steps
	give 1.3709068 at N = 10^6 instead of 1.3862934  */

    char text[32];
    snprintf(text, sizeof(text), "%.7f",
	     (double) RawExtraErrs(context, 1000000, 0));
    if ( strcmp(text, "1.3862934") )
    {
	fprintf(stderr, "RawExtraErrs(10^6, 0): got %s, expected 1.3862934\n",
		text);
	++failures;
    }

    /*  log(n-1) of a float n-1 near 10^5 rounds to a different float  */

    expect_int("Digits(9)", Digits(9), 3);
    expect_int("Digits(100001)", Digits(100001), 8);
    expect_int("Digits(99999)", Digits(99999), 7);

    return failures == 0 ? 0 : 1;
}
