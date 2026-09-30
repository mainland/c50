/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef C50_RNG_H
#define C50_RNG_H

#include <stdint.h>

typedef struct
{
    int first;
    int second;
    double values[55];
} KRState;

double KRandom(KRState *state);
void ResetKR(KRState *state, int seed);

/* The drand48 generator in its unseeded GNU C library state, kept per context
 * so that results do not depend on the platform C library or on other
 * callers. */
typedef struct
{
    uint64_t x;
} Drand48State;

double Drand48(Drand48State *state);
void ResetDrand48(Drand48State *state);

#endif
