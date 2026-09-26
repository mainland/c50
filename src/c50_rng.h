/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef C50_RNG_H
#define C50_RNG_H

typedef struct
{
    int first;
    int second;
    double values[55];
} KRState;

double KRandom(KRState *state);
void ResetKR(KRState *state, int seed);

#endif
