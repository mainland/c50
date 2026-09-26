/* Copyright 2010 Rulequest Research Pty Ltd. */
/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef C50_SORT_H
#define C50_SORT_H

#ifdef USEDOUBLE
typedef double C50SortValue;
#else
typedef float C50SortValue;
#endif

typedef struct _sort_rec
{
    C50SortValue V;
    int C;
    float W;
} SortRec;

void Cachesort(int first, int last, SortRec *records);

#endif
