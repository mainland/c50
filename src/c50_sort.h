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

#ifdef USEDOUBLE
typedef unsigned long long C50SortKey;
#else
typedef unsigned int C50SortKey;
#endif

static_assert(sizeof(C50SortKey) == sizeof(C50SortValue),
              "sort keys must have the width of continuous values");

/* Sort records[first..last] by value with the C5.0 reference tie order. */
void Cachesort(int first, int last, SortRec *records);

/* Sort records[first..last] by value, keeping equal values in their input
   order, as std::stable_sort does. The caller provides scratch arrays with
   at least last-first+1 elements each. */
void StableCachesort(int first, int last, SortRec *records,
                     SortRec *record_scratch, C50SortKey *keys,
                     C50SortKey *key_scratch);

#endif
