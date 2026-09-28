/*************************************************************************/
/*									 */
/*  Copyright 2010 Rulequest Research Pty Ltd.				 */
/*  Modifications Copyright 2026 Geoffrey Mainland.			 */
/*									 */
/*  This file is part of C5.0 GPL Edition, a single-threaded version	 */
/*  of C5.0 release 2.07.						 */
/*									 */
/*  C5.0 GPL Edition is free software: you can redistribute it and/or	 */
/*  modify it under the terms of the GNU General Public License as	 */
/*  published by the Free Software Foundation, either version 3 of the	 */
/*  License, or (at your option) any later version.			 */
/*									 */
/*  C5.0 GPL Edition is distributed in the hope that it will be useful,	 */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of	 */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU	 */
/*  General Public License for more details.				 */
/*									 */
/*  You should have received a copy of the GNU General Public License	 */
/*  (gpl.txt) along with C5.0 GPL Edition.  If not, see 		 */
/*									 */
/*      <http://www.gnu.org/licenses/>.					 */
/*									 */
/*************************************************************************/



/*************************************************************************/
/*									 */
/*	Sorting utilities						 */
/*	-----------------						 */
/*									 */
/*************************************************************************/


#include <algorithm>
#include <cstring>
#include <utility>

#include "defns.i"
#include "extern.i"
#include "c50_api_internal.h"



/*************************************************************************/
/*									 */
/*	Sort Records[Fp..Lp] by Key with the RuleQuest three-way	 */
/*	partition.  Equal keys may be reordered, and later floating-	 */
/*	point sums depend on that order, so every comparison and swap	 */
/*	must match the original algorithm.  Only the order in which	 */
/*	the disjoint outer groups are sorted differs: recursing into	 */
/*	the smaller group bounds the stack depth by log2 of the range	 */
/*	without changing the result.					 */
/*									 */
/*************************************************************************/


template <class Record, class KeyFn>
static void PartitionSort(Record *Records, CaseNo Fp, CaseNo Lp, KeyFn Key)
/*          -------------  */
{
    CaseNo	i, Middle, High;
    ContValue	Thresh, Val;

    while ( Fp < Lp )
    {
	/*  Equal to (Fp+Lp) / 2 for nonnegative bounds, without overflow  */

	Thresh = Key(Records[Fp + (Lp-Fp) / 2]);

	/*  Divide elements into three groups:
		Fp .. Middle-1: values < Thresh
		Middle .. High: values = Thresh
		High+1 .. Lp:   values > Thresh  */

	for ( Middle = Fp ; Key(Records[Middle]) < Thresh ; Middle++ )
	    ;

	for ( High = Lp ; Key(Records[High]) > Thresh ; High-- )
	    ;

	for ( i = Middle ; i <= High ; )
	{
	    if ( (Val = Key(Records[i])) < Thresh )
	    {
		std::swap(Records[Middle], Records[i]);
		Middle++;
		i++;
	    }
	    else
	    if ( Val > Thresh )
	    {
		std::swap(Records[High], Records[i]);
		High--;
	    }
	    else
	    {
		i++;
	    }
	}

	/*  Sort the smaller outer group, then continue with the other  */

	if ( Middle - Fp < Lp - High )
	{
	    PartitionSort(Records, Fp, Middle-1, Key);
	    Fp = High+1;
	}
	else
	{
	    PartitionSort(Records, High+1, Lp, Key);
	    Lp = Middle-1;
	}
    }
}

/*************************************************************************/
/*									 */
/*	Sort elements Fp to Lp of SRec					 */
/*									 */
/*************************************************************************/


void Cachesort(CaseNo Fp, CaseNo Lp, SortRec *SRec)
/*   ---------  */
{
    PartitionSort(SRec, Fp, Lp,
		  [](const SortRec &Record) { return Record.V; });
}



/*************************************************************************/
/*									 */
/*	Sort elements Fp to Lp of SRec, keeping equal values in their	 */
/*	input order.  The result is the unique stable order, so it does	 */
/*	not depend on the standard library or the platform.  Ranges	 */
/*	above StableInsertionLimit use a least-significant-digit radix	 */
/*	sort on keys whose unsigned order matches the value order.	 */
/*									 */
/*************************************************************************/


static const CaseNo StableInsertionLimit = 48;


static C50SortKey OrderedKey(C50SortValue Value)
/*                ----------  */
{
    const C50SortKey Sign = C50SortKey(1) << (8 * sizeof(C50SortKey) - 1);
    C50SortKey Bits;

    std::memcpy(&Bits, &Value, sizeof(Bits));

    /*  -0 and +0 compare equal, so they must share a key  */

    if ( Bits == Sign ) Bits = 0;

    return ( Bits & Sign ? ~Bits : Bits | Sign );
}


void StableCachesort(CaseNo Fp, CaseNo Lp, SortRec *SRec,
		     SortRec *RecordScratch, C50SortKey *Keys,
		     C50SortKey *KeyScratch)
/*   ---------------  */
{
    const int	Digits = sizeof(C50SortKey);
    CaseNo	i, j, N;
    SortRec	*Records, *From, *To, Moved;
    C50SortKey	*FromKeys, *ToKeys;
    CaseNo	Count[sizeof(C50SortKey)][256] = {};
    int		d, b;

    if ( Fp >= Lp ) return;

    Records = SRec + Fp;
    N = Lp - Fp + 1;

    if ( N <= StableInsertionLimit )
    {
	ForEach(i, 1, N-1)
	{
	    Moved = Records[i];
	    for ( j = i ; j > 0 && Moved.V < Records[j-1].V ; j-- )
	    {
		Records[j] = Records[j-1];
	    }
	    Records[j] = Moved;
	}
	return;
    }

    ForEach(i, 0, N-1)
    {
	Keys[i] = OrderedKey(Records[i].V);
	ForEach(d, 0, Digits-1)
	{
	    Count[d][(Keys[i] >> (8 * d)) & 255]++;
	}
    }

    From = Records;
    To = RecordScratch;
    FromKeys = Keys;
    ToKeys = KeyScratch;

    ForEach(d, 0, Digits-1)
    {
	CaseNo Next[256], Start = 0;

	/*  A digit shared by every key leaves the order unchanged  */

	if ( Count[d][(FromKeys[0] >> (8 * d)) & 255] == N ) continue;

	ForEach(b, 0, 255)
	{
	    Next[b] = Start;
	    Start += Count[d][b];
	}

	ForEach(i, 0, N-1)
	{
	    j = Next[(FromKeys[i] >> (8 * d)) & 255]++;
	    To[j] = From[i];
	    ToKeys[j] = FromKeys[i];
	}

	std::swap(From, To);
	std::swap(FromKeys, ToKeys);
    }

    if ( From != Records )
    {
	std::copy(From, From + N, Records);
    }
}

/*************************************************************************/
/*									 */
/*	Sort cases from Fp to Lp on attribute Att			 */
/*									 */
/*************************************************************************/


void SortCasesByAttribute(c50_context *Context, CaseNo Fp, CaseNo Lp,
			  Attribute Att)
/*   --------------------  */
{
    PartitionSort(Context->cases.records, Fp, Lp,
		  [Att](DataRec Case) { return CVal(Case, Att); });
}
