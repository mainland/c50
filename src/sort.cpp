/*************************************************************************/
/*									 */
/*  Copyright 2010 Rulequest Research Pty Ltd.				 */
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
/*	Sort cases from Fp to Lp on attribute Att			 */
/*									 */
/*************************************************************************/


void Quicksort(c50_context *Context, CaseNo Fp, CaseNo Lp, Attribute Att)
/*   ---------  */
{
    CaseNo	i, Middle, High;
    ContValue	Thresh, Val;

    if ( Fp < Lp )
    {
	Thresh = CVal(Context->cases.records[(Fp+Lp) / 2], Att);

	/*  Divide cases into three groups:
		Fp .. Middle-1: values < Thresh
		Middle .. High: values = Thresh
		High+1 .. Lp:   values > Thresh  */

	for ( Middle = Fp ; CVal(Context->cases.records[Middle], Att) < Thresh ; Middle++ )
	    ;

	for ( High = Lp ; CVal(Context->cases.records[High], Att) > Thresh ; High-- )
	    ;

	for ( i = Middle ; i <= High ; )
	{
	    if ( (Val = CVal(Context->cases.records[i], Att)) < Thresh )
	    {
		Swap(Middle, i);
		Middle++;
		i++;
	    }
	    else
	    if ( Val > Thresh )
	    {
		Swap(High, i);
		High--;
	    }
	    else
	    {
		i++;
	    }
	}

	/*  Sort the first and third groups  */

	Quicksort(Context, Fp, Middle-1, Att);
	Quicksort(Context, High+1, Lp, Att);
    }
}
