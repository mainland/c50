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
/*      Evaluation of discrete attribute subsets			 */
/*      ----------------------------------------			 */
/*									 */
/*************************************************************************/


#include "defns.i"
#include "extern.i"
#include "c50_api_internal.h"



/*************************************************************************/
/*									 */
/*	Set up tables for subsets					 */
/*									 */
/*************************************************************************/


void InitialiseBellNumbers(c50_context *Context)
/*   ---------------------  */
{
    DiscrValue	 n, k;

    /*  Table of Context->splits.bell_numbers numbers (used for subset test penalties)  */

    Context->splits.bell_numbers = AllocZero(Context->schema.max_discrete_value+1, double *);
    ForEach(n, 1, Context->schema.max_discrete_value)
    {
	Context->splits.bell_numbers[n] = AllocZero(n+1, double);
	ForEach(k, 1, n)
	{
	    Context->splits.bell_numbers[n][k] = ( k == 1 || k == n ? 1 :
			   Context->splits.bell_numbers[n-1][k-1] + k * Context->splits.bell_numbers[n-1][k] );
	}
    }
}



/*************************************************************************/
/*									 */
/*	Evaluate subsetting a discrete attribute and form the chosen	 */
/*	subsets Context->splits.subsets[Att][], setting Context->splits.subset_counts[Att] to the number of	 */
/*	subsets, and the Context->splits.information[] and Context->splits.gain[] of a test on the attribute	 */
/*									 */
/*************************************************************************/


void EvalSubset(c50_context *Context, Attribute Att, CaseCount Cases)
/*   ----------  */
{
    DiscrValue	V1, V2, V3, BestV1, BestV2, InitialBlocks, First=1, Prelim=0;
    ClassNo	c;
    double	BaseInfo, ThisGain, ThisInfo, Penalty, UnknownRate,
		Val, BestVal, BestGain, BestInfo, PrevGain, PrevInfo;
    int		MissingValues=0;
    CaseCount	KnownCases;
    Boolean	Better;

    /*  First compute Freq[][], ValFreq[], base info, and the gain
	and total info of a split on discrete attribute Att  */

    SetDiscrFreq(Context, Att);

    Context->training.environment->ReasonableSubsets = 0;
    ForEach(c, 1, Context->schema.max_attribute_value[Att])
    {
	if ( Context->training.environment->ValFreq[c] >= Context->options.minimum_cases ) Context->training.environment->ReasonableSubsets++;
    }

    if ( ! Context->training.environment->ReasonableSubsets )
    {
	Verbosity(2,
	    fprintf(Of, "\tAtt %s: poor initial split\n", Context->schema.attribute_names[Att]))

	return;
    }

    KnownCases  = Cases - Context->training.environment->ValFreq[0];
    UnknownRate = Context->training.environment->ValFreq[0] / Cases;

    BaseInfo = ( ! Context->training.environment->ValFreq[0] ? Context->splits.base_information :
		     DiscrKnownBaseInfo(Context, KnownCases, Context->schema.max_attribute_value[Att]) );

    PrevGain = ComputeGain(Context, BaseInfo, UnknownRate, Context->schema.max_attribute_value[Att], KnownCases);
    PrevInfo = TotalInfo(Context->training.environment->ValFreq, 0, Context->schema.max_attribute_value[Att]) / Cases;
    BestVal  = PrevGain / PrevInfo;

    Verbosity(2, fprintf(Of, "\tAtt %s", Context->schema.attribute_names[Att]))
    Verbosity(3, PrintDistribution(Context, Att, 0, Context->schema.max_attribute_value[Att], Context->training.environment->Freq,
				   Context->training.environment->ValFreq, true))
    Verbosity(2,
	fprintf(Of, "\tinitial inf %.3f, gain %.3f, val=%.3f\n",
		PrevInfo, PrevGain, BestVal))

    /*  Eliminate unrepresented attribute values from Freq[] and ValFreq[]
	and form a separate subset for each represented attribute value.
	Unrepresented N/A values are ignored  */

    Context->training.environment->Bytes = (Context->schema.max_attribute_value[Att]>>3) + 1;
    ClearBits(Context->training.environment->Bytes, Context->splits.subsets[Att][0]);

    Context->training.environment->Blocks = 0;
    ForEach(V1, 1, Context->schema.max_attribute_value[Att])
    {
	if ( Context->training.environment->ValFreq[V1] > Epsilon ||
	     ( V1 == 1 && Context->cases.some_not_applicable[Att] ) )
	{
	    if ( ++Context->training.environment->Blocks < V1 )
	    {
		Context->training.environment->ValFreq[Context->training.environment->Blocks] = Context->training.environment->ValFreq[V1];
		ForEach(c, 1, Context->schema.max_class)
		{
		    Context->training.environment->Freq[Context->training.environment->Blocks][c] = Context->training.environment->Freq[V1][c];
		}
	    }
	    ClearBits(Context->training.environment->Bytes, Context->training.environment->WSubset[Context->training.environment->Blocks]);
	    SetBit(V1, Context->training.environment->WSubset[Context->training.environment->Blocks]);
	    CopyBits(Context->training.environment->Bytes, Context->training.environment->WSubset[Context->training.environment->Blocks],
		     Context->splits.subsets[Att][Context->training.environment->Blocks]);

	    /*  Cannot merge N/A values with other blocks  */

	    if ( V1 == 1 ) First = 2;
	}
	else
	if ( V1 != 1 )
	{
	    SetBit(V1, Context->splits.subsets[Att][0]);
	    MissingValues++;
	}
    }

    /*  Set n-way branch as initial test  */

    Context->splits.gain[Att]    = PrevGain;
    Context->splits.information[Att]    = PrevInfo;
    Context->splits.subset_counts[Att] = InitialBlocks = Context->training.environment->Blocks;

    /*  As a preliminary step, merge values with identical distributions  */

    ForEach(V1, First, Context->training.environment->Blocks-1)
    {
	ForEach(V2, V1+1, Context->training.environment->Blocks)
	{
	    if ( SameDistribution(Context, V1, V2) )
	    {
		Prelim = V1;
		AddBlock(Context, V1, V2);
	    }
	}

	/*  Eliminate any merged values  */

	if ( Prelim == V1 )
	{
	    V3 = V1;

	    ForEach(V2, V1+1, Context->training.environment->Blocks)
	    {
		if ( Context->training.environment->ValFreq[V2] && ++V3 != V2 )
		{
		    MoveBlock(Context, V3, V2);
		}
	    }

	    Context->training.environment->Blocks = V3;
	}
    }

    if ( Prelim )
    {
	PrevInfo = TotalInfo(Context->training.environment->ValFreq, 0, Context->training.environment->Blocks) / Cases;

	Penalty  = ( finite(Context->splits.bell_numbers[InitialBlocks][Context->training.environment->Blocks]) ?
			Log(Context->splits.bell_numbers[InitialBlocks][Context->training.environment->Blocks]) :
			(InitialBlocks-Context->training.environment->Blocks+1) * Log(Context->training.environment->Blocks) );

	Val = (PrevGain - Penalty / Cases) / PrevInfo;
	Better = ( Context->training.environment->Blocks >= 2 && Context->training.environment->ReasonableSubsets >= 2 &&
		   Val >= BestVal );

	Verbosity(2,
	{
	    fprintf(Of, "\tprelim merges -> inf %.3f, gain %.3f, val %.3f%s%s",
			PrevInfo, PrevGain, Val,
		        ( Better ? " **" : "" ),
			(Context->options.verbosity > 2 ? "" : "\n" ));
	    Verbosity(3, PrintDistribution(Context, Att, 0, Context->training.environment->Blocks, Context->training.environment->Freq,
					   Context->training.environment->ValFreq, false))
	})

	if ( Better )
	{
	    Context->splits.subset_counts[Att] = Context->training.environment->Blocks;

	    ForEach(V1, 1, Context->training.environment->Blocks)
	    {
		CopyBits(Context->training.environment->Bytes, Context->training.environment->WSubset[V1], Context->splits.subsets[Att][V1]);
	    }

	    Context->splits.information[Att] = PrevInfo;
	    Context->splits.gain[Att] = PrevGain - Penalty / KnownCases;
	    BestVal   = Val;
	}
    }
		
    /*  Determine initial information and entropy values  */

    ForEach(V1, 1, Context->training.environment->Blocks)
    {
	Context->training.environment->SubsetInfo[V1] = -Context->training.environment->ValFreq[V1] * Log(Context->training.environment->ValFreq[V1] / Cases);
	Context->training.environment->SubsetEntr[V1] = TotalInfo(Context->training.environment->Freq[V1], 1, Context->schema.max_class);
    }

    ForEach(V1, First, Context->training.environment->Blocks-1)
    {
	ForEach(V2, V1+1, Context->training.environment->Blocks)
	{
	    EvaluatePair(Context, V1, V2, Cases);
	}
    }

    /*  Examine possible pair mergers and hill-climb  */

    while ( Context->training.environment->Blocks > 2 )
    {
	BestV1 = 0;
	BestGain = -Epsilon;

	/*  For each possible pair of values, calculate the gain and
	    total info of a split in which they are treated as one.
	    Keep track of the pair with the best gain.  */

	ForEach(V1, First, Context->training.environment->Blocks-1)
	{
	    ForEach(V2, V1+1, Context->training.environment->Blocks)
	    {
		if ( Context->training.environment->ReasonableSubsets == 2 &&
		     Context->training.environment->ValFreq[V1] >= Context->options.minimum_cases-Epsilon &&
		     Context->training.environment->ValFreq[V2] >= Context->options.minimum_cases-Epsilon )
		{
		    continue;
		}

		ThisGain = PrevGain -
			   ((1-UnknownRate) / KnownCases) *
			     (Context->training.environment->MergeEntr[V1][V2] -
			       (Context->training.environment->SubsetEntr[V1] + Context->training.environment->SubsetEntr[V2]));
		ThisInfo = PrevInfo + (Context->training.environment->MergeInfo[V1][V2] -
			   (Context->training.environment->SubsetInfo[V1] + Context->training.environment->SubsetInfo[V2])) / Cases;
		Verbosity(3,
		    fprintf(Of, "\t    combine %d %d info %.3f gain %.3f\n",
			    V1, V2, ThisInfo, ThisGain))

		/*  See whether this merge has the best gain so far  */

		if ( ThisGain > BestGain+Epsilon )
		{
		    BestGain = ThisGain;
		    BestInfo = ThisInfo;
		    BestV1   = V1;
		    BestV2   = V2;
		}
	    }
	}

	if ( ! BestV1 ) break;

	PrevGain = BestGain;
	PrevInfo = BestInfo;

	/*  Determine penalty as log of Context->splits.bell_numbers number.  If number is too
	    large, use an approximation of log  */

	Penalty  = ( finite(Context->splits.bell_numbers[InitialBlocks][Context->training.environment->Blocks-1]) ?
			Log(Context->splits.bell_numbers[InitialBlocks][Context->training.environment->Blocks-1]) :
			(InitialBlocks-Context->training.environment->Blocks+1) * Log(Context->training.environment->Blocks-1) );

	Val = (BestGain - Penalty / Cases) / BestInfo;

	Merge(Context, BestV1, BestV2, Cases);

	Verbosity(2,
	    fprintf(Of, "\tform subset ");
	    PrintSubset(Context, Att, Context->training.environment->WSubset[BestV1]);
	    fprintf(Of, ": %d subsets, inf %.3f, gain %.3f, val %.3f%s\n",
		   Context->training.environment->Blocks, BestInfo, BestGain, Val,
		   ( Val > BestVal ? " **" : "" ));
	    Verbosity(3,
		PrintDistribution(Context, Att, 0, Context->training.environment->Blocks, Context->training.environment->Freq, Context->training.environment->ValFreq,
				  false))
	    )

	if ( Val >= BestVal )
	{
	    Context->splits.subset_counts[Att] = Context->training.environment->Blocks;

	    ForEach(V1, 1, Context->training.environment->Blocks)
	    {
		CopyBits(Context->training.environment->Bytes, Context->training.environment->WSubset[V1], Context->splits.subsets[Att][V1]);
	    }

	    Context->splits.information[Att] = BestInfo;
	    Context->splits.gain[Att] = BestGain - Penalty / KnownCases;
	    BestVal   = Val;
	}
    }

    /*  Add missing values as another branch  */

    if ( MissingValues )
    {
	Context->splits.subset_counts[Att]++;
	CopyBits(Context->training.environment->Bytes, Context->splits.subsets[Att][0], Context->splits.subsets[Att][Context->splits.subset_counts[Att]]);
    }

    Verbosity(2,
	fprintf(Of, "\tfinal inf %.3f, gain %.3f, val=%.3f\n",
		Context->splits.information[Att], Context->splits.gain[Att], Context->splits.gain[Att] / (Context->splits.information[Att] + 1E-3)))
}



/*************************************************************************/
/*									 */
/*	Combine the distribution figures of subsets x and y.		 */
/*	Update Freq, ValFreq, SubsetInfo, SubsetEntr, MergeInfo, and	 */
/*	MergeEntr.							 */
/*									 */
/*************************************************************************/


void Merge(c50_context *Context, DiscrValue x, DiscrValue y,
	   CaseCount Cases)
/*   -----  */
{
    ClassNo	c;
    double	Entr=0;
    CaseCount	KnownCases=0;
    int		R, C;

    AddBlock(Context, x, y);

    ForEach(c, 1, Context->schema.max_class)
    {
	Entr -= Context->training.environment->Freq[x][c] * Log(Context->training.environment->Freq[x][c]);
	KnownCases += Context->training.environment->Freq[x][c];
    }

    Context->training.environment->SubsetInfo[x] = - Context->training.environment->ValFreq[x] * Log(Context->training.environment->ValFreq[x] / Cases);
    Context->training.environment->SubsetEntr[x] = Entr + KnownCases * Log(KnownCases);

    /*  Eliminate y from working blocks  */

    ForEach(R, y, Context->training.environment->Blocks-1)
    {
	MoveBlock(Context, R, R+1);

	Context->training.environment->SubsetInfo[R] = Context->training.environment->SubsetInfo[R+1];
	Context->training.environment->SubsetEntr[R] = Context->training.environment->SubsetEntr[R+1];

	ForEach(C, 1, Context->training.environment->Blocks)
	{
	    Context->training.environment->MergeInfo[R][C] = Context->training.environment->MergeInfo[R+1][C];
	    Context->training.environment->MergeEntr[R][C] = Context->training.environment->MergeEntr[R+1][C];
	}
    }

    ForEach(C, y, Context->training.environment->Blocks-1)
    {
	ForEach(R, 1, Context->training.environment->Blocks-1)
	{
	    Context->training.environment->MergeInfo[R][C] = Context->training.environment->MergeInfo[R][C+1];
	    Context->training.environment->MergeEntr[R][C] = Context->training.environment->MergeEntr[R][C+1];
	}
    }
    Context->training.environment->Blocks--;

    /*  Update information for newly-merged block  */

    ForEach(C, 1, Context->training.environment->Blocks)
    {
	if ( C != x ) EvaluatePair(Context, x, C, Cases);
    }
}



/*************************************************************************/
/*									 */
/*	Calculate the effect of merging subsets x and y			 */
/*									 */
/*************************************************************************/


void EvaluatePair(c50_context *Context, DiscrValue x, DiscrValue y,
		  CaseCount Cases)
/*   ------------  */
{
    ClassNo	c;
    double	Entr=0;
    CaseCount	KnownCases=0, F;

    if ( y < x )
    {
	c = y;
	y = x;
	x = c;
    }

    F = Context->training.environment->ValFreq[x] + Context->training.environment->ValFreq[y];
    Context->training.environment->MergeInfo[x][y] = - F * Log(F / Cases);

    ForEach(c, 1, Context->schema.max_class)
    {
	F = Context->training.environment->Freq[x][c] + Context->training.environment->Freq[y][c];
	Entr -= F * Log(F);
	KnownCases += F;
    }
    Context->training.environment->MergeEntr[x][y] = Entr + KnownCases * Log(KnownCases);
}



/*************************************************************************/
/*									 */
/*	Check whether two values have same class distribution		 */
/*									 */
/*************************************************************************/


Boolean SameDistribution(c50_context *Context, DiscrValue V1,
			 DiscrValue V2)
/*	----------------  */
{
    ClassNo	c;
    CaseCount	D1, D2;

    D1 = Context->training.environment->ValFreq[V1];
    D2 = Context->training.environment->ValFreq[V2];

    ForEach(c, 1, Context->schema.max_class)
    {
	if ( fabs(Context->training.environment->Freq[V1][c] / D1 - Context->training.environment->Freq[V2][c] / D2) > 0.001 )
	{
	    return false;
	}
    }

    return true;
}



/*************************************************************************/
/*									 */
/*	Add frequency and subset information from block V2 to V1	 */
/*									 */
/*************************************************************************/


void AddBlock(c50_context *Context, DiscrValue V1, DiscrValue V2)
/*   --------  */
{
    ClassNo	c;
    int		b;

    if ( Context->training.environment->ValFreq[V1] >= Context->options.minimum_cases-Epsilon &&
	 Context->training.environment->ValFreq[V2] >= Context->options.minimum_cases-Epsilon )
    {
	Context->training.environment->ReasonableSubsets--;
    }
    else
    if ( Context->training.environment->ValFreq[V1] < Context->options.minimum_cases-Epsilon &&
	 Context->training.environment->ValFreq[V2] < Context->options.minimum_cases-Epsilon &&
	 Context->training.environment->ValFreq[V1] + Context->training.environment->ValFreq[V2] >= Context->options.minimum_cases-Epsilon )
    {
	Context->training.environment->ReasonableSubsets++;
    }

    ForEach(c, 1, Context->schema.max_class)
    {
	Context->training.environment->Freq[V1][c] += Context->training.environment->Freq[V2][c];
    }
    Context->training.environment->ValFreq[V1] += Context->training.environment->ValFreq[V2];
    Context->training.environment->ValFreq[V2] = 0;
    ForEach(b, 0, Context->training.environment->Bytes-1)
    {
	Context->training.environment->WSubset[V1][b] |= Context->training.environment->WSubset[V2][b];
    }
}



/*************************************************************************/
/*									 */
/*	Move frequency and subset information from block V2 to V1	 */
/*									 */
/*************************************************************************/


void MoveBlock(c50_context *Context, DiscrValue V1, DiscrValue V2)
/*   ---------  */
{
    ClassNo	c;

    ForEach(c, 1, Context->schema.max_class)
    {
	Context->training.environment->Freq[V1][c] = Context->training.environment->Freq[V2][c];
    }
    Context->training.environment->ValFreq[V1] = Context->training.environment->ValFreq[V2];
    CopyBits(Context->training.environment->Bytes, Context->training.environment->WSubset[V2], Context->training.environment->WSubset[V1]);
}



/*************************************************************************/
/*									 */
/*	Print the values of attribute Att which are in the subset Ss	 */
/*									 */
/*************************************************************************/


void PrintSubset(c50_context *Context, Attribute Att, Set Ss)
/*   -----------  */
{
    DiscrValue	V1;
    Boolean	First=true;

    ForEach(V1, 1, Context->schema.max_attribute_value[Att])
    {
	if ( In(V1, Ss) )
	{
	    if ( First )
	    {
		First = false;
	    }
	    else
	    {
		fprintf(Of, ", ");
	    }

	    fprintf(Of, "%s", Context->schema.attribute_value_names[Att][V1]);
	}
    }
}



/*************************************************************************/
/*									 */
/*	Construct and return a node for a test on a subset of values	 */
/*									 */
/*************************************************************************/


void SubsetTest(c50_context *Context, Tree Node, Attribute Att)
/*   -----------  */
{
    int	S, Bytes;

    Sprout(Node, Context->splits.subset_counts[Att]);

    Node->NodeType = BrSubset;
    Node->Tested   = Att;

    Bytes = (Context->schema.max_attribute_value[Att]>>3) + 1;
    Node->Subset = AllocZero(Context->splits.subset_counts[Att]+1, Set);
    ForEach(S, 1, Node->Forks)
    {
	Node->Subset[S] = Alloc(Bytes, Byte);
	CopyBits(Bytes, Context->splits.subsets[Att][S], Node->Subset[S]);
    }
}
