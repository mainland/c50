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
/*	subsets in the per-attribute result, setting its subset count to the number of	 */
/*	subsets, and the Context->splits.information[] and Context->splits.gain[] of a test on the attribute	 */
/*									 */
/*************************************************************************/


void EvalSubset(c50_context *Context, SplitWorkspace &Workspace, SplitResult &Result, Attribute Att, CaseCount Cases)
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

    SetDiscrFreq(Context, Workspace, Att);

    Workspace.ReasonableSubsets = 0;
    ForEach(c, 1, Context->schema.max_attribute_value[Att])
    {
	if ( Workspace.ValFreq[c] >= Context->options.minimum_cases ) Workspace.ReasonableSubsets++;
    }

    if ( ! Workspace.ReasonableSubsets )
    {
	Verbosity(2,
	    fprintf(Context->io.output, "\tAtt %s: poor initial split\n", Context->schema.attribute_names[Att]))

	return;
    }

    KnownCases  = Cases - Workspace.ValFreq[0];
    UnknownRate = Workspace.ValFreq[0] / Cases;

    BaseInfo = ( ! Workspace.ValFreq[0] ? Context->splits.base_information :
		     DiscrKnownBaseInfo(Context, Workspace, KnownCases, Context->schema.max_attribute_value[Att]) );

    PrevGain = ComputeGain(Context, Workspace, BaseInfo, UnknownRate, Context->schema.max_attribute_value[Att], KnownCases);
    PrevInfo = TotalInfo(Workspace.ValFreq, 0, Context->schema.max_attribute_value[Att]) / Cases;
    BestVal  = PrevGain / PrevInfo;

    Verbosity(2, fprintf(Context->io.output, "\tAtt %s", Context->schema.attribute_names[Att]))
    Verbosity(3, PrintDistribution(Context, Att, 0, Context->schema.max_attribute_value[Att], Workspace.Freq,
				   Workspace.ValFreq, true))
    Verbosity(2,
	fprintf(Context->io.output, "\tinitial inf %.3f, gain %.3f, val=%.3f\n",
		PrevInfo, PrevGain, BestVal))

    /*  Eliminate unrepresented attribute values from Freq[] and ValFreq[]
	and form a separate subset for each represented attribute value.
	Unrepresented N/A values are ignored  */

    Workspace.Bytes = (Context->schema.max_attribute_value[Att]>>3) + 1;
    ClearBits(Workspace.Bytes, Result.Subsets[0]);

    Workspace.Blocks = 0;
    ForEach(V1, 1, Context->schema.max_attribute_value[Att])
    {
	if ( Workspace.ValFreq[V1] > Epsilon ||
	     ( V1 == 1 && Context->cases.some_not_applicable[Att] ) )
	{
	    if ( ++Workspace.Blocks < V1 )
	    {
		Workspace.ValFreq[Workspace.Blocks] = Workspace.ValFreq[V1];
		ForEach(c, 1, Context->schema.max_class)
		{
		    Workspace.Freq[Workspace.Blocks][c] = Workspace.Freq[V1][c];
		}
	    }
	    ClearBits(Workspace.Bytes, Workspace.WSubset[Workspace.Blocks]);
	    SetBit(V1, Workspace.WSubset[Workspace.Blocks]);
	    CopyBits(Workspace.Bytes, Workspace.WSubset[Workspace.Blocks],
		     Result.Subsets[Workspace.Blocks]);

	    /*  Cannot merge N/A values with other blocks  */

	    if ( V1 == 1 ) First = 2;
	}
	else
	if ( V1 != 1 )
	{
	    SetBit(V1, Result.Subsets[0]);
	    MissingValues++;
	}
    }

    /*  Set n-way branch as initial test  */

    Result.Gain    = PrevGain;
    Result.Information    = PrevInfo;
    Result.SubsetCount = InitialBlocks = Workspace.Blocks;

    /*  As a preliminary step, merge values with identical distributions  */

    ForEach(V1, First, Workspace.Blocks-1)
    {
	ForEach(V2, V1+1, Workspace.Blocks)
	{
	    if ( SameDistribution(Context, Workspace, V1, V2) )
	    {
		Prelim = V1;
		AddBlock(Context, Workspace, V1, V2);
	    }
	}

	/*  Eliminate any merged values  */

	if ( Prelim == V1 )
	{
	    V3 = V1;

	    ForEach(V2, V1+1, Workspace.Blocks)
	    {
		if ( Workspace.ValFreq[V2] && ++V3 != V2 )
		{
		    MoveBlock(Context, Workspace, V3, V2);
		}
	    }

	    Workspace.Blocks = V3;
	}
    }

    if ( Prelim )
    {
	PrevInfo = TotalInfo(Workspace.ValFreq, 0, Workspace.Blocks) / Cases;

	Penalty  = ( finite(Context->splits.bell_numbers[InitialBlocks][Workspace.Blocks]) ?
			Log(Context->splits.bell_numbers[InitialBlocks][Workspace.Blocks]) :
			(InitialBlocks-Workspace.Blocks+1) * Log(Workspace.Blocks) );

	Val = (PrevGain - Penalty / Cases) / PrevInfo;
	Better = ( Workspace.Blocks >= 2 && Workspace.ReasonableSubsets >= 2 &&
		   Val >= BestVal );

	Verbosity(2,
	{
	    fprintf(Context->io.output, "\tprelim merges -> inf %.3f, gain %.3f, val %.3f%s%s",
			PrevInfo, PrevGain, Val,
		        ( Better ? " **" : "" ),
			(Context->options.verbosity > 2 ? "" : "\n" ));
	    Verbosity(3, PrintDistribution(Context, Att, 0, Workspace.Blocks, Workspace.Freq,
					   Workspace.ValFreq, false))
	})

	if ( Better )
	{
	    Result.SubsetCount = Workspace.Blocks;

	    ForEach(V1, 1, Workspace.Blocks)
	    {
		CopyBits(Workspace.Bytes, Workspace.WSubset[V1], Result.Subsets[V1]);
	    }

	    Result.Information = PrevInfo;
	    Result.Gain = PrevGain - Penalty / KnownCases;
	    BestVal   = Val;
	}
    }
		
    /*  Determine initial information and entropy values  */

    ForEach(V1, 1, Workspace.Blocks)
    {
	Workspace.SubsetInfo[V1] = -Workspace.ValFreq[V1] * Log(Workspace.ValFreq[V1] / Cases);
	Workspace.SubsetEntr[V1] = TotalInfo(Workspace.Freq[V1], 1, Context->schema.max_class);
    }

    ForEach(V1, First, Workspace.Blocks-1)
    {
	ForEach(V2, V1+1, Workspace.Blocks)
	{
	    EvaluatePair(Context, Workspace, V1, V2, Cases);
	}
    }

    /*  Examine possible pair mergers and hill-climb  */

    while ( Workspace.Blocks > 2 )
    {
	BestV1 = 0;
	BestGain = -Epsilon;

	/*  For each possible pair of values, calculate the gain and
	    total info of a split in which they are treated as one.
	    Keep track of the pair with the best gain.  */

	ForEach(V1, First, Workspace.Blocks-1)
	{
	    ForEach(V2, V1+1, Workspace.Blocks)
	    {
		if ( Workspace.ReasonableSubsets == 2 &&
		     Workspace.ValFreq[V1] >= Context->options.minimum_cases-Epsilon &&
		     Workspace.ValFreq[V2] >= Context->options.minimum_cases-Epsilon )
		{
		    continue;
		}

		ThisGain = PrevGain -
			   ((1-UnknownRate) / KnownCases) *
			     (Workspace.MergeEntr[V1][V2] -
			       (Workspace.SubsetEntr[V1] + Workspace.SubsetEntr[V2]));
		ThisInfo = PrevInfo + (Workspace.MergeInfo[V1][V2] -
			   (Workspace.SubsetInfo[V1] + Workspace.SubsetInfo[V2])) / Cases;
		Verbosity(3,
		    fprintf(Context->io.output, "\t    combine %d %d info %.3f gain %.3f\n",
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

	Penalty  = ( finite(Context->splits.bell_numbers[InitialBlocks][Workspace.Blocks-1]) ?
			Log(Context->splits.bell_numbers[InitialBlocks][Workspace.Blocks-1]) :
			(InitialBlocks-Workspace.Blocks+1) * Log(Workspace.Blocks-1) );

	Val = (BestGain - Penalty / Cases) / BestInfo;

	Merge(Context, Workspace, BestV1, BestV2, Cases);

	Verbosity(2,
	    fprintf(Context->io.output, "\tform subset ");
	    PrintSubset(Context, Att, Workspace.WSubset[BestV1]);
	    fprintf(Context->io.output, ": %d subsets, inf %.3f, gain %.3f, val %.3f%s\n",
		   Workspace.Blocks, BestInfo, BestGain, Val,
		   ( Val > BestVal ? " **" : "" ));
	    Verbosity(3,
		PrintDistribution(Context, Att, 0, Workspace.Blocks, Workspace.Freq, Workspace.ValFreq,
				  false))
	    )

	if ( Val >= BestVal )
	{
	    Result.SubsetCount = Workspace.Blocks;

	    ForEach(V1, 1, Workspace.Blocks)
	    {
		CopyBits(Workspace.Bytes, Workspace.WSubset[V1], Result.Subsets[V1]);
	    }

	    Result.Information = BestInfo;
	    Result.Gain = BestGain - Penalty / KnownCases;
	    BestVal   = Val;
	}
    }

    /*  Add missing values as another branch  */

    if ( MissingValues )
    {
	Result.SubsetCount++;
	CopyBits(Workspace.Bytes, Result.Subsets[0], Result.Subsets[Result.SubsetCount]);
    }

    Verbosity(2,
	fprintf(Context->io.output, "\tfinal inf %.3f, gain %.3f, val=%.3f\n",
		Result.Information, Result.Gain, Result.Gain / (Result.Information + 1E-3)))
}



/*************************************************************************/
/*									 */
/*	Combine the distribution figures of subsets x and y.		 */
/*	Update Freq, ValFreq, SubsetInfo, SubsetEntr, MergeInfo, and	 */
/*	MergeEntr.							 */
/*									 */
/*************************************************************************/


void Merge(c50_context *Context, SplitWorkspace &Workspace, DiscrValue x, DiscrValue y,
	   CaseCount Cases)
/*   -----  */
{
    ClassNo	c;
    double	Entr=0;
    CaseCount	KnownCases=0;
    int		R, C;

    AddBlock(Context, Workspace, x, y);

    ForEach(c, 1, Context->schema.max_class)
    {
	Entr -= Workspace.Freq[x][c] * Log(Workspace.Freq[x][c]);
	KnownCases += Workspace.Freq[x][c];
    }

    Workspace.SubsetInfo[x] = - Workspace.ValFreq[x] * Log(Workspace.ValFreq[x] / Cases);
    Workspace.SubsetEntr[x] = Entr + KnownCases * Log(KnownCases);

    /*  Eliminate y from working blocks  */

    ForEach(R, y, Workspace.Blocks-1)
    {
	MoveBlock(Context, Workspace, R, R+1);

	Workspace.SubsetInfo[R] = Workspace.SubsetInfo[R+1];
	Workspace.SubsetEntr[R] = Workspace.SubsetEntr[R+1];

	ForEach(C, 1, Workspace.Blocks)
	{
	    Workspace.MergeInfo[R][C] = Workspace.MergeInfo[R+1][C];
	    Workspace.MergeEntr[R][C] = Workspace.MergeEntr[R+1][C];
	}
    }

    ForEach(C, y, Workspace.Blocks-1)
    {
	ForEach(R, 1, Workspace.Blocks-1)
	{
	    Workspace.MergeInfo[R][C] = Workspace.MergeInfo[R][C+1];
	    Workspace.MergeEntr[R][C] = Workspace.MergeEntr[R][C+1];
	}
    }
    Workspace.Blocks--;

    /*  Update information for newly-merged block  */

    ForEach(C, 1, Workspace.Blocks)
    {
	if ( C != x ) EvaluatePair(Context, Workspace, x, C, Cases);
    }
}



/*************************************************************************/
/*									 */
/*	Calculate the effect of merging subsets x and y			 */
/*									 */
/*************************************************************************/


void EvaluatePair(c50_context *Context, SplitWorkspace &Workspace, DiscrValue x, DiscrValue y,
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

    F = Workspace.ValFreq[x] + Workspace.ValFreq[y];
    Workspace.MergeInfo[x][y] = - F * Log(F / Cases);

    ForEach(c, 1, Context->schema.max_class)
    {
	F = Workspace.Freq[x][c] + Workspace.Freq[y][c];
	Entr -= F * Log(F);
	KnownCases += F;
    }
    Workspace.MergeEntr[x][y] = Entr + KnownCases * Log(KnownCases);
}



/*************************************************************************/
/*									 */
/*	Check whether two values have same class distribution		 */
/*									 */
/*************************************************************************/


Boolean SameDistribution(c50_context *Context, SplitWorkspace &Workspace, DiscrValue V1,
			 DiscrValue V2)
/*	----------------  */
{
    ClassNo	c;
    CaseCount	D1, D2;

    D1 = Workspace.ValFreq[V1];
    D2 = Workspace.ValFreq[V2];

    ForEach(c, 1, Context->schema.max_class)
    {
	if ( fabs(Workspace.Freq[V1][c] / D1 - Workspace.Freq[V2][c] / D2) > 0.001 )
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


void AddBlock(c50_context *Context, SplitWorkspace &Workspace, DiscrValue V1, DiscrValue V2)
/*   --------  */
{
    ClassNo	c;
    int		b;

    if ( Workspace.ValFreq[V1] >= Context->options.minimum_cases-Epsilon &&
	 Workspace.ValFreq[V2] >= Context->options.minimum_cases-Epsilon )
    {
	Workspace.ReasonableSubsets--;
    }
    else
    if ( Workspace.ValFreq[V1] < Context->options.minimum_cases-Epsilon &&
	 Workspace.ValFreq[V2] < Context->options.minimum_cases-Epsilon &&
	 Workspace.ValFreq[V1] + Workspace.ValFreq[V2] >= Context->options.minimum_cases-Epsilon )
    {
	Workspace.ReasonableSubsets++;
    }

    ForEach(c, 1, Context->schema.max_class)
    {
	Workspace.Freq[V1][c] += Workspace.Freq[V2][c];
    }
    Workspace.ValFreq[V1] += Workspace.ValFreq[V2];
    Workspace.ValFreq[V2] = 0;
    ForEach(b, 0, Workspace.Bytes-1)
    {
	Workspace.WSubset[V1][b] |= Workspace.WSubset[V2][b];
    }
}



/*************************************************************************/
/*									 */
/*	Move frequency and subset information from block V2 to V1	 */
/*									 */
/*************************************************************************/


void MoveBlock(c50_context *Context, SplitWorkspace &Workspace, DiscrValue V1, DiscrValue V2)
/*   ---------  */
{
    ClassNo	c;

    ForEach(c, 1, Context->schema.max_class)
    {
	Workspace.Freq[V1][c] = Workspace.Freq[V2][c];
    }
    Workspace.ValFreq[V1] = Workspace.ValFreq[V2];
    CopyBits(Workspace.Bytes, Workspace.WSubset[V2], Workspace.WSubset[V1]);
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
		fprintf(Context->io.output, ", ");
	    }

	    fprintf(Context->io.output, "%s", Context->schema.attribute_value_names[Att][V1]);
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

    Sprout(Context, Node, Context->splits.subset_counts[Att]);

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
