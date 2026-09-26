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
/*                                                                	 */
/*	Evaluation of a test on a continuous valued attribute	  	 */
/*	-----------------------------------------------------	  	 */
/*								  	 */
/*************************************************************************/

#include "defns.i"
#include "extern.i"
#include "c50_api_internal.h"

#define	PartInfo(n) (-(n)*Log((n)/Workspace.Cases))


/*************************************************************************/
/*								  	 */
/*	Continuous attributes are treated as if they have possible	 */
/*	values 0 (unknown), 1 (not applicable), 2 (less than cut) and	 */
/*	3 (greater than cut).						 */
/*	This routine finds the best cut for cases Fp through Lp and	 */
/*	sets the per-attribute result before serial publication					 */
/*								  	 */
/*************************************************************************/


void EvalContinuousAtt(c50_context *Context, SplitWorkspace &Workspace, SplitResult &Result,
		       Attribute Att, CaseNo Fp,
		       CaseNo Lp)
/*   -----------------  */
{
    CaseNo	i, j, BestI, Tries=0;
    double	LowInfo, LHInfo, LeastInfo=1E38,
		w, BestGain, BestInfo, ThreshCost=1;
    ClassNo	c;
    ContValue	Interval;

    Verbosity(3, fprintf(Context->io.output, "\tAtt %s\n", Context->schema.attribute_names[Att]))

    Result.Gain = None;
    PrepareForContin(Context, Workspace, Att, Fp, Lp);

    /*  Special case when very few known values  */

    if ( Workspace.ApplicCases < 2 * Context->options.minimum_cases )
    {
	Verbosity(2,
	    fprintf(Context->io.output, "\tAtt %s\tinsufficient cases with known values\n",
			Context->schema.attribute_names[Att]))
	return;
    }

    /*  Try possible cuts between cases i and i+1, and determine the
	information and gain of the split in each case  */

    /*  We have to be wary of splitting a small number of cases off one end,
	as this has little predictive power.  The minimum split Workspace.MinSplit is
	the maximum of Context->options.minimum_cases or (the minimum of 25 and 10% of the cases
	per class)  */

    Workspace.MinSplit = 0.10 * Workspace.KnownCases / Context->schema.max_class;
    if ( Workspace.MinSplit > 25 ) Workspace.MinSplit = 25;
    if ( Workspace.MinSplit < Context->options.minimum_cases ) Workspace.MinSplit = Context->options.minimum_cases;

    /*	Find first possible cut point and initialise scan parameters  */

    i = PrepareForScan(Context, Workspace, Lp);

    /*  Repeatedly check next possible cut  */

    for ( ; i <= Workspace.Ep ; i++ )
    {
	c = Workspace.SRec[i].C;
	w = Workspace.SRec[i].W;
	assert(c >= 1 && c <= Context->schema.max_class);

	Workspace.LowCases   += w;
	Workspace.Freq[2][c] += w;
	Workspace.Freq[3][c] -= w;

	Workspace.HighVal = Workspace.SRec[i+1].V;
	if ( Workspace.HighVal > Workspace.LowVal )
	{
	    Tries++;

	    Workspace.LowClass  = Workspace.HighClass;
	    Workspace.HighClass = Workspace.SRec[i+1].C;
	    for ( j = i+2 ;
		  Workspace.HighClass && j <= Workspace.Ep && Workspace.SRec[j].V == Workspace.HighVal ;
		  j++ )
	    {
		if ( Workspace.SRec[j].C != Workspace.HighClass ) Workspace.HighClass = 0;
	    }

	    if ( ! Workspace.LowClass || Workspace.LowClass != Workspace.HighClass || j > Workspace.Ep )
	    {
		LowInfo = TotalInfo(Workspace.Freq[2], 1, Context->schema.max_class);

		/*  If cannot improve on best so far, count remaining
		    possible cuts and break  */

		if ( LowInfo >= LeastInfo )
		{
		    for ( i++ ; i <= Workspace.Ep ; i++ )
		    {
			if ( Workspace.SRec[i+1].V > Workspace.SRec[i].V )
			{
			    Tries++;
			}
		    }
		    break;
		}

		LHInfo = LowInfo + TotalInfo(Workspace.Freq[3], 1, Context->schema.max_class);
		if ( LHInfo < LeastInfo )
		{
		    LeastInfo = LHInfo;
		    BestI     = i;

		    BestInfo = (Workspace.FixedSplitInfo
				+ PartInfo(Workspace.LowCases)
				+ PartInfo(Workspace.ApplicCases - Workspace.LowCases))
			       / Workspace.Cases;
		}

		Verbosity(3,
		{
		    fprintf(Context->io.output, "\t\tCut at %.3f  (gain %.3f):",
			   (Workspace.LowVal + Workspace.HighVal) / 2,
			   (1 - Workspace.UnknownRate) *
			   (Workspace.BaseInfo - (Workspace.NAInfo + LHInfo) / Workspace.KnownCases));
		    PrintDistribution(Context, Att, 2, 3, Workspace.Freq, Workspace.ValFreq, true);
		})
	    }

	    Workspace.LowVal = Workspace.HighVal;
	}
    }

    BestGain = (1 - Workspace.UnknownRate) *
	       (Workspace.BaseInfo - (Workspace.NAInfo + LeastInfo) / Workspace.KnownCases);

    /*  The threshold cost is the lesser of the cost of indicating the
	cases to split between or the interval containing the split  */

    if ( BestGain > 0 )
    {
	Interval = (Workspace.SRec[Lp].V - Workspace.SRec[Workspace.Xp].V) /
		   (Workspace.SRec[BestI+1].V - Workspace.SRec[BestI].V);
	ThreshCost = ( Interval < Tries ? Log(Interval) : Log(Tries) )
		     / Workspace.Cases;
    }

    BestGain -= ThreshCost;

    /*  If a test on the attribute is able to make a gain,
	set the best break point, gain and information  */

    if ( BestGain <= 0 )
    {
	Verbosity(2, fprintf(Context->io.output, "\tAtt %s\tno gain\n", Context->schema.attribute_names[Att]))
    }
    else
    {
	Result.Gain = BestGain;
	Result.Information = BestInfo;

	Workspace.LowVal  = Workspace.SRec[BestI].V;
	Workspace.HighVal = Workspace.SRec[BestI+1].V;

	/*  Set threshold, making sure that rounding problems do not
	    cause it to reach upper value  */

	if ( (Result.Threshold = (ContValue) (0.5 * (Workspace.LowVal + Workspace.HighVal)))
	     >= Workspace.HighVal )
	{
	    Result.Threshold = Workspace.LowVal;
	}

	Verbosity(2,
	    fprintf(Context->io.output, "\tAtt %s\tcut=%.3f, inf %.3f, gain %.3f\n",
		   Context->schema.attribute_names[Att], Result.Threshold, Result.Information, Result.Gain))
    }
}



/*************************************************************************/
/*                                                                	 */
/*	Estimate max gain ratio available from any cut, using sample	 */
/*	of Context->splits.sample_fraction of all cases					 */
/*                                                                	 */
/*************************************************************************/


void EstimateMaxGR(c50_context *Context, SplitWorkspace &Workspace, SplitResult &Result,
		   Attribute Att, CaseNo Fp, CaseNo Lp)
/*   -------------  */
{
    CaseNo	i, j;
    double	LHInfo, w, SplitInfo, ThisGain, GR;
    ClassNo	c;

    Result.EstimatedMaxGR = 0;

    if ( Skip(Att) || Att == Context->schema.class_attribute ) return;

    PrepareForContin(Context, Workspace, Att, Fp, Lp);

    /*  Special case when very few known values  */

    if ( Workspace.ApplicCases < 2 * Context->options.minimum_cases * Context->splits.sample_fraction )
    {
	return;
    }

    /*  Try possible cuts between cases i and i+1.  Use conservative
	value of Workspace.MinSplit to allow for sampling  */

    Workspace.MinSplit = 0.10 * Workspace.KnownCases / Context->schema.max_class;
    if ( Workspace.MinSplit > 25 ) Workspace.MinSplit = 25;
    if ( Workspace.MinSplit < Context->options.minimum_cases ) Workspace.MinSplit = Context->options.minimum_cases;

    Workspace.MinSplit *= Context->splits.sample_fraction * 0.33;

    i = PrepareForScan(Context, Workspace, Lp);

    /*  Repeatedly check next possible cut  */

    for ( ; i <= Workspace.Ep ; i++ )
    {
	c = Workspace.SRec[i].C;
	w = Workspace.SRec[i].W;
	assert(c >= 1 && c <= Context->schema.max_class);

	Workspace.LowCases   += w;
	Workspace.Freq[2][c] += w;
	Workspace.Freq[3][c] -= w;

	Workspace.HighVal = Workspace.SRec[i+1].V;
	if ( Workspace.HighVal > Workspace.LowVal )
	{
	    Workspace.LowClass  = Workspace.HighClass;
	    Workspace.HighClass = Workspace.SRec[i+1].C;
	    for ( j = i+2 ;
		  Workspace.HighClass && j <= Workspace.Ep && Workspace.SRec[j].V == Workspace.HighVal ;
		  j++ )
	    {
		if ( Workspace.SRec[j].C != Workspace.HighClass ) Workspace.HighClass = 0;
	    }

	    if ( ! Workspace.LowClass || Workspace.LowClass != Workspace.HighClass || j > Workspace.Ep )
	    {
		LHInfo = TotalInfo(Workspace.Freq[2], 1, Context->schema.max_class)
			 + TotalInfo(Workspace.Freq[3], 1, Context->schema.max_class);

		SplitInfo = (Workspace.FixedSplitInfo
			    + PartInfo(Workspace.LowCases)
			    + PartInfo(Workspace.ApplicCases - Workspace.LowCases)) / Workspace.Cases;

		ThisGain = (1 - Workspace.UnknownRate) *
			   (Workspace.BaseInfo - (Workspace.NAInfo + LHInfo) / Workspace.KnownCases);
		if ( ThisGain > Result.Gain ) Result.Gain = ThisGain;

		/*  Adjust GR to make it more conservative upper bound  */

		GR = (ThisGain + 1E-5) / SplitInfo;
		if ( GR > Result.EstimatedMaxGR )
		{
		    Result.EstimatedMaxGR = GR;
		}

		Verbosity(3,
		{
		    fprintf(Context->io.output, "\t\tCut at %.3f  (gain %.3f):",
			   (Workspace.LowVal + Workspace.HighVal) / 2, ThisGain);
		    PrintDistribution(Context, Att, 2, 3, Workspace.Freq, Workspace.ValFreq, true);
		})
	    }

	    Workspace.LowVal = Workspace.HighVal;
	}
    }

    Verbosity(2,
	fprintf(Context->io.output, "\tAtt %s: max GR estimate %.3f\n",
		    Context->schema.attribute_names[Att], Result.EstimatedMaxGR))
}



/*************************************************************************/
/*								  	 */
/*	Routine to set some preparatory values used by both		 */
/*	EvalContinuousAtt and EstimateMaxGR				 */
/*								  	 */
/*************************************************************************/


void PrepareForContin(c50_context *Context, SplitWorkspace &Workspace,
		      Attribute Att, CaseNo Fp,
		      CaseNo Lp)
/*   ----------------  */
{
    CaseNo	i;
    ClassNo	c;
    DiscrValue	v;

    /*  Reset frequency tables  */

    ForEach(v, 0, 3)
    {
	ForEach(c, 1, Context->schema.max_class)
	{
	    Workspace.Freq[v][c] = 0;
	}
	Workspace.ValFreq[v] = 0;
    }

    /*  Omit and count unknown and N/A values */

    Workspace.Cases = 0;

    if ( Context->cases.some_missing[Att] || Context->cases.some_not_applicable[Att] )
    {
	Workspace.Xp = Lp+1;

	ForEach(i, Fp, Lp)
	{
	    assert(Class(Context->cases.records[i]) >= 1 && Class(Context->cases.records[i]) <= Context->schema.max_class);

	    Workspace.Cases += Weight(Context->cases.records[i]);

	    if ( Unknown(Context->cases.records[i], Att) )
	    {
		Workspace.Freq[ 0 ][ Class(Context->cases.records[i]) ] += Weight(Context->cases.records[i]);
	    }
	    else
	    if ( NotApplic(Context, Context->cases.records[i], Att) )
	    {
		Workspace.Freq[ 1 ][ Class(Context->cases.records[i]) ] += Weight(Context->cases.records[i]);
	    }
	    else
	    {
		Workspace.Freq[ 3 ][ Class(Context->cases.records[i]) ] += Weight(Context->cases.records[i]);
		Workspace.Xp--;
		Workspace.SRec[Workspace.Xp].V = CVal(Context->cases.records[i], Att);
		Workspace.SRec[Workspace.Xp].W = Weight(Context->cases.records[i]);
		Workspace.SRec[Workspace.Xp].C = Class(Context->cases.records[i]);
	    }
	}

	ForEach(c, 1, Context->schema.max_class)
	{
	    Workspace.ValFreq[0] += Workspace.Freq[0][c];
	    Workspace.ValFreq[1] += Workspace.Freq[1][c];
	}

	Workspace.NAInfo = TotalInfo(Workspace.Freq[1], 1, Context->schema.max_class);
	Workspace.FixedSplitInfo = PartInfo(Workspace.ValFreq[0]) + PartInfo(Workspace.ValFreq[1]);

	Verbosity(3, PrintDistribution(Context, Att, 0, 1, Workspace.Freq, Workspace.ValFreq, true))
    }
    else
    {
	Workspace.Xp = Fp;

	ForEach(i, Fp, Lp)
	{
	    Workspace.SRec[i].V = CVal(Context->cases.records[i], Att);
	    Workspace.SRec[i].W = Weight(Context->cases.records[i]);
	    Workspace.SRec[i].C = Class(Context->cases.records[i]);

	    Workspace.Freq[3][Class(Context->cases.records[i])] += Weight(Context->cases.records[i]);
	}

	ForEach(c, 1, Context->schema.max_class)
	{
	    Workspace.Cases += Workspace.Freq[3][c];
	}

	Workspace.NAInfo = Workspace.FixedSplitInfo = 0;
    }

    Workspace.KnownCases  = Workspace.Cases - Workspace.ValFreq[0];
    Workspace.ApplicCases = Workspace.KnownCases - Workspace.ValFreq[1];

    Workspace.UnknownRate = 1.0 - Workspace.KnownCases / Workspace.Cases;

    Cachesort(Workspace.Xp, Lp, Workspace.SRec);

    /*  If unknowns or using sampling, must recompute base information  */

    if ( Workspace.ValFreq[0] > 0 || Context->splits.sample_fraction < 1 )
    {
	/*  Determine base information using Workspace.Freq[0] as temp buffer  */

	ForEach(c, 1, Context->schema.max_class)
	{
	    Workspace.Freq[0][c] = Workspace.Freq[1][c] + Workspace.Freq[3][c];
	}

	Workspace.BaseInfo = TotalInfo(Workspace.Freq[0], 1, Context->schema.max_class) / Workspace.KnownCases;
    }
    else
    {
	Workspace.BaseInfo = Context->splits.base_information;
    }
}



/*************************************************************************/
/*								  	 */
/*	Set low and high bounds for scan and initial class		 */
/*	(used by EvalContinuousAtt and EstimateMaxGR)			 */
/*								  	 */
/*************************************************************************/


CaseNo PrepareForScan(c50_context *Context, SplitWorkspace &Workspace, CaseNo Lp)
/*     --------------  */
{
    CaseNo	i, j;
    ClassNo	c;
    double	w;

    /*  Find last possible split  */

    Workspace.HighCases = Workspace.LowCases = 0;

    for ( Workspace.Ep = Lp ; Workspace.Ep >= Workspace.Xp && Workspace.HighCases < Workspace.MinSplit ; Workspace.Ep-- )
    {
	Workspace.HighCases += Workspace.SRec[Workspace.Ep].W;
    }

    /*  Skip cases before first possible cut  */

    for ( i = Workspace.Xp ;
	  i <= Workspace.Ep &&
	  ( Workspace.LowCases + Workspace.SRec[i].W < Workspace.MinSplit - 1E-5 ||
	    Workspace.SRec[i].V == Workspace.SRec[i+1].V ) ;
	  i++ )
    {
	c = Workspace.SRec[i].C;
	w = Workspace.SRec[i].W;
	assert(c >= 1 && c <= Context->schema.max_class);

	Workspace.LowCases   += w;
	Workspace.Freq[2][c] += w;
	Workspace.Freq[3][c] -= w;
    }

    /*  Find the class key for the first interval  */

    Workspace.HighClass = Workspace.SRec[i].C;
    for ( j = i-1; Workspace.HighClass && j >= Workspace.Xp ; j-- )
    {
	if ( Workspace.SRec[j].C != Workspace.HighClass ) Workspace.HighClass = 0;
    }
    assert(Workspace.HighClass <= Context->schema.max_class);
    assert(j+1 >= Workspace.Xp);

    Workspace.LowVal = Workspace.SRec[i].V;

    return i;
}



/*************************************************************************/
/*                                                                	 */
/*	Change a leaf into a test on a continuous attribute           	 */
/*                                                                	 */
/*************************************************************************/


void ContinTest(c50_context *Context, Tree Node, Attribute Att)
/*   ----------  */
{
    Sprout(Context, Node, 3);

    Node->NodeType = BrThresh;
    Node->Tested   = Att;
    Node->Cut 	   =
    Node->Lower	   =
    Node->Upper    = Context->splits.thresholds[Att];
}



/*************************************************************************/
/*                                                                	 */
/*	Adjust thresholds of all continuous attributes so that cuts	 */
/*	are values that appear in the data				 */
/*                                                                	 */
/*************************************************************************/


void AdjustAllThresholds(c50_context *Context, Tree T)
/*   -------------------  */
{
    Attribute	Att;
    CaseNo	Ep;

    ForEach(Att, 1, Context->schema.max_attribute)
    {
	if ( Continuous(Att) )
	{
	    Ep = -1;
	    AdjustThresholds(Context, T, Att, &Ep);
	}
    }
}



void AdjustThresholds(c50_context *Context, Tree T, Attribute Att, CaseNo *Ep)
/*   ----------------  */
{
    DiscrValue	v;
    CaseNo	i;

    if ( T->NodeType == BrThresh && T->Tested == Att )
    {
	if ( *Ep == -1 )
	{
	    ForEach(i, 0, Context->cases.max_case)
	    {
		if ( ! Unknown(Context->cases.records[i], Att) && ! NotApplic(Context, Context->cases.records[i], Att) )
		{
		    Context->training.environment->SRec[++(*Ep)].V = CVal(Context->cases.records[i], Att);
		}
	    }
	    Cachesort(0, *Ep, Context->training.environment->SRec);

	    if ( Context->splits.possible_cuts && Context->trees.trial == 0 )
	    {
		int Cuts=0;

		ForEach(i, 1, *Ep)
		{
		    if ( Context->training.environment->SRec[i].V != Context->training.environment->SRec[i-1].V ) Cuts++;
		}
		Context->splits.possible_cuts[Att] = Cuts;
	    }
	}

	T->Cut = T->Lower = T->Upper =
	    GreatestValueBelow(Context, T->Cut, Ep);
    }

    if ( T->NodeType )
    {
	ForEach(v, 1, T->Forks)
	{
	    AdjustThresholds(Context, T->Branch[v], Att, Ep);
	}
    }
}



/*************************************************************************/
/*                                                                	 */
/*	Return the greatest value of attribute Att below threshold Th  	 */
/*	(Assumes values of Att have been sorted.)			 */
/*                                                                	 */
/*************************************************************************/


ContValue GreatestValueBelow(c50_context *Context, ContValue Th, CaseNo *Ep)
/*	  ------------------  */
{
    CaseNo	Low, Mid, High;

    Low  = 0;
    High = *Ep;

    while ( Low < High )
    {
	Mid = (Low + High + 1) / 2;

	if ( Context->training.environment->SRec[Mid].V > Th )
	{
	    High = Mid - 1;
	}
	else
	{
	    Low = Mid;
	}
    }

    return Context->training.environment->SRec[Low].V;
}
