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
/*                                                                	 */
/*	Evaluation of a test on a continuous valued attribute	  	 */
/*	-----------------------------------------------------	  	 */
/*								  	 */
/*************************************************************************/

#include "defns.i"
#include "extern.i"
#include "c50_api_internal.h"

#define	PartInfo(n) (-(n)*Log((n)/Context->training.environment->Cases))


/*************************************************************************/
/*								  	 */
/*	Continuous attributes are treated as if they have possible	 */
/*	values 0 (unknown), 1 (not applicable), 2 (less than cut) and	 */
/*	3 (greater than cut).						 */
/*	This routine finds the best cut for cases Fp through Lp and	 */
/*	sets Context->splits.information[], Context->splits.gain[] and Context->splits.thresholds[]					 */
/*								  	 */
/*************************************************************************/


void EvalContinuousAtt(c50_context *Context, Attribute Att, CaseNo Fp,
		       CaseNo Lp)
/*   -----------------  */
{
    CaseNo	i, j, BestI, Tries=0;
    double	LowInfo, LHInfo, LeastInfo=1E38,
		w, BestGain, BestInfo, ThreshCost=1;
    ClassNo	c;
    ContValue	Interval;

    Verbosity(3, fprintf(Context->io.output, "\tAtt %s\n", Context->schema.attribute_names[Att]))

    Context->splits.gain[Att] = None;
    PrepareForContin(Context, Att, Fp, Lp);

    /*  Special case when very few known values  */

    if ( Context->training.environment->ApplicCases < 2 * Context->options.minimum_cases )
    {
	Verbosity(2,
	    fprintf(Context->io.output, "\tAtt %s\tinsufficient cases with known values\n",
			Context->schema.attribute_names[Att]))
	return;
    }

    /*  Try possible cuts between cases i and i+1, and determine the
	information and gain of the split in each case  */

    /*  We have to be wary of splitting a small number of cases off one end,
	as this has little predictive power.  The minimum split Context->training.environment->MinSplit is
	the maximum of Context->options.minimum_cases or (the minimum of 25 and 10% of the cases
	per class)  */

    Context->training.environment->MinSplit = 0.10 * Context->training.environment->KnownCases / Context->schema.max_class;
    if ( Context->training.environment->MinSplit > 25 ) Context->training.environment->MinSplit = 25;
    if ( Context->training.environment->MinSplit < Context->options.minimum_cases ) Context->training.environment->MinSplit = Context->options.minimum_cases;

    /*	Find first possible cut point and initialise scan parameters  */

    i = PrepareForScan(Context, Lp);

    /*  Repeatedly check next possible cut  */

    for ( ; i <= Context->training.environment->Ep ; i++ )
    {
	c = Context->training.environment->SRec[i].C;
	w = Context->training.environment->SRec[i].W;
	assert(c >= 1 && c <= Context->schema.max_class);

	Context->training.environment->LowCases   += w;
	Context->training.environment->Freq[2][c] += w;
	Context->training.environment->Freq[3][c] -= w;

	Context->training.environment->HighVal = Context->training.environment->SRec[i+1].V;
	if ( Context->training.environment->HighVal > Context->training.environment->LowVal )
	{
	    Tries++;

	    Context->training.environment->LowClass  = Context->training.environment->HighClass;
	    Context->training.environment->HighClass = Context->training.environment->SRec[i+1].C;
	    for ( j = i+2 ;
		  Context->training.environment->HighClass && j <= Context->training.environment->Ep && Context->training.environment->SRec[j].V == Context->training.environment->HighVal ;
		  j++ )
	    {
		if ( Context->training.environment->SRec[j].C != Context->training.environment->HighClass ) Context->training.environment->HighClass = 0;
	    }

	    if ( ! Context->training.environment->LowClass || Context->training.environment->LowClass != Context->training.environment->HighClass || j > Context->training.environment->Ep )
	    {
		LowInfo = TotalInfo(Context->training.environment->Freq[2], 1, Context->schema.max_class);

		/*  If cannot improve on best so far, count remaining
		    possible cuts and break  */

		if ( LowInfo >= LeastInfo )
		{
		    for ( i++ ; i <= Context->training.environment->Ep ; i++ )
		    {
			if ( Context->training.environment->SRec[i+1].V > Context->training.environment->SRec[i].V )
			{
			    Tries++;
			}
		    }
		    break;
		}

		LHInfo = LowInfo + TotalInfo(Context->training.environment->Freq[3], 1, Context->schema.max_class);
		if ( LHInfo < LeastInfo )
		{
		    LeastInfo = LHInfo;
		    BestI     = i;

		    BestInfo = (Context->training.environment->FixedSplitInfo
				+ PartInfo(Context->training.environment->LowCases)
				+ PartInfo(Context->training.environment->ApplicCases - Context->training.environment->LowCases))
			       / Context->training.environment->Cases;
		}

		Verbosity(3,
		{
		    fprintf(Context->io.output, "\t\tCut at %.3f  (gain %.3f):",
			   (Context->training.environment->LowVal + Context->training.environment->HighVal) / 2,
			   (1 - Context->training.environment->UnknownRate) *
			   (Context->training.environment->BaseInfo - (Context->training.environment->NAInfo + LHInfo) / Context->training.environment->KnownCases));
		    PrintDistribution(Context, Att, 2, 3, Context->training.environment->Freq, Context->training.environment->ValFreq, true);
		})
	    }

	    Context->training.environment->LowVal = Context->training.environment->HighVal;
	}
    }

    BestGain = (1 - Context->training.environment->UnknownRate) *
	       (Context->training.environment->BaseInfo - (Context->training.environment->NAInfo + LeastInfo) / Context->training.environment->KnownCases);

    /*  The threshold cost is the lesser of the cost of indicating the
	cases to split between or the interval containing the split  */

    if ( BestGain > 0 )
    {
	Interval = (Context->training.environment->SRec[Lp].V - Context->training.environment->SRec[Context->training.environment->Xp].V) /
		   (Context->training.environment->SRec[BestI+1].V - Context->training.environment->SRec[BestI].V);
	ThreshCost = ( Interval < Tries ? Log(Interval) : Log(Tries) )
		     / Context->training.environment->Cases;
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
	Context->splits.gain[Att] = BestGain;
	Context->splits.information[Att] = BestInfo;

	Context->training.environment->LowVal  = Context->training.environment->SRec[BestI].V;
	Context->training.environment->HighVal = Context->training.environment->SRec[BestI+1].V;

	/*  Set threshold, making sure that rounding problems do not
	    cause it to reach upper value  */

	if ( (Context->splits.thresholds[Att] = (ContValue) (0.5 * (Context->training.environment->LowVal + Context->training.environment->HighVal)))
	     >= Context->training.environment->HighVal )
	{
	    Context->splits.thresholds[Att] = Context->training.environment->LowVal;
	}

	Verbosity(2,
	    fprintf(Context->io.output, "\tAtt %s\tcut=%.3f, inf %.3f, gain %.3f\n",
		   Context->schema.attribute_names[Att], Context->splits.thresholds[Att], Context->splits.information[Att], Context->splits.gain[Att]))
    }
}



/*************************************************************************/
/*                                                                	 */
/*	Estimate max gain ratio available from any cut, using sample	 */
/*	of Context->splits.sample_fraction of all cases					 */
/*                                                                	 */
/*************************************************************************/


void EstimateMaxGR(c50_context *Context, Attribute Att, CaseNo Fp, CaseNo Lp)
/*   -------------  */
{
    CaseNo	i, j;
    double	LHInfo, w, SplitInfo, ThisGain, GR;
    ClassNo	c;

    Context->splits.estimated_max_gain_ratio[Att] = 0;

    if ( Skip(Att) || Att == Context->schema.class_attribute ) return;

    PrepareForContin(Context, Att, Fp, Lp);

    /*  Special case when very few known values  */

    if ( Context->training.environment->ApplicCases < 2 * Context->options.minimum_cases * Context->splits.sample_fraction )
    {
	return;
    }

    /*  Try possible cuts between cases i and i+1.  Use conservative
	value of Context->training.environment->MinSplit to allow for sampling  */

    Context->training.environment->MinSplit = 0.10 * Context->training.environment->KnownCases / Context->schema.max_class;
    if ( Context->training.environment->MinSplit > 25 ) Context->training.environment->MinSplit = 25;
    if ( Context->training.environment->MinSplit < Context->options.minimum_cases ) Context->training.environment->MinSplit = Context->options.minimum_cases;

    Context->training.environment->MinSplit *= Context->splits.sample_fraction * 0.33;

    i = PrepareForScan(Context, Lp);

    /*  Repeatedly check next possible cut  */

    for ( ; i <= Context->training.environment->Ep ; i++ )
    {
	c = Context->training.environment->SRec[i].C;
	w = Context->training.environment->SRec[i].W;
	assert(c >= 1 && c <= Context->schema.max_class);

	Context->training.environment->LowCases   += w;
	Context->training.environment->Freq[2][c] += w;
	Context->training.environment->Freq[3][c] -= w;

	Context->training.environment->HighVal = Context->training.environment->SRec[i+1].V;
	if ( Context->training.environment->HighVal > Context->training.environment->LowVal )
	{
	    Context->training.environment->LowClass  = Context->training.environment->HighClass;
	    Context->training.environment->HighClass = Context->training.environment->SRec[i+1].C;
	    for ( j = i+2 ;
		  Context->training.environment->HighClass && j <= Context->training.environment->Ep && Context->training.environment->SRec[j].V == Context->training.environment->HighVal ;
		  j++ )
	    {
		if ( Context->training.environment->SRec[j].C != Context->training.environment->HighClass ) Context->training.environment->HighClass = 0;
	    }

	    if ( ! Context->training.environment->LowClass || Context->training.environment->LowClass != Context->training.environment->HighClass || j > Context->training.environment->Ep )
	    {
		LHInfo = TotalInfo(Context->training.environment->Freq[2], 1, Context->schema.max_class)
			 + TotalInfo(Context->training.environment->Freq[3], 1, Context->schema.max_class);

		SplitInfo = (Context->training.environment->FixedSplitInfo
			    + PartInfo(Context->training.environment->LowCases)
			    + PartInfo(Context->training.environment->ApplicCases - Context->training.environment->LowCases)) / Context->training.environment->Cases;

		ThisGain = (1 - Context->training.environment->UnknownRate) *
			   (Context->training.environment->BaseInfo - (Context->training.environment->NAInfo + LHInfo) / Context->training.environment->KnownCases);
		if ( ThisGain > Context->splits.gain[Att] ) Context->splits.gain[Att] = ThisGain;

		/*  Adjust GR to make it more conservative upper bound  */

		GR = (ThisGain + 1E-5) / SplitInfo;
		if ( GR > Context->splits.estimated_max_gain_ratio[Att] )
		{
		    Context->splits.estimated_max_gain_ratio[Att] = GR;
		}

		Verbosity(3,
		{
		    fprintf(Context->io.output, "\t\tCut at %.3f  (gain %.3f):",
			   (Context->training.environment->LowVal + Context->training.environment->HighVal) / 2, ThisGain);
		    PrintDistribution(Context, Att, 2, 3, Context->training.environment->Freq, Context->training.environment->ValFreq, true);
		})
	    }

	    Context->training.environment->LowVal = Context->training.environment->HighVal;
	}
    }

    Verbosity(2,
	fprintf(Context->io.output, "\tAtt %s: max GR estimate %.3f\n",
		    Context->schema.attribute_names[Att], Context->splits.estimated_max_gain_ratio[Att]))
}



/*************************************************************************/
/*								  	 */
/*	Routine to set some preparatory values used by both		 */
/*	EvalContinuousAtt and EstimateMaxGR				 */
/*								  	 */
/*************************************************************************/


void PrepareForContin(c50_context *Context, Attribute Att, CaseNo Fp,
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
	    Context->training.environment->Freq[v][c] = 0;
	}
	Context->training.environment->ValFreq[v] = 0;
    }

    /*  Omit and count unknown and N/A values */

    Context->training.environment->Cases = 0;

    if ( Context->cases.some_missing[Att] || Context->cases.some_not_applicable[Att] )
    {
	Context->training.environment->Xp = Lp+1;

	ForEach(i, Fp, Lp)
	{
	    assert(Class(Context->cases.records[i]) >= 1 && Class(Context->cases.records[i]) <= Context->schema.max_class);

	    Context->training.environment->Cases += Weight(Context->cases.records[i]);

	    if ( Unknown(Context->cases.records[i], Att) )
	    {
		Context->training.environment->Freq[ 0 ][ Class(Context->cases.records[i]) ] += Weight(Context->cases.records[i]);
	    }
	    else
	    if ( NotApplic(Context, Context->cases.records[i], Att) )
	    {
		Context->training.environment->Freq[ 1 ][ Class(Context->cases.records[i]) ] += Weight(Context->cases.records[i]);
	    }
	    else
	    {
		Context->training.environment->Freq[ 3 ][ Class(Context->cases.records[i]) ] += Weight(Context->cases.records[i]);
		Context->training.environment->Xp--;
		Context->training.environment->SRec[Context->training.environment->Xp].V = CVal(Context->cases.records[i], Att);
		Context->training.environment->SRec[Context->training.environment->Xp].W = Weight(Context->cases.records[i]);
		Context->training.environment->SRec[Context->training.environment->Xp].C = Class(Context->cases.records[i]);
	    }
	}

	ForEach(c, 1, Context->schema.max_class)
	{
	    Context->training.environment->ValFreq[0] += Context->training.environment->Freq[0][c];
	    Context->training.environment->ValFreq[1] += Context->training.environment->Freq[1][c];
	}

	Context->training.environment->NAInfo = TotalInfo(Context->training.environment->Freq[1], 1, Context->schema.max_class);
	Context->training.environment->FixedSplitInfo = PartInfo(Context->training.environment->ValFreq[0]) + PartInfo(Context->training.environment->ValFreq[1]);

	Verbosity(3, PrintDistribution(Context, Att, 0, 1, Context->training.environment->Freq, Context->training.environment->ValFreq, true))
    }
    else
    {
	Context->training.environment->Xp = Fp;

	ForEach(i, Fp, Lp)
	{
	    Context->training.environment->SRec[i].V = CVal(Context->cases.records[i], Att);
	    Context->training.environment->SRec[i].W = Weight(Context->cases.records[i]);
	    Context->training.environment->SRec[i].C = Class(Context->cases.records[i]);

	    Context->training.environment->Freq[3][Class(Context->cases.records[i])] += Weight(Context->cases.records[i]);
	}

	ForEach(c, 1, Context->schema.max_class)
	{
	    Context->training.environment->Cases += Context->training.environment->Freq[3][c];
	}

	Context->training.environment->NAInfo = Context->training.environment->FixedSplitInfo = 0;
    }

    Context->training.environment->KnownCases  = Context->training.environment->Cases - Context->training.environment->ValFreq[0];
    Context->training.environment->ApplicCases = Context->training.environment->KnownCases - Context->training.environment->ValFreq[1];

    Context->training.environment->UnknownRate = 1.0 - Context->training.environment->KnownCases / Context->training.environment->Cases;

    Cachesort(Context->training.environment->Xp, Lp, Context->training.environment->SRec);

    /*  If unknowns or using sampling, must recompute base information  */

    if ( Context->training.environment->ValFreq[0] > 0 || Context->splits.sample_fraction < 1 )
    {
	/*  Determine base information using Context->training.environment->Freq[0] as temp buffer  */

	ForEach(c, 1, Context->schema.max_class)
	{
	    Context->training.environment->Freq[0][c] = Context->training.environment->Freq[1][c] + Context->training.environment->Freq[3][c];
	}

	Context->training.environment->BaseInfo = TotalInfo(Context->training.environment->Freq[0], 1, Context->schema.max_class) / Context->training.environment->KnownCases;
    }
    else
    {
	Context->training.environment->BaseInfo = Context->splits.base_information;
    }
}



/*************************************************************************/
/*								  	 */
/*	Set low and high bounds for scan and initial class		 */
/*	(used by EvalContinuousAtt and EstimateMaxGR)			 */
/*								  	 */
/*************************************************************************/


CaseNo PrepareForScan(c50_context *Context, CaseNo Lp)
/*     --------------  */
{
    CaseNo	i, j;
    ClassNo	c;
    double	w;

    /*  Find last possible split  */

    Context->training.environment->HighCases = Context->training.environment->LowCases = 0;

    for ( Context->training.environment->Ep = Lp ; Context->training.environment->Ep >= Context->training.environment->Xp && Context->training.environment->HighCases < Context->training.environment->MinSplit ; Context->training.environment->Ep-- )
    {
	Context->training.environment->HighCases += Context->training.environment->SRec[Context->training.environment->Ep].W;
    }

    /*  Skip cases before first possible cut  */

    for ( i = Context->training.environment->Xp ;
	  i <= Context->training.environment->Ep &&
	  ( Context->training.environment->LowCases + Context->training.environment->SRec[i].W < Context->training.environment->MinSplit - 1E-5 ||
	    Context->training.environment->SRec[i].V == Context->training.environment->SRec[i+1].V ) ;
	  i++ )
    {
	c = Context->training.environment->SRec[i].C;
	w = Context->training.environment->SRec[i].W;
	assert(c >= 1 && c <= Context->schema.max_class);

	Context->training.environment->LowCases   += w;
	Context->training.environment->Freq[2][c] += w;
	Context->training.environment->Freq[3][c] -= w;
    }

    /*  Find the class key for the first interval  */

    Context->training.environment->HighClass = Context->training.environment->SRec[i].C;
    for ( j = i-1; Context->training.environment->HighClass && j >= Context->training.environment->Xp ; j-- )
    {
	if ( Context->training.environment->SRec[j].C != Context->training.environment->HighClass ) Context->training.environment->HighClass = 0;
    }
    assert(Context->training.environment->HighClass <= Context->schema.max_class);
    assert(j+1 >= Context->training.environment->Xp);

    Context->training.environment->LowVal = Context->training.environment->SRec[i].V;

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
