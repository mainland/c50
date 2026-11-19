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
/* Modifications Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Modified 2026 by Geoffrey Mainland. */



/*************************************************************************/
/*									 */
/*	Carry out crossvalidation trials				 */
/*	--------------------------------				 */
/*									 */
/*************************************************************************/

#include "defns.i"
#include "extern.i"
#include "c50_api_internal.h"

#include <cmath>
#include <vector>

namespace c50::core {


/*************************************************************************/
/*									 */
/*	Outer function (differs from xval script)			 */
/*									 */
/*************************************************************************/


void CrossVal(c50_context *Context)
/*   --------  */
{
    CaseNo	i, Size, Start=0, Next, SaveMaxCase;
    int		f, SmallTestBlocks, t, SaveTRIALS;
    ClassNo	c;
    /*  Check for left-overs after interrupt  */

    if ( Context->cross_validation.results )
    {
	FreeVector((void **) Context->cross_validation.results, 0, Context->cross_validation.saved_folds-1);
	Free(Context->cross_validation.confusion_matrix);
    }

    /*  Folds are sized by dividing by the number of cases, and at least two
	folds each need a case to hold out and a case to train on  */

    if ( Context->cases.max_case < 0 )
    {
	Error(Context, NOXVALCASES, "", "");
    }

    if ( Context->cases.max_case < 1 )
    {
	Error(Context, ONEXVALCASE, "", "");
    }

    if ( Context->options.folds > Context->cases.max_case+1 )
    {
	c50_diagnostic_printf(Context->io.output, T_FoldsReduced);
	Context->options.folds = Context->cases.max_case+1;
    }

    Context->cross_validation.results	 = AllocZero((Context->cross_validation.saved_folds = Context->options.folds), float *);
    Context->cross_validation.blocked_cases	 = Alloc(Context->cases.max_case+1, DataRec);
    Context->cross_validation.confusion_matrix = AllocZero(static_cast<size_t>(Context->schema.max_class + 1) *
	                                                      static_cast<size_t>(Context->schema.max_class + 1), CaseNo);

    Prepare(Context);

    SaveMaxCase = Context->cases.max_case;
    SaveTRIALS  = Context->options.trials;

    /*  First test blocks may be smaller than the others  */

    SmallTestBlocks = Context->options.folds - ((Context->cases.max_case+1) % Context->options.folds);
    Size = (Context->cases.max_case + 1) / Context->options.folds;

    ForEach(f, 0, Context->options.folds-1)
    {
	c50_diagnostic_printf(Context->io.output, "\n\n[ " T_Fold " %d ]\n", f+1);
	Context->cross_validation.results[f] = AllocZero(3, float);

	if ( f == SmallTestBlocks ) Size++;
	Context->cases.max_case = SaveMaxCase - Size;

	ForEach(i, 0, Context->cases.max_case)
	{
	    Context->cases.records[i] = Context->cross_validation.blocked_cases[Start];
	    Start = (Start + 1) % (SaveMaxCase + 1);
	}

	/*  A fold's cost is summed in float, as in Release 2.07, and again in
	    double for the case in which the float sum is not finite.  */

	double FoldCost = 0;

	ConstructClassifiers(Context);

	/*  Check size (if appropriate) and errors  */

	if ( Context->options.trials == 1 )
	{
	    Context->cross_validation.results[f][0] = ( Context->options.rules ? Context->rules.sets[0]->SNRules :
				     TreeSize(Context->trees.pruned[0]) );
	    Next = Start;
	    ForEach(i, 0, Size-1)
	    {
		Context->cases.records[i] = Context->cross_validation.blocked_cases[Next];
		c = ( Context->options.rules ? RuleClassify(Context, Context->cross_validation.blocked_cases[Next], Context->rules.sets[0]) :
			      TreeClassify(Context, Context->cross_validation.blocked_cases[Next], Context->trees.pruned[0]) );
		if ( c != Class(Context->cross_validation.blocked_cases[Next]) )
		{
		    Context->cross_validation.results[f][1] += 1.0;
		    if ( Context->costs.matrix )
		    {
			Context->cross_validation.results[f][2] += Context->costs.matrix[c][Class(Context->cross_validation.blocked_cases[Next])];
			FoldCost += Context->costs.matrix[c][Class(Context->cross_validation.blocked_cases[Next])];
		    }
		}

		/*  Add to confusion matrix for target classifier  */

		Context->cross_validation.confusion_matrix[ FlatIndex(Class(Context->cross_validation.blocked_cases[Next]), Context->schema.max_class+1, c) ]++;

		Next = (Next + 1) % (SaveMaxCase + 1);
	    }
	}
	else
	{
	    Context->cross_validation.results[f][0] = -1;
	    Next = Start;
	    Context->default_class =
		( Context->options.rules ? Context->rules.sets[0]->SDefault : Context->trees.pruned[0]->Leaf );
	    ForEach(i, 0, Size-1)
	    {
		Context->cases.records[i] = Context->cross_validation.blocked_cases[Next];
		c = BoostClassify(Context, Context->cross_validation.blocked_cases[Next], Context->options.trials-1);
		if ( c != Class(Context->cross_validation.blocked_cases[Next]) )
		{
		    Context->cross_validation.results[f][1] += 1.0;
		    if ( Context->costs.matrix )
		    {
			Context->cross_validation.results[f][2] += Context->costs.matrix[c][Class(Context->cross_validation.blocked_cases[Next])];
			FoldCost += Context->costs.matrix[c][Class(Context->cross_validation.blocked_cases[Next])];
		    }
		}

		/*  Add to confusion matrix for target classifier  */

		Context->cross_validation.confusion_matrix[ FlatIndex(Class(Context->cross_validation.blocked_cases[Next]), Context->schema.max_class+1, c) ]++;

		Next = (Next + 1) % (SaveMaxCase + 1);
	    }
	}

	Context->cross_validation.results[f][1] = (100.0 * Context->cross_validation.results[f][1]) / Size;
	if ( std::isfinite(Context->cross_validation.results[f][2]) )
	    Context->cross_validation.results[f][2] /= Size;
	else
	    Context->cross_validation.results[f][2] = static_cast<float>(FoldCost / Size);

	c50_diagnostic_printf(Context->io.output, T_EvalHoldOut, Size);
	Context->cases.max_case = Size-1;
	Evaluate(Context, 0);

	/*  Free space used by classifiers  */

	ForEach(t, 0, Context->trees.max_tree)
	{
	    FreeClassifier(Context, t);
	}
	Context->trees.max_tree = -1;

	Context->options.trials = SaveTRIALS;
    }

    /*  Print summary of crossvalidation  */

    Context->cases.max_case = SaveMaxCase;

    Summary(Context);
    PrintConfusionMatrix(Context, Context->cross_validation.confusion_matrix);

    /*  Free local storage  */

    ForEach(i, 0, Context->cases.max_case)
    {
	Context->cases.records[i] = Context->cross_validation.blocked_cases[i];
    }

    FreeVector((void **) Context->cross_validation.results, 0, Context->options.folds-1);		Context->cross_validation.results = Nil;
    Free(Context->cross_validation.blocked_cases);					Context->cross_validation.blocked_cases = Nil;
    Free(Context->cross_validation.confusion_matrix);					Context->cross_validation.confusion_matrix = Nil;
}



/*************************************************************************/
/*                                                                       */
/*      Prepare data for crossvalidation (similar to xval-prep.c)	 */
/*                                                                       */
/*************************************************************************/


void Prepare(c50_context *Context)
/*   -------  */
{
    CaseNo	i, First=0, Last, *Temp, Hold, Next=0;
    ClassNo	Group;

    Temp = Alloc(Context->cases.max_case+1, CaseNo);
    ForEach(i, 0, Context->cases.max_case)
    {
	Temp[i] = i;
    }

    Shuffle(Context, Temp);

    /*  Sort into class groups  */

    while ( First <= Context->cases.max_case )
    {
	Last = First;
	Group = Class(Context->cases.records[Temp[First]]);

	ForEach(i, First+1, Context->cases.max_case)
	{
	    if ( Class(Context->cases.records[Temp[i]]) == Group )
	    {
		Last++;
		Hold = Temp[Last];
		Temp[Last] = Temp[i];
		Temp[i] = Hold;
	    }
	}

	First = Last+1;
    }

    /*  Organize into stratified blocks  */

    ForEach(First, 0, Context->options.folds-1)
    {
	for ( i = First ; i <= Context->cases.max_case ; i += Context->options.folds )
	{
	    Context->cross_validation.blocked_cases[Next++] = Context->cases.records[Temp[i]];
	}
    }

    Free(Temp);
}



/*************************************************************************/
/*                                                                       */
/*      Shuffle the data cases                                           */
/*                                                                       */
/*************************************************************************/


void Shuffle(c50_context *Context, int *Vec)
/*   -------  */
{
    int	This=0, Alt, Left=Context->cases.max_case+1, Hold;

    ResetKR(&Context->random, Context->io.random_initial_seed);

    while ( Left )
    {
	Alt = This + (Left--) * KRandom(&Context->random);

	Hold 	    = Vec[This];
	Vec[This++] = Vec[Alt];
	Vec[Alt]    = Hold;
    }
}



/*************************************************************************/
/*									 */
/*	Summarise a crossvalidation					 */
/*									 */
/*************************************************************************/


/*  The mean and standard error of one column of fold results.  Release
    2.07 sums the values and their squares in float.  Each of the two numbers
    keeps that arithmetic, and so its printed text, whenever it is finite.
    The squares overflow float for a value above about 1.8e19, the sum
    overflows near the limit of float, and the difference of two nearly equal
    sums can be negative.  Release 2.07 prints each of those as infinity or
    nan.  Only a number that is not finite is recomputed in double, from the
    deviations about the mean, which neither overflow nor cancel.  The other
    number of the column is not affected.  */

struct FoldSummary
{
    double	mean;
    double	se;
};

static FoldSummary Summarise(const std::vector<float> &Values)
/*                 ---------  */
{
    const int	No = static_cast<int>(Values.size());
    float	Sum = 0, SumSq = 0;

    for ( float V : Values )
    {
	Sum += V;
	SumSq += V * V;
    }

    FoldSummary Result = { Sum / No, SE(Sum, SumSq, No) };

    if ( No < 2 || (std::isfinite(Result.mean) && std::isfinite(Result.se)) )
    {
	return Result;
    }

    double Mean = 0, Deviations = 0;

    for ( float V : Values ) Mean += V;
    Mean /= No;

    if ( ! std::isfinite(Result.mean) )
    {
	Result.mean = Mean;
    }

    if ( ! std::isfinite(Result.se) )
    {
	for ( float V : Values ) Deviations += (V - Mean) * (V - Mean);
	Result.se = sqrt(Deviations / (No - 1) / No);
    }

    return Result;
}



void Summary(c50_context *Context)
/*   -------  */
{
    int		i, f, t;
    Boolean	PrintSize=true;
    FoldSummary	Stats[3];
    const char *const FoldHead[] = { F_Fold, F_UFold, "" };
    const char *const StdP[] = { "  " F_DecisionTree16 "  ",
			    "  ----------------  ", "  " F_SizeErrors "  " };
    const char *const StdPC[] = { "  " F_DecisionTree23 "  ",
			     "  -----------------------  ",
			     "  " F_SizeErrorsCost "  " };
    const char *const Extra[] = { "  " F_Rules16, "  ----------------",
			     "  " F_NoErrors };
    const char *const ExtraC[] = { "  " F_Rules23,
			      "  -----------------------",
			      "  " F_NoErrorsCost };

    ForEach(f, 0, Context->options.folds-1)
    {
	if ( Context->cross_validation.results[f][0] < 1 ) PrintSize = false;
    }

    for ( i = 0 ; i < 3 ; i++ )
    {
	std::vector<float> Column;
	ForEach(f, 0, Context->options.folds-1)
	{
	    Column.push_back(Context->cross_validation.results[f][i]);
	}
	Stats[i] = Summarise(Column);
    }

    c50_diagnostic_printf(Context->io.output, "\n\n[ " T_Summary " ]\n\n");

    ForEach(t, 0, 2)
    {
	c50_diagnostic_printf(Context->io.output, "%s", FoldHead[t]);
	c50_diagnostic_putc('\t', Context->io.output);
	if ( Context->options.rules )
	{
	    c50_diagnostic_printf(Context->io.output, "%s", ( Context->costs.matrix ? ExtraC[t] : Extra[t] ));
	}
	else
	{
	    c50_diagnostic_printf(Context->io.output, "%s", ( Context->costs.matrix ? StdPC[t] : StdP[t] ));
	}
	c50_diagnostic_putc('\n', Context->io.output);
    }
    c50_diagnostic_putc('\n', Context->io.output);

    ForEach(f, 0, Context->options.folds-1)
    {
	c50_diagnostic_printf(Context->io.output, "%4d\t", f+1);

	if ( PrintSize )
	{
	    c50_diagnostic_printf(Context->io.output, " %5g", Context->cross_validation.results[f][0]);
	}
	else
	{
	    c50_diagnostic_printf(Context->io.output, "     *");
	}
	c50_diagnostic_printf(Context->io.output, " %10.1f%%", Context->cross_validation.results[f][1]);

	if ( Context->costs.matrix )
	{
	    c50_diagnostic_printf(Context->io.output, "%7.2f", Context->cross_validation.results[f][2]);
	}
	c50_diagnostic_printf(Context->io.output, "\n");
    }

    c50_diagnostic_printf(Context->io.output, "\n  " T_Mean "\t");

    if ( ! PrintSize )
    {
	c50_diagnostic_printf(Context->io.output, "      ");
    }
    else
    {
	c50_diagnostic_printf(Context->io.output, "%6.1f", Stats[0].mean);
    }

    c50_diagnostic_printf(Context->io.output, " %10.1f%%", Stats[1].mean);

    if ( Context->costs.matrix )
    {
	c50_diagnostic_printf(Context->io.output, "%7.2f", Stats[2].mean);
    }

    c50_diagnostic_printf(Context->io.output, "\n  " T_SE "\t");

    if ( ! PrintSize )
    {
	c50_diagnostic_printf(Context->io.output, "      ");
    }
    else
    {
	c50_diagnostic_printf(Context->io.output, "%6.1f", Stats[0].se);
    }

    c50_diagnostic_printf(Context->io.output, " %10.1f%%", Stats[1].se);

    if ( Context->costs.matrix )
    {
	c50_diagnostic_printf(Context->io.output, "%7.2f", Stats[2].se);
    }
    c50_diagnostic_printf(Context->io.output, "\n");
}



float SE(float sum, float sumsq, int no)
/*    --  */
{
    float mean;

    mean = sum / no;

    return sqrt( (double) (((sumsq - no * mean * mean) / (no - 1)) / no) );
}

}  // namespace c50::core
