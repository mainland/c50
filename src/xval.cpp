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
/*	Carry out crossvalidation trials				 */
/*	--------------------------------				 */
/*									 */
/*************************************************************************/

#include "defns.i"
#include "extern.i"
#include "c50_api_internal.h"


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

    if ( Context->options.folds > Context->cases.max_case+1 )
    {
	fprintf(Of, T_FoldsReduced);
	Context->options.folds = Context->cases.max_case+1;
    }

    Context->cross_validation.results	 = AllocZero((Context->cross_validation.saved_folds = Context->options.folds), float *);
    Context->cross_validation.blocked_cases	 = Alloc(Context->cases.max_case+1, DataRec);
    Context->cross_validation.confusion_matrix = AllocZero((Context->schema.max_class+1)*(Context->schema.max_class+1), CaseNo);

    Prepare(Context);

    SaveMaxCase = Context->cases.max_case;
    SaveTRIALS  = Context->options.trials;

    /*  First test blocks may be smaller than the others  */

    SmallTestBlocks = Context->options.folds - ((Context->cases.max_case+1) % Context->options.folds);
    Size = (Context->cases.max_case + 1) / Context->options.folds;

    ForEach(f, 0, Context->options.folds-1)
    {
	fprintf(Of, "\n\n[ " T_Fold " %d ]\n", f+1);
	Context->cross_validation.results[f] = AllocZero(3, float);

	if ( f == SmallTestBlocks ) Size++;
	Context->cases.max_case = SaveMaxCase - Size;

	ForEach(i, 0, Context->cases.max_case)
	{
	    Context->cases.records[i] = Context->cross_validation.blocked_cases[Start];
	    Start = (Start + 1) % (SaveMaxCase + 1);
	}

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
		    }
		}

		/*  Add to confusion matrix for target classifier  */

		Context->cross_validation.confusion_matrix[ Class(Context->cross_validation.blocked_cases[Next])*(Context->schema.max_class+1)+c ]++;

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
		    }
		}

		/*  Add to confusion matrix for target classifier  */

		Context->cross_validation.confusion_matrix[ Class(Context->cross_validation.blocked_cases[Next])*(Context->schema.max_class+1)+c ]++;

		Next = (Next + 1) % (SaveMaxCase + 1);
	    }
	}

	Context->cross_validation.results[f][1] = (100.0 * Context->cross_validation.results[f][1]) / Size;
	Context->cross_validation.results[f][2] /= Size;

	fprintf(Of, T_EvalHoldOut, Size);
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

    ResetKR(&Context->random, KRInit);

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


void Summary(c50_context *Context)
/*   -------  */
{
    int		i, f, t;
    Boolean	PrintSize=true;
    float	Sum[3], SumSq[3];
    const char	*FoldHead[] = { F_Fold, F_UFold, "" };
    extern char	*StdP[], *StdPC[], *Extra[], *ExtraC[];

    for ( i = 0 ; i < 3 ; i++ )
    {
	Sum[i] = SumSq[i] = 0;
    }

    ForEach(f, 0, Context->options.folds-1)
    {
	if ( Context->cross_validation.results[f][0] < 1 ) PrintSize = false;
    }

    fprintf(Of, "\n\n[ " T_Summary " ]\n\n");

    ForEach(t, 0, 2)
    {
	fprintf(Of, "%s", FoldHead[t]);
	putc('\t', Of);
	if ( Context->options.rules )
	{
	    fprintf(Of, "%s", ( Context->costs.matrix ? ExtraC[t] : Extra[t] ));
	}
	else
	{
	    fprintf(Of, "%s", ( Context->costs.matrix ? StdPC[t] : StdP[t] ));
	}
	putc('\n', Of);
    }
    putc('\n', Of);

    ForEach(f, 0, Context->options.folds-1)
    {
	fprintf(Of, "%4d\t", f+1);

	if ( PrintSize )
	{
	    fprintf(Of, " %5g", Context->cross_validation.results[f][0]);
	}
	else
	{
	    fprintf(Of, "     *");
	}
	fprintf(Of, " %10.1f%%", Context->cross_validation.results[f][1]);

	if ( Context->costs.matrix )
	{
	    fprintf(Of, "%7.2f", Context->cross_validation.results[f][2]);
	}
	fprintf(Of, "\n");

	for ( i = 0 ; i < 3 ; i++ )
	{
	    Sum[i] += Context->cross_validation.results[f][i];
	    SumSq[i] += Context->cross_validation.results[f][i] * Context->cross_validation.results[f][i];
	}
    }

    fprintf(Of, "\n  " T_Mean "\t");

    if ( ! PrintSize )
    {
	fprintf(Of, "      ");
    }
    else
    {
	fprintf(Of, "%6.1f", Sum[0] / Context->options.folds);
    }

    fprintf(Of, " %10.1f%%", Sum[1] / Context->options.folds);

    if ( Context->costs.matrix )
    {
	fprintf(Of, "%7.2f", Sum[2] / Context->options.folds);
    }

    fprintf(Of, "\n  " T_SE "\t");

    if ( ! PrintSize )
    {
	fprintf(Of, "      ");
    }
    else
    {
	fprintf(Of, "%6.1f", SE(Sum[0], SumSq[0], Context->options.folds));
    }

    fprintf(Of, " %10.1f%%", SE(Sum[1], SumSq[1], Context->options.folds));

    if ( Context->costs.matrix )
    {
	fprintf(Of, "%7.2f", SE(Sum[2], SumSq[2], Context->options.folds));
    }
    fprintf(Of, "\n");
}



float SE(float sum, float sumsq, int no)
/*    --  */
{
    float mean;

    mean = sum / no;

    return sqrt( ((sumsq - no * mean * mean) / (no - 1)) / no );
}
