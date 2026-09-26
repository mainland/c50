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
/*	Routines for winnowing attributes				 */
/*	---------------------------------				 */
/*									 */
/*************************************************************************/


#include "defns.i"
#include "extern.i"
#include "c50_api_internal.h"

/*************************************************************************/
/*									 */
/*	Winnow attributes by constructing a tree from half the data.	 */
/*	Remove those that are never used as splits and those that	 */
/*	increase error on the remaining data, and check that the new	 */
/*	error cost does not increase					 */
/*									 */
/*************************************************************************/


void WinnowAtts(c50_context *Context)
/*   ----------  */
{
    Attribute	Att, Removed=0, Best;
    CaseNo	i, Bp, Ep;
    float	Base;
    Boolean	First=true, *Upper;
    ClassNo	c;
    Context->attributes_winnowed = false;

    /*  Save original case order  */

    Context->cases.saved_records = Alloc(Context->cases.max_case+1, DataRec);
    ForEach(i, 0, Context->cases.max_case)
    {
	Context->cases.saved_records[i] = Context->cases.records[i];
    }

    /*  Context->training.split_attributes data into two halves with equal class frequencies  */

    Upper = AllocZero(Context->schema.max_class+1, Boolean);

    Bp = 0;
    Ep = Context->cases.max_case;
    ForEach(i, 0, Context->cases.max_case)
    {
	c = Class(Context->cases.saved_records[i]);

	if ( Upper[c] )
	{
	    Context->cases.records[Ep--] = Context->cases.saved_records[i];
	}
	else
	{
	    Context->cases.records[Bp++] = Context->cases.saved_records[i];
	}

	Upper[c] = ! Upper[c];
    }

    Free(Upper);

    /*  Use first 50% of the cases for building a winnowing tree
	and remaining 50% for measuring attribute importance  */

    Context->training.attribute_importance = AllocZero(Context->schema.max_attribute+1, float);
    Context->training.split_attributes  = AllocZero(Context->schema.max_attribute+1, Boolean);
    Context->training.used_attributes   = AllocZero(Context->schema.max_attribute+1, Boolean);

    Base = TrialTreeCost(Context, true);

    /*  Remove attributes when doing so would reduce error cost  */

    ForEach(Att, 1, Context->schema.max_attribute)
    {
	if ( Context->training.attribute_importance[Att] < 0 )
	{
	    Context->schema.special_status[Att] ^= SKIP;
	    Removed++;
	}
    }

    /*  If any removed, rebuild tree and reinstate if error increases  */

    if ( Removed && TrialTreeCost(Context, false) > Base )
    {
	ForEach(Att, 1, Context->schema.max_attribute)
	{
	    if ( Context->training.attribute_importance[Att] < 0 )
	    {
		Context->training.attribute_importance[Att] = 1;
		Context->schema.special_status[Att] ^= SKIP;
		Verbosity(1, fprintf(Of, "  re-including %s\n", Context->schema.attribute_names[Att]))
	    }
	}

	Removed=0;
    }

    /*  Discard unused attributes  */

    ForEach(Att, 1, Context->schema.max_attribute)
    {
	if ( Att != Context->schema.class_attribute && ! Skip(Att) && ! Context->training.split_attributes[Att] )
	{
	    Context->schema.special_status[Att] ^= SKIP;
	    Removed++;
	}
    }

    /*  Print summary of winnowing  */

    if ( ! Removed )
    {
	fprintf(Of, T_NoWinnow);
    }
    else
    {
	fprintf(Of, T_AttributesWinnowed, Removed, Plural(Removed));

	/*  Print remaining attributes ordered by importance  */

	while ( true )
	{
	    Best = 0;
	    ForEach(Att, 1, Context->schema.max_attribute)
	    {
		if ( Context->training.attribute_importance[Att] >= 1 &&
		     ( ! Best || Context->training.attribute_importance[Att] > Context->training.attribute_importance[Best] ) )
		{
		    Best = Att;
		}
	    }
	    if ( ! Best ) break;

	    if ( First )
	    {
		fprintf(Of, T_EstImportance);
		First = false;
	    }
	    if ( Context->training.attribute_importance[Best] >= 1.005 )
	    {
		fprintf(Of, "%7d%%  %s\n",
			    (int) ((Context->training.attribute_importance[Best] - 1) * 100 + 0.5),
			    Context->schema.attribute_names[Best]);
	    }
	    else
	    {
		fprintf(Of, "     <1%%  %s\n", Context->schema.attribute_names[Best]);
	    }
	    Context->training.attribute_importance[Best] = 0;
	}
    }

    Context->attributes_winnowed = Removed != 0;

    if ( Removed )
    {
	/*  Reset Context->splits.discrete_attributes  */

	Context->splits.discrete_attribute_count = 0;
	ForEach(Att, 1, Context->schema.max_attribute)
	{
	    if ( Context->splits.discrete_frequencies[Att] && ! Skip(Att) )
	    {
		Context->splits.discrete_attributes[Context->splits.discrete_attribute_count++] = Att;
	    }
	}
    }

    /*  Restore case order and clean up  */

    ForEach(i, 0, Context->cases.max_case)
    {
	Context->cases.records[i] = Context->cases.saved_records[i];
    }

    FreeUnlessNil(Context->cases.saved_records);				Context->cases.saved_records = Nil;
    FreeUnlessNil(Context->training.attribute_importance);				Context->training.attribute_importance = Nil;
    FreeUnlessNil(Context->training.split_attributes);				Context->training.split_attributes = Nil;
    FreeUnlessNil(Context->training.used_attributes);				Context->training.used_attributes = Nil;

    Now = 0;
}



/*************************************************************************/
/*									 */
/*	Build trial tree and check error cost on remaining data.	 */
/*	If first time, note split attributes and check effect of	 */
/*	removing every attribute					 */
/*									 */
/*************************************************************************/


float TrialTreeCost(c50_context *Context, Boolean FirstTime)
/*    -------------  */
{
    Attribute	Att;
    float	Base, Cost, SaveMINITEMS;
    CaseNo	SaveMaxCase, Cut;
    int		SaveVERBOSITY;

    Verbosity(1,
	fprintf(Of, ( FirstTime ? "\nWinnow cycle:\n" : "\nCheck:\n" )))

    /*  Build and prune trial tree  */

    SaveMaxCase   = Context->cases.max_case;
    SaveVERBOSITY = Context->options.verbosity;
    SaveMINITEMS  = Context->options.minimum_cases;
    Context->options.minimum_cases      = Max(Context->options.minimum_cases / 2, 2.0);

    Cut = (Context->cases.max_case+1) / 2 - 1;

    InitialiseWeights(Context);
    Context->options.leaf_ratio = 0;
    Context->options.verbosity = 0;
    Context->cases.max_case   = Cut;

    memset(Context->splits.tested_attributes, 0, Context->schema.max_attribute+1);		/* reset tested attributes */

    SetMinGainThresh(Context);
    FormTree(Context, 0, Cut, 0, &Context->trees.winnow);

    if ( FirstTime )
    {
	/*  Find attributes used in unpruned tree  */

	ScanTree(Context->trees.winnow, Context->training.split_attributes);
    }

    Prune(Context, Context->trees.winnow);

    Context->options.verbosity = SaveVERBOSITY;
    Context->cases.max_case   = SaveMaxCase;
    Context->options.minimum_cases  = SaveMINITEMS;

    Verbosity(2,
	PrintTree(Context, Context->trees.winnow, "Winnowing tree:");
	fprintf(Of, "\n  training error cost %g\n",
		ErrCost(Context, Context->trees.winnow, 0, Cut)))

    Base = ErrCost(Context, Context->trees.winnow, Cut+1, Context->cases.max_case);

    Verbosity(1,
	fprintf(Of, "  initial error cost %g\n", Base))

    if ( FirstTime )
    {
	/*  Check each attribute used in pruned tree  */

	ScanTree(Context->trees.winnow, Context->training.used_attributes);

	ForEach(Att, 1, Context->schema.max_attribute)
	{

	    if ( ! Context->training.used_attributes[Att] )
	    {
		Verbosity(1,
		    if ( Att != Context->schema.class_attribute && ! Skip(Att) )
		    {
			fprintf(Of, "  %s not used\n", Context->schema.attribute_names[Att]);
		    })

		if ( Context->training.split_attributes[Att] )
		{
		    Context->training.attribute_importance[Att] = 1;
		}

		continue;
	    }

	    /*  Determine error cost if this attribute omitted  */

	    Context->schema.special_status[Att] ^= SKIP;

	    Cost = ErrCost(Context, Context->trees.winnow, Cut+1, Context->cases.max_case);

	    Context->training.attribute_importance[Att] = ( Cost < Base ? -1 : Cost / Base );
	    Verbosity(1,
		fprintf(Of, "  error cost without %s = %g%s\n",
			    Context->schema.attribute_names[Att], Cost,
			    ( Cost < Base ? " - excluded" : "" )))

	    Context->schema.special_status[Att] ^= SKIP;
	}
    }

    if ( Context->trees.winnow )
    {
	FreeTree(Context->trees.winnow);				Context->trees.winnow = Nil;
    }

    return Base;
}



/*************************************************************************/
/*									 */
/*	Determine the error rate or cost of T on cases Fp through Lp	 */
/*									 */
/*************************************************************************/


float ErrCost(c50_context *Context, Tree T, CaseNo Fp, CaseNo Lp)
/*    -------  */
{
    CaseNo	i;
    float	ErrCost=0;
    ClassNo	Pred;

    if ( Context->costs.matrix )
    {
	ForEach(i, Fp, Lp)
	{
	    if ( (Pred = TreeClassify(Context, Context->cases.records[i], T)) != Class(Context->cases.records[i]) )
	    {
		ErrCost += Context->costs.matrix[Pred][Class(Context->cases.records[i])];
	    }
	}
    }
    else
    {
	ForEach(i, Fp, Lp)
	{
	    if ( TreeClassify(Context, Context->cases.records[i], T) != Class(Context->cases.records[i]) )
	    {
		ErrCost += 1.0;
	    }
	}
    }

    return ErrCost;
}



/*************************************************************************/
/*									 */
/*	Find attributes used in tree T					 */
/*									 */
/*************************************************************************/


void ScanTree(Tree T, Boolean *used)
/*   --------  */
{
    DiscrValue	v;

    if ( T->NodeType )
    {
	used[T->Tested] = true;

	ForEach(v, 1, T->Forks)
	{
	    ScanTree(T->Branch[v], used);
	}
    }
}
