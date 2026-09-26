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
/*	Read variable misclassification costs				 */
/*	-------------------------------------				 */
/*									 */
/*************************************************************************/


#include "defns.i"
#include "extern.i"
#include "c50_api_internal.h"


void GetMCostsInput(c50_context *Context, c50_input *Cf)
/*   --------------  */
{
    ClassNo	Pred, Real, p, r;
    char	Name[1000];
    CaseNo	i;
    float	Val, Sum=0;

    LineNo = 0;
    Context->line_buffer_position = Context->line_buffer;
    Context->line_buffer[0] = '\0';

    /*  Read entries from cost file  */

    while ( ReadNameInput(Context, Cf, Name, 1000, ':') )
    {
	if ( ! (Pred = Which(Name, Context->schema.class_names, 1, Context->schema.max_class)) )
	{
	    Error(BADCOSTCLASS, Name, "");
	}

	if ( ! ReadNameInput(Context, Cf, Name, 1000, ':') ||
	     ! (Real = Which(Name, Context->schema.class_names, 1, Context->schema.max_class)) )
	{
	    Error(BADCOSTCLASS, Name, "");
	}

	if ( ! ReadNameInput(Context, Cf, Name, 1000, ':') ||
	     sscanf(Name, "%f", &Val) != 1 || Val < 0 )
	{
	    Error(BADCOST, "", "");
	    Val = 1;
	}

	if ( Pred > 0 && Real > 0 && Pred != Real && Val != 1 )
	{
	    /*  Have a non-trivial cost entry  */

	    if ( ! Context->costs.matrix )
	    {
		/*  Set up cost matrices  */

		Context->costs.matrix = Alloc(Context->schema.max_class+1, float *);
		ForEach(p, 1, Context->schema.max_class)
		{
		    Context->costs.matrix[p] = Alloc(Context->schema.max_class+1, float);
		    ForEach(r, 1, Context->schema.max_class)
		    {
			Context->costs.matrix[p][r] = ( p == r ? 0.0 : 1.0 );
		    }
		}
	    }

	    Context->costs.matrix[Pred][Real] = Val;
	}
    }
    /*  Don't need weights etc. for predict or interpret, or
	if not using cost weighting  */

    if ( ! (Context->costs.weighted = Context->schema.max_class == 2 && Context->cases.max_case >= 0 && Context->costs.matrix) )
    {
	return;
    }

    /*  Determine class frequency distribution  */

    Context->training.class_frequencies = AllocZero(Context->schema.max_class+1, double);

    if ( Context->schema.case_weight_attribute )
    {
	Context->average_case_weight = 1;			/* relative weights not yet set */
	ForEach(i, 0, Context->cases.max_case)
	{
	    Context->training.class_frequencies[Class(Context->cases.records[i])] += RelCWt(Context, Context->cases.records[i]);
	}
    }
    else
    {
	ForEach(i, 0, Context->cases.max_case)
	{
	    Context->training.class_frequencies[Class(Context->cases.records[i])]++;
	}
    }

    /*  Find normalised weight multipliers  */

    Context->costs.weight_multipliers = Alloc(3, float);

    Sum = (Context->training.class_frequencies[1] * Context->costs.matrix[2][1] + Context->training.class_frequencies[2] * Context->costs.matrix[1][2]) /
	  (Context->training.class_frequencies[1] + Context->training.class_frequencies[2]);

    Context->costs.weight_multipliers[1] = Context->costs.matrix[2][1] / Sum;
    Context->costs.weight_multipliers[2] = Context->costs.matrix[1][2] / Sum;

    /*  Adjust Context->options.minimum_cases to take account of case reweighting  */

    Context->options.minimum_cases *= Min(Context->costs.weight_multipliers[1], Context->costs.weight_multipliers[2]);

    Free(Context->training.class_frequencies);					Context->training.class_frequencies = Nil;
}



void GetMCosts(c50_context *Context, FILE *Cf)
/*   ---------  */
{
    c50_input Input;

    c50_input_init_file(&Input, Cf);
    GetMCostsInput(Context, &Input);
    fclose(Cf);
}
