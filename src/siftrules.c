/*************************************************************************/
/*									 */
/*  Copyright 2010 Rulequest Research Pty Ltd.				 */
/*  Author: Ross Quinlan (quinlan@rulequest.com) [Rev Jan 2016]		 */
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
/*	Find a good subset of a set of rules				 */
/*	------------------------------------				 */
/*									 */
/*************************************************************************/


#include "defns.i"
#include "extern.i"
#include "c50_api_internal.h"


/*************************************************************************/
/*									 */
/*	Main rule selection routine.					 */
/*	1.  Form initial theory						 */
/*      2.  Hillclimb in MDL space					 */
/*									 */
/*************************************************************************/


void SiftRules(c50_context *Context, float EstErrRate)
/*   ---------  */
{
    RuleNo	r;
    int		d, *bp;
    CRule	R;
    float	CodeLength;
    CaseNo	i;

    NotifyStage(SIFTRULES);
    Progress(-(float) Context->rules.count);

    /*  Determine inverse of Context->rule_build.fires in Context->rule_build.coverage_counts, Context->rule_selection.coverage_pointers, Context->rule_selection.coverage_block  */

    InvertFires(Context);

    /*  Clean up any subsets in conditions by removing values that do
	not appear in the covered cases  */

    if ( Context->options.subset_splits )
    {
	PruneSubsets(Context);
    }

    Context->rule_selection.covered_cases = Alloc(Context->cases.max_case+1, Boolean);
    Context->rule_selection.rules_included  = AllocZero(Context->rules.count+1, Boolean);

    /*  Set initial theory  */

    SetInitialTheory(Context);

    Context->rule_selection.rule_bits = Alloc(Context->rules.count+1, float);

    /*  Calculate the number of bits associated with attribute tests;
	this is not repeated in boosting, composite rulesets etc  */

    if ( ! Context->rule_build.branch_bits || Context->rules.count > Context->cases.max_case )
    {
	GenerateLogs(Context,
	    Max(Context->cases.max_case+1,
		Max(Context->schema.max_attribute,
		    Max(Context->schema.max_class,
			Max(Context->schema.max_discrete_value,
			    Context->rules.count)))));
    }

    if ( ! Context->rule_build.branch_bits )
    {
	FindTestCodes(Context);
    }

    /*  Determine rule codelengths  */

    if ( Context->rules.count >= Context->cases.max_case+1 )
    {
	Realloc(Context->rule_build.list, Context->rules.count+1, CaseNo);
    }

    ForEach(r, 1, Context->rules.count)
    {
	R = Context->rules.rules[r];

	CodeLength = 0;
	ForEach(d, 1, R->Size)
	{
	    CodeLength += CondBits(Context, R->Lhs[d]);
	}
	Context->rule_selection.rule_bits[r] = CodeLength + Context->rule_build.log_case_count[R->Size] - Context->rule_build.log_factorial[R->Size];
    }

    /*  Use estimated error rate to determine the bits required to
	label a theory's prediction for a case as an error or correct  */

    if ( EstErrRate > 0.5 ) EstErrRate = 0.45;

    Context->rule_selection.error_bits = - Log(EstErrRate);
    Context->rule_selection.correct_bits  = - Log(1.0 - EstErrRate);


    /*  Allocate tables used in hillclimbing  */

    Context->rule_selection.delta_errors = Alloc(Context->rules.count+1, float);
    Context->rule_selection.top_classes = Alloc(Context->cases.max_case+1, ClassNo);

    Context->rule_selection.alternate_classes = Alloc(Context->cases.max_case+1, ClassNo);
    Context->rule_selection.total_votes  = Alloc(Context->cases.max_case+1, int *);

    bp = AllocZero((Context->cases.max_case+1) * (Context->schema.max_class+1), int);
    ForEach(i, 0, Context->cases.max_case)
    {
	Context->rule_selection.total_votes[i] = bp;
	bp += Context->schema.max_class + 1;
    }

    /*  Now find best subset of rules  */

    HillClimb(Context);

    /*  Determine default class and reorder rules  */

    SetDefaultClass(Context);
    OrderRules(Context);

    /*  Deallocate storage  */

    FreeSiftRuleData(Context);
}



/*************************************************************************/
/*								  	 */
/*	Find inverse of Context->rule_build.fires[][] in Context->rule_build.coverage_counts, Context->rule_selection.coverage_pointers, and Context->rule_selection.coverage_block.	 */
/*								  	 */
/*	Context->rule_build.coverage_counts[i] = number of rules covering case i (set by NewRule)	 */
/*								  	 */
/*	Set up Context->rule_selection.coverage_pointers as pointers into Context->rule_selection.coverage_block so that		 */
/*	Context->rule_selection.coverage_pointers[i] is the start of the compressed entry for case i	 */
/*								  	 */
/*************************************************************************/


void InvertFires(c50_context *Context)
/*   -----------  */
{
    RuleNo	r, Entry;
    int		j, Blocks, Extra;
    CaseNo	i;
    Byte	*p, *From, *To, *Next;
    size_t	CovByBlockSize=0;

    Context->rule_selection.coverage_pointers = Alloc(Context->cases.max_case+2, Byte *);
    Extra = Context->rules.count / 128;		/* max number of filler entries */
    ForEach(i, 1, Context->cases.max_case+1)
    {
	CovByBlockSize += Context->rule_build.coverage_counts[i-1] + Extra;
    }

    Context->rule_selection.coverage_block = Alloc(CovByBlockSize, Byte);
    Context->rule_selection.coverage_pointers[0] = Context->rule_selection.coverage_block;
    ForEach(i, 1, Context->cases.max_case+1)
    {
	Context->rule_selection.coverage_pointers[i] = Context->rule_selection.coverage_pointers[i-1] + Context->rule_build.coverage_counts[i-1] + Extra;
    }

    Context->rule_selection.last_covering_rule = AllocZero(Context->cases.max_case+1, RuleNo);

    /*  Add entries for each rule  */

    ForEach(r, 1, Context->rules.count)
    {
	Uncompress(Context->rule_build.fires[r], Context->rule_build.list);
	ForEach(j, 1, Context->rule_build.list[0])
	{
	    i = Context->rule_build.list[j];

	    /*  Add compressed entry for this rule  */

	    p = Context->rule_selection.coverage_pointers[i];
	    Entry = r - Context->rule_selection.last_covering_rule[i];
	    Context->rule_selection.last_covering_rule[i] = r;

	    while ( Entry > 127 )
	    {
		Blocks = (Entry >> 7);
		if ( Blocks > 127 ) Blocks = 127;
		Entry -= Blocks * 128;
		*p++   = Blocks + 128;
	    }

	    *p++ = Entry;
	    Context->rule_selection.coverage_pointers[i] = p;
	}
    }

    Free(Context->rule_selection.last_covering_rule);					Context->rule_selection.last_covering_rule = Nil;

    /*  Reset Context->rule_selection.coverage_pointers entries and compact  */

    To   = Context->rule_selection.coverage_pointers[0];
    From = Context->rule_selection.coverage_pointers[0] = Context->rule_selection.coverage_block;

    ForEach(i, 1, Context->cases.max_case)
    {
	From += Context->rule_build.coverage_counts[i-1] + Extra;
	Next  = Context->rule_selection.coverage_pointers[i];
	Context->rule_selection.coverage_pointers[i] = To;

	for ( p = From ; p < Next ; )
	{
	    *To++ = *p++;
	}
    }

    /*  Reduce Context->rule_selection.coverage_block to size actually used  */

    From = Context->rule_selection.coverage_block;			/* current address */

    Realloc(Context->rule_selection.coverage_block, To - Context->rule_selection.coverage_block, Byte);

    if ( Context->rule_selection.coverage_block != From )
    {
	/*  Context->rule_selection.coverage_block has been moved  */

	ForEach(i, 0, Context->cases.max_case)
	{
	    Context->rule_selection.coverage_pointers[i] += Context->rule_selection.coverage_block - From;
	}
    }
}



/*************************************************************************/
/*								  	 */
/*	Determine code lengths for attributes and branches		 */
/*								  	 */
/*************************************************************************/


void FindTestCodes(c50_context *Context)
/*   -------------  */
{
    Attribute	Att;
    DiscrValue	v, V;
    CaseNo	i, *ValFreq;
    int		PossibleAtts=0;
    float	Sum;

    Context->rule_build.branch_bits = AllocZero(Context->schema.max_attribute+1, float);
    Context->rule_build.attribute_values  = AllocZero(Context->schema.max_attribute+1, int);

    ForEach(Att, 1, Context->schema.max_attribute)
    {
	if ( Skip(Att) || Att == Context->schema.class_attribute ) continue;

	PossibleAtts++;

	if ( Ordered(Att) )
	{
	    Context->rule_build.branch_bits[Att] = 1 + 0.5 * Context->rule_build.log_case_count[Context->schema.max_attribute_value[Att] - 1];
	}
	else
	if ( (V = Context->schema.max_attribute_value[Att]) )
	{
	    /*  Discrete attribute  */

	    ValFreq = AllocZero(V+1, CaseNo);

	    ForEach(i, 0, Context->cases.max_case)
	    {
		assert(XDVal(Context->cases.records[i],Att) >= 0 && XDVal(Context->cases.records[i],Att) <= V);
		ValFreq[ XDVal(Context->cases.records[i],Att) ]++;
	    }

	    Sum = 0;
	    ForEach(v, 1, V)
	    {
		if ( ValFreq[v] )
		{
		    Sum += (ValFreq[v] / (Context->cases.max_case+1.0)) *
			   (Context->rule_build.log_case_count[Context->cases.max_case+1] - Context->rule_build.log_case_count[ValFreq[v]]);
		    Context->rule_build.attribute_values[Att]++;
		}
	    }
	    Free(ValFreq);

	    Context->rule_build.branch_bits[Att] = Sum;
	}
	else
	{
	    /*  Continuous attribute  */

	    Context->rule_build.branch_bits[Att] = Context->splits.possible_cuts[Att] > 1 ?
			      1 + 0.5 * Context->rule_build.log_case_count[Context->splits.possible_cuts[Att]] : 0 ;
	}
    }

    Context->rule_build.attribute_test_bits = Context->rule_build.log_case_count[PossibleAtts];
}



/*************************************************************************/
/*									 */
/*	Determine the number of bits required to encode a condition	 */
/*									 */
/*************************************************************************/


float CondBits(c50_context *Context, Condition C)
/*    --------  */
{
    Attribute	Att;
    float	Code=0;
    int		Elts=0;
    DiscrValue	v;

    Att = C->Tested;
    switch ( C->NodeType )
    {
	case BrDiscr:		/* test of discrete attribute */
	case BrThresh:		/* test of continuous attribute */

	    return Context->rule_build.attribute_test_bits + Context->rule_build.branch_bits[Att];

	case BrSubset:		/* subset test on discrete attribute  */

	    /* Ignore subset test form for ordered attributes  */

	    if ( Ordered(Att) )
	    {
		return Context->rule_build.attribute_test_bits + Context->rule_build.branch_bits[Att];
	    }

	    ForEach(v, 1, Context->schema.max_attribute_value[Att])
	    {
		if ( In(v, C->Subset) )
		{
		    Elts++;
		}
	    }
	    Elts = Min(Elts, Context->rule_build.attribute_values[Att] - 1);  /* if values not present */
	    Code = Context->rule_build.log_factorial[Context->rule_build.attribute_values[Att]] -
		   (Context->rule_build.log_factorial[Elts] + Context->rule_build.log_factorial[Context->rule_build.attribute_values[Att] - Elts]);

	    return Context->rule_build.attribute_test_bits + Code;
    }

    return 0;
}



/*************************************************************************/
/*									 */
/*	Select initial theory.  This is important, since the greedy	 */
/*	optimization procedure is very sensitive to starting with	 */
/*	a reasonable theory.						 */
/*									 */
/*	The theory is constructed class by class.  For each class,	 */
/*	rules are added in confidence order until all of the cases of	 */
/*	that class are covered.  Rules that do not improve coverage	 */
/*	are skipped.							 */
/*									 */
/*************************************************************************/


void SetInitialTheory(c50_context *Context)
/*   ----------------  */
{
    ClassNo	c;
    RuleNo	r, Active=0;

    ForEach(c, 1, Context->schema.max_class)
    {
	CoverClass(Context, c);
    }

    /*  Remove rules that don't help coverage  */

    ForEach(r, 1, Context->rules.count)
    {
	if ( (Context->rule_selection.rules_included[r] &= 1) ) Active++;
    }
}



void CoverClass(c50_context *Context, ClassNo Target)
/*   ----------  */
{
    CaseNo	i;
    double	Remaining, FalsePos=0, NewFalsePos, NewTruePos;
    RuleNo	r, Best;
    int		j;

    memset(Context->rule_selection.covered_cases, false, Context->cases.max_case+1);

    Remaining = Context->training.class_frequencies[Target];

    while ( Remaining > FalsePos )
    {
	/*  Find most accurate unused rule from a leaf  */

	Best = 0;
	ForEach(r, 1, Context->rules.count)
	{
	    if ( Context->rules.rules[r]->Rhs == Target && ! Context->rule_selection.rules_included[r] &&
		 Context->rules.rules[r]->Correct >= Context->options.minimum_cases )
	    {
		if ( ! Best || Context->rules.rules[r]->Vote > Context->rules.rules[Best]->Vote ) Best = r;
	    }
	}

	if ( ! Best ) return;

	/*  Check increased coverage  */

	NewFalsePos = NewTruePos = 0;

	Uncompress(Context->rule_build.fires[Best], Context->rule_build.list);
	for( j = Context->rule_build.list[0] ; j ; j-- )
	{
	    i = Context->rule_build.list[j];
	    if ( ! Context->rule_selection.covered_cases[i] )
	    {
		if ( Class(Context->cases.records[i]) == Target )
		{
		    NewTruePos += Weight(Context->cases.records[i]);
		}
		else
		{
		    NewFalsePos += Weight(Context->cases.records[i]);
		}
	    }
	}

	/*  If coverage is not increased, set Context->rule_selection.rules_included to 2 so that
	    the rule can be removed later  */

	if ( NewTruePos - NewFalsePos <= Context->options.minimum_cases + Epsilon )
	{
	    Context->rule_selection.rules_included[Best] = 2;
	}
	else
	{
	    Remaining -= NewTruePos;
	    FalsePos  += NewFalsePos;

	    Context->rule_selection.rules_included[Best] = true;

	    Uncompress(Context->rule_build.fires[Best], Context->rule_build.list);
	    for( j = Context->rule_build.list[0] ; j ; j-- )
	    {
		i = Context->rule_build.list[j];
		if ( ! Context->rule_selection.covered_cases[i] )
		{
		    Context->rule_selection.covered_cases[i] = true;
		}
	    }
	}
    }
}



/*************************************************************************/
/*									 */
/*	Calculate total message length as				 */
/*	  THEORYFRAC * cost of transmitting theory			 */
/*	  + cost of identifying and correcting errors			 */
/*									 */
/*	The cost of identifying errors assumes that the final theory	 */
/*	will have about the same error rate as the pruned tree, so	 */
/*	is approx. the sum of the corresponding messages.		 */
/*									 */
/*************************************************************************/


double MessageLength(c50_context *Context, RuleNo NR, double RuleBits,
		     float Errs)
/*  -------------  */
{
    return
	(THEORYFRAC * Max(0, RuleBits - Context->rule_build.log_factorial[NR]) +
	 Errs * Context->rule_selection.error_bits + (Context->cases.max_case+1 - Errs) * Context->rule_selection.correct_bits +
	 Errs * Context->rule_build.log_case_count[Context->schema.max_class-1]);
}



/*************************************************************************/
/*									 */
/*	Improve a subset of rules by adding and deleting rules.		 */
/*	MDL costs are rounded to nearest 0.01 bit			 */
/*									 */
/*************************************************************************/


void HillClimb(c50_context *Context)
/*   ---------  */
{
    RuleNo	r, RuleCount=0, OriginalCount, Toggle, LastToggle=0;
    int		OutCount;
    CaseNo	i;
    int		j;
    CaseCount	Errs;
    double	RuleBits=0;
    double	LastCost=1E99, CurrentCost, AltCost, NewCost;
    Boolean	DeleteOnly=false;

    (void) LastCost;  /* Used only when VerbOpt is enabled. */

    ForEach(r, 1, Context->rules.count)
    {
	if ( Context->rule_selection.rules_included[r] )
	{
	    RuleBits += Context->rule_selection.rule_bits[r];
	    RuleCount++;
	}
    }
    OriginalCount = RuleCount;

    InitialiseVotes(Context);
    Verbosity(1, fprintf(Of, "\n"))

    /*  Initialise Context->rule_selection.delta_errors[]  */

    Errs = CalculateDeltaErrs(Context);

    /*  Add or drop rule with greatest reduction in coding cost  */

    while ( true )
    {
	CurrentCost = NewCost =
	    MessageLength(Context, RuleCount, RuleBits, Errs);

	Verbosity(1,
	    fprintf(Of, "\t%d rules, %.1f errs, cost=%.1f bits\n",
		   RuleCount, Errs, CurrentCost/100.0);

	    if ( ! DeleteOnly && CurrentCost > LastCost )
	    {
		fprintf(Of, "ERROR %g %g\n",
			    CurrentCost/1000.0, LastCost/100.0);
		break;
	    })

	Toggle = OutCount = 0;

	ForEach(r, 1, Context->rules.count)
	{
	    if ( r == LastToggle ) continue;

	    if ( Context->rule_selection.rules_included[r] )
	    {
		AltCost = MessageLength(Context, RuleCount - 1,
					RuleBits - Context->rule_selection.rule_bits[r],
					Errs + Context->rule_selection.delta_errors[r]);
	    }
	    else
	    {
		if ( Errs < 1E-3 || DeleteOnly ) continue;

		AltCost = MessageLength(Context, RuleCount + 1,
					RuleBits + Context->rule_selection.rule_bits[r],
					Errs + Context->rule_selection.delta_errors[r]);
	    }

	    Verbosity(2,
		if ( ! (OutCount++ % 5) ) fprintf(Of, "\n\t\t");
		fprintf(Of, "%d<%g=%.1f> ",
			    r, Context->rule_selection.delta_errors[r], (AltCost - CurrentCost)/100.0))

	    if ( AltCost < NewCost ||
		 ( AltCost == NewCost && Context->rule_selection.rules_included[r] ) )
	    {
		Toggle  = r;
		NewCost = AltCost;
	    }
	}

	if ( ! DeleteOnly && NewCost > CurrentCost )
	{
	    DeleteOnly = true;
	    Verbosity(1, fprintf(Of, "(start delete mode)\n"))
	}

	Verbosity(2, fprintf(Of, "\n"))

	if ( ! Toggle || ( DeleteOnly && RuleCount <= OriginalCount ) ) break;

	Verbosity(1,
	    fprintf(Of, "\t%s rule %d/%d (errs=%.1f, cost=%.1f bits)\n",
		   ( Context->rule_selection.rules_included[Toggle] ? "Delete" : "Add" ),
		   Context->rules.rules[Toggle]->TNo, Context->rules.rules[Toggle]->RNo,
		   Errs + Context->rule_selection.delta_errors[Toggle], NewCost/100.0))

	/*  Adjust vote information  */

	Uncompress(Context->rule_build.fires[Toggle], Context->rule_build.list);
	for ( j = Context->rule_build.list[0] ; j ; j-- )
	{
	    i = Context->rule_build.list[j];

	    /*  Downdate Context->rule_selection.delta_errors for all rules except Toggle that cover i  */

	    UpdateDeltaErrs(Context, i, -Weight(Context->cases.records[i]), Toggle);

	    if ( Context->rule_selection.rules_included[Toggle] )
	    {
		Context->rule_selection.total_votes[i][Context->rules.rules[Toggle]->Rhs] -= Context->rules.rules[Toggle]->Vote;
	    }
	    else
	    {
		Context->rule_selection.total_votes[i][Context->rules.rules[Toggle]->Rhs] += Context->rules.rules[Toggle]->Vote;
	    }

	    CountVotes(Context, i);

	    /*  Update Context->rule_selection.delta_errors for all rules except Toggle that cover i  */

	    UpdateDeltaErrs(Context, i, Weight(Context->cases.records[i]), Toggle);
	}

	/*  Update information about rules selected and current errors  */

	if ( Context->rule_selection.rules_included[Toggle] )
	{
	    Context->rule_selection.rules_included[Toggle] = false;
	    RuleBits -= Context->rule_selection.rule_bits[Toggle];
	    RuleCount--;
	}
	else
	{
	    Context->rule_selection.rules_included[Toggle] = true;
	    RuleBits += Context->rule_selection.rule_bits[Toggle];
	    RuleCount++;
	}

	Errs += Context->rule_selection.delta_errors[Toggle];
	Context->rule_selection.delta_errors[Toggle] = - Context->rule_selection.delta_errors[Toggle];

	LastToggle = Toggle;
	LastCost   = CurrentCost;

	Progress(1.0);
    }
}



/*************************************************************************/
/*									 */
/*	Determine votes for each case from initial rules		 */
/*	Note: no vote for default class					 */
/*									 */
/*************************************************************************/


void InitialiseVotes(c50_context *Context)
/*   ---------------  */
{
    CaseNo	i;
    int		j, Vote;
    ClassNo	Rhs;
    RuleNo	r;

    /*  Adjust vote for each case covered by rule  */

    ForEach(r, 1, Context->rules.count)
    {
	if ( ! Context->rule_selection.rules_included[r] ) continue;

	Rhs  = Context->rules.rules[r]->Rhs;
	Vote = Context->rules.rules[r]->Vote;

	Uncompress(Context->rule_build.fires[r], Context->rule_build.list);
	for ( j = Context->rule_build.list[0] ; j ; j-- )
	{
	    Context->rule_selection.total_votes[Context->rule_build.list[j]][Rhs] += Vote;
	}
    }

    /*  Find the best and alternate class for each case  */

    ForEach(i, 0, Context->cases.max_case)
    {
	CountVotes(Context, i);
    }
}



/*************************************************************************/
/*									 */
/*	Find the best and second-best class for each case using the	 */
/*	current values of Context->rule_selection.total_votes					 */
/*									 */
/*************************************************************************/


void CountVotes(c50_context *Context, CaseNo i)
/*   ----------  */
{
    ClassNo	c, First=0, Second=0;
    int		V;

    ForEach(c, 1, Context->schema.max_class)
    {
	if ( (V = Context->rule_selection.total_votes[i][c]) )
	{
	    if ( ! First || V > Context->rule_selection.total_votes[i][First] )
	    {
		Second = First;
		First  = c;
	    }
	    else
	    if ( ! Second || V > Context->rule_selection.total_votes[i][Second] )
	    {
		Second = c;
	    }
	}
    }

    Context->rule_selection.top_classes[i] = First;
    Context->rule_selection.alternate_classes[i] = Second;
}



/*************************************************************************/
/*									 */
/*	Adjust DeltaErrors for all rules except Toggle that cover case i */
/*									 */
/*************************************************************************/


#define Prefer(d,c1,c2) ((d) > 0 || ((d) == 0 && c1 < c2))

void UpdateDeltaErrs(c50_context *Context, CaseNo i, double Delta,
		     RuleNo Toggle)
/*   ---------------  */
{
    ClassNo	RealClass, Top, Alt, Rhs;
    RuleNo	r;
    Byte	*p;
    int		k;

    RealClass = Class(Context->cases.records[i]);
    Top	= Context->rule_selection.top_classes[i];
    Alt = Context->rule_selection.alternate_classes[i];

    r = 0;
    p = Context->rule_selection.coverage_pointers[i];
    ForEach(k, 1, Context->rule_build.coverage_counts[i])
    {
	/*  Update r to next rule covering case i  */

	while ( (*p) & 128 )
	{
	    r += ((*p++) & 127) * 128;
	}
	r += *p++;

	if ( r != Toggle )
	{
	    /*  Examine effect of adding or deleting rule  */
	
	    Rhs = Context->rules.rules[r]->Rhs;

	    if ( Context->rule_selection.rules_included[r] )
	    {
		if ( Rhs == Top &&
		     Prefer(Context->rule_selection.total_votes[i][Alt] - (Context->rule_selection.total_votes[i][Top] - Context->rules.rules[r]->Vote),
			    Alt, Top) )
		{
		    Context->rule_selection.delta_errors[r] +=
			(Context->costs.normalized_matrix[Alt][RealClass] - Context->costs.normalized_matrix[Top][RealClass]) * Delta;
		}
	    }
	    else
	    {
		if ( Rhs != Top &&
		     Prefer(Context->rule_selection.total_votes[i][Rhs] + Context->rules.rules[r]->Vote - Context->rule_selection.total_votes[i][Top],
			    Rhs, Top) )
		{
		    Context->rule_selection.delta_errors[r] +=
			(Context->costs.normalized_matrix[Rhs][RealClass] - Context->costs.normalized_matrix[Top][RealClass]) * Delta;
		}
	    }
	}
    }
}



/*************************************************************************/
/*									 */
/*	Calculate initial value of Context->rule_selection.delta_errors and total errors		 */
/*									 */
/*************************************************************************/


CaseCount CalculateDeltaErrs(c50_context *Context)
/*        ------------------  */
{
    RuleNo	r;
    CaseNo	i;
    double	Errs=0;

    ForEach(i, 0, Context->cases.max_case)
    {
	Errs += Weight(Context->cases.records[i]) * Context->costs.normalized_matrix[Context->rule_selection.top_classes[i]][Class(Context->cases.records[i])];
    }

    ForEach(r, 1, Context->rules.count)
    {
	Context->rule_selection.delta_errors[r] = 0;
    }

    ForEach(i, 0, Context->cases.max_case)
    {
	UpdateDeltaErrs(Context, i, Weight(Context->cases.records[i]), 0);
    }

    return Errs;
}



/*************************************************************************/
/*									 */
/*	Remove unrepresented values from subsets			 */
/*									 */
/*************************************************************************/


void PruneSubsets(c50_context *Context)
/*   ------------  */
{
    Set		*PossibleValues;
    Attribute	Att, *Atts, Last;
    int		*Bytes, d, NAtts, j, b;
    CaseNo	i;
    CRule	R;
    RuleNo	r;

    /*  Allocate subsets for possible values  */

    Atts  = Alloc(Context->schema.max_attribute+1, Attribute);
    Bytes = Alloc(Context->schema.max_attribute+1, int);

    PossibleValues = AllocZero(Context->schema.max_attribute+1, Set);
    ForEach(Att, 1, Context->schema.max_attribute)
    {
	if ( Context->schema.max_attribute_value[Att] > 3 )
	{
	    Bytes[Att] = (Context->schema.max_attribute_value[Att]>>3)+1;
	    PossibleValues[Att] = AllocZero(Bytes[Att], Byte);
	}
    }

    /*  Check each rule in turn  */

    ForEach(r, 1, Context->rules.count)
    {
	R = Context->rules.rules[r];
	NAtts = 0;

	/*  Find all subset conditions  */

	ForEach(d, 1, R->Size)
	{
	    if ( R->Lhs[d]->NodeType != BrSubset ) continue;

	    Atts[++NAtts] = Att = R->Lhs[d]->Tested;
	    ClearBits(Bytes[Att], PossibleValues[Att]);
	}

	if ( ! NAtts ) continue;	/* no subset conditions */

	/*  Scan cases covered by this rule  */

	Uncompress(Context->rule_build.fires[r], Context->rule_build.list);
	for ( j = Context->rule_build.list[0] ; j ; j-- )
	{
	    i = Context->rule_build.list[j];

	    /*  Record values of listed attributes  */

	    ForEach(d, 1, NAtts)
	    {
		Att = Atts[d];
		SetBit(DVal(Context->cases.records[i], Att), PossibleValues[Att]);
	    }
	}

	/*  Delete unrepresented values  */

	ForEach(d, 1, R->Size)
	{
	    if ( R->Lhs[d]->NodeType != BrSubset ) continue;

	    Att = R->Lhs[d]->Tested;
	    ForEach(b, 0, Bytes[Att]-1)
	    {
		R->Lhs[d]->Subset[b] &= PossibleValues[Att][b];
	    }

	    if ( Elements(Context, Att, R->Lhs[d]->Subset, &Last) == 1 )
	    {
		R->Lhs[d]->NodeType  = BrDiscr;
		R->Lhs[d]->TestValue = Last;
		Free(R->Lhs[d]->Subset);
	    }
	}
    }

    FreeVector((void **) PossibleValues, 1, Context->schema.max_attribute);
    Free(Bytes);
    Free(Atts);
}



/*************************************************************************/
/*									 */
/*	Choose the default class as the one with the maximum		 */
/*	weight of uncovered cases					 */
/*									 */
/*************************************************************************/


void SetDefaultClass(c50_context *Context)
/*   ---------------  */
{
    RuleNo	r;
    ClassNo	c;
    double	*UncoveredWeight, TotUncovered=1E-3;
    CaseNo	i, j;

    memset(Context->rule_selection.covered_cases, false, Context->cases.max_case+1);
    UncoveredWeight = AllocZero(Context->schema.max_class+1, double);

    /*  Check which cases are covered by at least one rule  */

    ForEach(r, 1, Context->rules.count)
    {
	if ( ! Context->rule_selection.rules_included[r] ) continue;

	Uncompress(Context->rule_build.fires[r], Context->rule_build.list);
	for ( j = Context->rule_build.list[0] ; j ; j-- )
	{
	    Context->rule_selection.covered_cases[Context->rule_build.list[j]] = true;
	}
    }

    /*  Find weights by class of uncovered cases  */

    ForEach(i, 0, Context->cases.max_case)
    {
	if ( ! Context->rule_selection.covered_cases[i] )
	{
	    UncoveredWeight[ Class(Context->cases.records[i]) ] += Weight(Context->cases.records[i]);
	    TotUncovered += Weight(Context->cases.records[i]);
	}
    }

    /*  Choose new default class using rel freq and rel uncovered  */

    Verbosity(1, fprintf(Of, "\n    Weights of uncovered cases:\n"));

    ForEach(c, 1, Context->schema.max_class)
    {
	Verbosity(1, fprintf(Of, "\t%s (%.2f): %.1f\n",
			    Context->schema.class_names[c], Context->training.class_frequencies[c] / (Context->cases.max_case + 1.0),
			    UncoveredWeight[c]));

	Context->class_sum[c] = (UncoveredWeight[c] + 1) / (TotUncovered + 2.0) +
		      Context->training.class_frequencies[c] / (Context->cases.max_case + 1.0);
    }

    Context->default_class =
	SelectClass(Context, 1, (Boolean) (Context->costs.matrix && ! Context->costs.weighted));

    Free(UncoveredWeight);
}



/*************************************************************************/
/*									 */
/*	Swap two rules							 */
/*									 */
/*************************************************************************/


void SwapRule(c50_context *Context, RuleNo A, RuleNo B)
/*   --------  */
{
    CRule	Hold;
    Boolean	HoldIn;

    Hold    = Context->rules.rules[A];
    Context->rules.rules[A] = Context->rules.rules[B];
    Context->rules.rules[B] = Hold;

    HoldIn    = Context->rule_selection.rules_included[A];
    Context->rule_selection.rules_included[A] = Context->rule_selection.rules_included[B];
    Context->rule_selection.rules_included[B] = HoldIn;
}



/*************************************************************************/
/*									 */
/*	Order rules by utility, least important first			 */
/*	(Called after HilClimb(), so Context->rule_selection.rules_included etc already known.)		 */
/*									 */
/*************************************************************************/


int OrderByUtility(c50_context *Context)
/*  --------------  */
{
    RuleNo	r, *Drop, NDrop=0, NewNRules=0, Toggle;
    CaseNo	i;
    int		j, OutCount;
    double	Errs=0;

    Verbosity(1, fprintf(Of, "\n    Determining rule utility\n"))

    Drop = Alloc(Context->rules.count, RuleNo);

    /*  Find the rule that has the least beneficial effect on accuracy  */

    while ( true )
    {
	Toggle = OutCount = 0;

	ForEach(r, 1, Context->rules.count)
	{
	    if ( ! Context->rule_selection.rules_included[r] ) continue;

	    Verbosity(2,
		if ( ! (OutCount++ %10 ) ) fprintf(Of, "\n\t\t");
		fprintf(Of, "%d<%g> ", r, Context->rule_selection.delta_errors[r]))

	    if ( ! Toggle ||
		 Context->rule_selection.delta_errors[r] < Context->rule_selection.delta_errors[Toggle] - 1E-3 ||
		 ( Context->rule_selection.delta_errors[r] < Context->rule_selection.delta_errors[Toggle] + 1E-3 &&
		   Context->rule_selection.rule_bits[r] > Context->rule_selection.rule_bits[Toggle] ) )
	    {
		Toggle = r;
	    }
	}
	Verbosity(2, fprintf(Of, "\n"))

	if ( ! Toggle ) break;

	Verbosity(1,
	    fprintf(Of, "\tDelete rule %d/%d (errs up %.1f)\n",
		   Context->rules.rules[Toggle]->TNo, Context->rules.rules[Toggle]->RNo,
		   Errs + Context->rule_selection.delta_errors[Toggle]))

	/*  Adjust vote information  */

	Uncompress(Context->rule_build.fires[Toggle], Context->rule_build.list);
	for ( j = Context->rule_build.list[0] ; j ; j-- )
	{
	    i = Context->rule_build.list[j];

	    /*  Downdate Context->rule_selection.delta_errors for all rules except Toggle that cover i  */

	    UpdateDeltaErrs(Context, i, -Weight(Context->cases.records[i]), Toggle);

	    Context->rule_selection.total_votes[i][Context->rules.rules[Toggle]->Rhs] -= Context->rules.rules[Toggle]->Vote;

	    CountVotes(Context, i);

	    /*  Update Context->rule_selection.delta_errors for all rules except Toggle that cover i  */

	    UpdateDeltaErrs(Context, i, Weight(Context->cases.records[i]), Toggle);
	}

	Drop[NDrop++]  = Toggle;
	Context->rule_selection.rules_included[Toggle] = false;

	Errs += Context->rule_selection.delta_errors[Toggle];
    }

    /*  Now reverse the order  */

    while ( --NDrop >= 0 )
    {
	NewNRules++;
	Context->rule_selection.rules_included[Drop[NDrop]] = true;
	SwapRule(Context, Drop[NDrop], NewNRules);

	/*  Have to alter rule number in Drop  */
	ForEach(r, 0, NDrop-1)
	{
	    if ( Drop[r] == NewNRules ) Drop[r] = Drop[NDrop];
	}
    }
    Free(Drop);

    return NewNRules;
}




/*************************************************************************/
/*									 */
/*	Order rules by class and then by rule Context->options.confidence_factor			 */
/*									 */
/*************************************************************************/


int OrderByClass(c50_context *Context)
/*  ------------  */
{
    RuleNo	r, nr, NewNRules=0;
    ClassNo	c;

    ForEach(c, 1, Context->schema.max_class)
    {
	while ( true )
	{
	    nr = 0;
	    ForEach(r, NewNRules+1, Context->rules.count)
	    {
		if ( Context->rule_selection.rules_included[r] && Context->rules.rules[r]->Rhs == c &&
		     ( ! nr || Context->rules.rules[r]->Vote > Context->rules.rules[nr]->Vote ) )
		{
		    nr = r;
		}
	    }

	    if ( ! nr ) break;

	    NewNRules++;
	    if ( nr != NewNRules )
	    {
		SwapRule(Context, NewNRules, nr);
	    }
	}
    }

    return NewNRules;
}



/*************************************************************************/
/*									 */
/*	Discard deleted rules and sequence and renumber those remaining. */
/*	Sort by class and then by rule Context->options.confidence_factor or by utility			 */
/*									 */
/*************************************************************************/


void OrderRules(c50_context *Context)
/*   ----------  */
{
    RuleNo	r, NewNRules;

    NewNRules = ( Context->options.utility_bands ? OrderByUtility(Context) : OrderByClass(Context) );

    ForEach(r, 1, NewNRules)
    {
	Context->rules.rules[r]->RNo = r;
    }

    /*  Free discarded rules  */

    ForEach(r, NewNRules+1, Context->rules.count)
    {
	FreeRule(Context->rules.rules[r]);
    }

    Context->rules.count = NewNRules;
}



/*************************************************************************/
/*									 */
/*	Tabluate logs and log factorials (to improve speed)		 */
/*									 */
/*************************************************************************/


void GenerateLogs(c50_context *Context, int MaxN)
/*   ------------  */
{
    CaseNo	i;

    if ( Context->rule_build.log_case_count )
    {
	Realloc(Context->rule_build.log_case_count, MaxN+2, double);
	Realloc(Context->rule_build.log_factorial, MaxN+2, double);
    }
    else
    {
	Context->rule_build.log_case_count = Alloc(MaxN+2, double);
	Context->rule_build.log_factorial   = Alloc(MaxN+2, double);
    }

    Context->rule_build.log_case_count[0] = -1E38;
    Context->rule_build.log_case_count[1] = 0;

    Context->rule_build.log_factorial[0] = Context->rule_build.log_factorial[1] = 0;

    ForEach(i, 2, MaxN+1)
    {
	Context->rule_build.log_case_count[i] = Log((double) i);
	Context->rule_build.log_factorial[i]   = Context->rule_build.log_factorial[i-1] + Context->rule_build.log_case_count[i];
    }
}



void FreeSiftRuleData(c50_context *Context)
/*   ----------------  */
{
    FreeUnlessNil(Context->rule_build.list);				Context->rule_build.list = Nil;
    FreeVector((void **) Context->rule_build.fires, 1, Context->rules.capacity-1);	Context->rule_build.fires = Nil;
    FreeUnlessNil(Context->rule_build.compression_buffer);				Context->rule_build.compression_buffer = Nil;
    FreeUnlessNil(Context->rule_selection.covered_cases);				Context->rule_selection.covered_cases = Nil;
    FreeUnlessNil(Context->rule_selection.rules_included);				Context->rule_selection.rules_included = Nil;
    FreeUnlessNil(Context->rule_build.coverage_counts);				Context->rule_build.coverage_counts = Nil;
    FreeUnlessNil(Context->rule_selection.coverage_pointers);				Context->rule_selection.coverage_pointers = Nil;
    FreeUnlessNil(Context->rule_build.branch_bits);				Context->rule_build.branch_bits = Nil;
    FreeUnlessNil(Context->rule_build.attribute_values);				Context->rule_build.attribute_values = Nil;

    FreeUnlessNil(Context->rule_selection.delta_errors);				Context->rule_selection.delta_errors = Nil;
    FreeUnlessNil(Context->rule_selection.coverage_block);				Context->rule_selection.coverage_block = Nil;
    FreeUnlessNil(Context->rule_selection.rule_bits);				Context->rule_selection.rule_bits = Nil;
    FreeUnlessNil(Context->rule_selection.top_classes);				Context->rule_selection.top_classes = Nil;
    FreeUnlessNil(Context->rule_selection.alternate_classes);				Context->rule_selection.alternate_classes = Nil;
    if ( Context->rule_selection.total_votes )
    {
	FreeUnlessNil(Context->rule_selection.total_votes[0]);
	FreeUnlessNil(Context->rule_selection.total_votes);				Context->rule_selection.total_votes = Nil;
    }
}
