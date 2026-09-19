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
/*								  	 */
/*	Form a set of rules from a decision tree			 */
/*	----------------------------------------			 */
/*								  	 */
/*	The cases are partitioned into sublists:			 */
/*	  * Context->rule_build.fail_zero: those cases that satisfy all undeleted conditions	 */
/*	  * Context->rule_build.fail_one: those that satisfy all but one of the above		 */
/*	  * Context->rule_build.fail_many: the remaining cases				 */
/*	Lists are implemented via Context->rule_build.successors; Context->rule_build.successors[i] is the number of the	 */
/*	case that follows case i.					 */
/*									 */
/*************************************************************************/


#include "defns.i"
#include "extern.i"
#include "c50_api_internal.h"


/*************************************************************************/
/*								  	 */
/*	Process a tree to extract a ruleset				 */
/*								  	 */
/*************************************************************************/


CRuleSet FormRules(c50_context *Context, Tree T)
    /*	 ---------  */
{
    int		i;
    CRuleSet	RS;

    NotifyStage(Context, FORMRULES);
    Progress(Context, -(Context->cases.max_case+1.0));

    Verbosity(2, PrintTree(Context, T, "Context->trees.pruned tree:"))

    /*  Find essential parameters and allocate storage  */

    Context->rule_build.max_rule_depth = TreeDepth(T);

    Context->rule_build.condition_errors	 = AllocZero(Context->rule_build.max_rule_depth+2, double);
    Context->rule_build.condition_totals	 = AllocZero(Context->rule_build.max_rule_depth+2, double);

    Context->rule_build.pessimistic_errors	 = AllocZero(Context->rule_build.max_rule_depth+2, float);
    Context->rule_build.condition_costs	 = AllocZero(Context->rule_build.max_rule_depth+2, float);

    Context->rule_build.condition_failed_by = AllocZero(Context->rule_build.max_rule_depth+2, Boolean *);
    Context->rule_build.deleted_conditions	 = AllocZero(Context->rule_build.max_rule_depth+2, Boolean);

    Context->rule_build.condition_stack	 = AllocZero(Context->rule_build.max_rule_depth+2, Condition);

    ForEach(i, 0, Context->rule_build.max_rule_depth+1)
    {
	Context->rule_build.condition_stack[i]	= Alloc(1, CondRec);
	Context->rule_build.condition_failed_by[i] = AllocZero(Context->cases.max_case+1, Boolean);
    }

    Context->rule_build.failure_count	 = AllocZero(Context->cases.max_case+1, short);
    Context->rule_build.local_failure_count	 = AllocZero(Context->cases.max_case+1, short);

    Context->rule_build.coverage_counts	 = AllocZero(Context->cases.max_case+2, int);

    Context->rule_build.list	 = Alloc(Context->cases.max_case+2, CaseNo);
    Context->rule_build.successors	 = Alloc(Context->cases.max_case+1, CaseNo);

    Context->rule_build.compression_buffer	 = Alloc(4 + (Context->cases.max_case+1) + (Context->cases.max_case+1)/128, Byte);

    Context->rules.count = Context->rules.capacity = 0;
    FindClassFreq(Context, Context->training.class_frequencies, 0, Context->cases.max_case);

    if ( ! Context->rule_build.branch_bits )
    {
	GenerateLogs(Context,
	    Max(Context->cases.max_case+1,
		Max(Context->schema.max_attribute,
		    Max(Context->schema.max_class,
			Context->schema.max_discrete_value))));
	FindTestCodes(Context);
    }

    SetupNCost(Context);

    /*  Extract and prune paths from root to leaves  */

    Context->rule_build.condition_count = 0;
    Scan(Context, T);

    Context->default_class = T->Leaf;

    /*  Deallocate storage  */

    FreeFormRuleData(Context);

    /*  Select final rules  */

    SiftRules(Context,
	      (T->Errors + Context->schema.max_class-1) /
	      (Context->cases.max_case+1 + Context->schema.max_class));

    FreeVector((void **) Context->costs.normalized_matrix, 0, Context->schema.max_class);		Context->costs.normalized_matrix = Nil;

    CheckActiveSpace(Context, Context->rules.count);

    RS = Alloc(1, RuleSetRec);

    RS->SNRules  = Context->rules.count;
    RS->SRule    = Context->rules.rules;				Context->rules.rules = Nil;
    RS->SDefault = Context->default_class;

    ConstructRuleTree(Context, RS);

    return RS;
}



/*************************************************************************/
/*								  	 */
/*	Set up normalised costs.  These are all 0/1 if Context->costs.matrix is not	 */
/*	defined or if cost weighting is used.  Otherwise, Context->costs.matrix is	 */
/*	divided by an estimated average error cost, determined as	 */
/*	follows:							 */
/*									 */
/*	Assume E errors.  The expected number of cases misclassified	 */
/*	as class C is E * P(C), with the real classes distributed	 */
/*	in accordance with their priors.  This gives an expected	 */
/*	total error cost of						 */
/*	    E * sum/C { P(C) * sum/D!=C { P(D)/(1-P(C)) * M[C][D] } }	 */
/*	and dividing by E gives an expected average cost.		 */
/*									 */
/*	The above tends to be pessimistic, so we reduce it somewhat.	 */
/*								  	 */
/*	Siftrules requires a row of Context->costs.normalized_matrix corresponding to predicted	 */
/*	class 0 (case not covered by any rule).  All costs in this row	 */
/*	are set to 1.							 */
/*								  	 */
/*************************************************************************/


void SetupNCost(c50_context *Context)
/*   ----------  */
{
    ClassNo	Real, Pred;
    double	AvErrCost=0, ProbPred, ProbReal;

    Context->costs.normalized_matrix = Alloc(Context->schema.max_class+1, float *);

    ForEach(Pred, 0, Context->schema.max_class)
    {
	Context->costs.normalized_matrix[Pred] = Alloc(Context->schema.max_class+1, float);

	if ( ! Context->costs.matrix || Context->costs.weighted || Pred == 0 )
	{
	    ForEach(Real, 1, Context->schema.max_class)
	    {
		Context->costs.normalized_matrix[Pred][Real] = ( Pred != Real );
	    }
	}
	else
 	{
	    ProbPred = Context->training.class_frequencies[Pred] / (Context->cases.max_case+1);
	    ForEach(Real, 1, Context->schema.max_class)
	    {
		Context->costs.normalized_matrix[Pred][Real] = Context->costs.matrix[Pred][Real];
		if ( Real == Pred ) continue;

		ProbReal = Context->training.class_frequencies[Real] / (Context->cases.max_case+1);
		AvErrCost +=
		    ProbPred * (ProbReal / (1 - ProbPred)) * Context->costs.matrix[Pred][Real];
	    }
	}
    }

    if ( Context->costs.matrix && ! Context->costs.weighted )
    {
	AvErrCost = (AvErrCost + 1) / 2;	/* reduced average cost */
	ForEach(Real, 1, Context->schema.max_class)
	{
	    ForEach(Pred, 1, Context->schema.max_class)
	    {
		Context->costs.normalized_matrix[Pred][Real] /= AvErrCost;
	    }
	}
    }
}



/*************************************************************************/
/*								  	 */
/*	Extract paths from tree T and prune them to form rules		 */
/*								  	 */
/*************************************************************************/


void Scan(c50_context *Context, Tree T)
/*   ----  */
{
    DiscrValue	v, Last;
    Condition	Term;

    if ( T->NodeType )
    {
	Context->rule_build.condition_count++;
	Term = Context->rule_build.condition_stack[Context->rule_build.condition_count];

	Term->NodeType = T->NodeType;
	Term->Tested   = T->Tested;
	Term->Cut      = T->Cut;

	ForEach(v, 1, T->Forks)
	{
	    /*  Skip branches with empty leaves  */

	    if ( T->Branch[v]->Cases < MinLeaf ) continue;

	    Term->TestValue = v;

	    if ( T->NodeType == BrSubset )
	    {
		if ( Elements(Context, T->Tested, T->Subset[v], &Last) == 1 )
		{
		    /*  Subset contains a single element  */

		    Term->NodeType  = BrDiscr;
		    Term->TestValue = Last;
		}
		else
		{
		    Term->NodeType  = BrSubset;
		    Term->Subset    = T->Subset[v];
		    Term->TestValue = 1;
		}
	    }

	    Context->rule_build.condition_costs[Context->rule_build.condition_count] = CondBits(Context, Term);

	    /*  Adjust number of failed conditions  */

	    PushCondition(Context);

	    Scan(Context, T->Branch[v]);

	    /*  Reset number of failed conditions  */

	    PopCondition(Context);
	}

	Context->rule_build.condition_count--;
    }

    /*  Make a rule from every node of the tree other than the root  */

    if ( Context->rule_build.condition_count > 0 && T->Cases >= 1 )
    {

	memcpy(Context->rule_build.local_failure_count, Context->rule_build.failure_count, (Context->cases.max_case + 1) * sizeof(short));

	Context->rule_build.target_class = T->Leaf;
	PruneRule(Context, Context->rule_build.condition_stack);

	if ( ! T->NodeType ) Progress(Context, T->Cases);
    }
}



/*************************************************************************/
/*								  	 */
/*	Update Context->rule_build.failure_count when a condition is added to/removed from Context->rule_build.condition_stack	 */
/*								  	 */
/*************************************************************************/


void PushCondition(c50_context *Context)
/*   -------------  */
{
    int i;

    ForEach(i, 0, Context->cases.max_case)
    {
	if ( (Context->rule_build.condition_failed_by[Context->rule_build.condition_count][i] =
	      ! Satisfies(Context, Context->cases.records[i], Context->rule_build.condition_stack[Context->rule_build.condition_count])) )
	{
	    Context->rule_build.failure_count[i]++;
	}
    }
}



void PopCondition(c50_context *Context)
/*   -------------  */
{
    int i;

    ForEach(i, 0, Context->cases.max_case)
    {
	if ( Context->rule_build.condition_failed_by[Context->rule_build.condition_count][i] )
	{
	    Context->rule_build.failure_count[i]--;
	}
    }
}



/*************************************************************************/
/*									 */
/*	Prune the rule given by the conditions Cond, and the number of	 */
/*	conditions Context->rule_build.condition_count, and add the resulting rule to the current	 */
/*	ruleset if it is sufficiently accurate				 */
/*									 */
/*************************************************************************/

#define TI(a,b)		(((a)+(b)) * Log((a)+(b)) - (a) * Log(a) - (b) * Log(b))


void PruneRule(c50_context *Context, Condition Cond[])
/*   ---------  */
{
    int		d, id, Bestid, Remaining=Context->rule_build.condition_count;
    double	RealTotal, RealCorrect;
    CaseNo	i, LL=0;
    float	Prior;
    double	Base, Gain, Cost=0;

    (void) Bestid;  /* Used only when VerbOpt is enabled. */

    ForEach(d, 0, Context->rule_build.condition_count)
    {
	Context->rule_build.deleted_conditions[d] = false;
	Context->rule_build.condition_totals[d]   =
	Context->rule_build.condition_errors[d]  = 0;

	if ( d ) Cost += Context->rule_build.condition_costs[d];
    }
    Cost -= Context->rule_build.log_factorial[Context->rule_build.condition_count];

    Base = TI(Context->training.class_frequencies[Context->rule_build.target_class], Context->cases.max_case+1 - Context->training.class_frequencies[Context->rule_build.target_class]);

    /*  Initialise all fail lists  */

    Context->rule_build.best_condition = 0;
    ProcessLists(Context);

    ForEach(d, 1, Context->rule_build.condition_count)
    {
	Context->rule_build.condition_totals[d]  += Context->rule_build.condition_totals[0];
	Context->rule_build.condition_errors[d] += Context->rule_build.condition_errors[0];
    }

    /*  Find conditions to delete  */

    Verbosity(1, fprintf(Of, "\n  Pruning rule for %s", Context->schema.class_names[Context->rule_build.target_class]))

    while (true )
    {
	/*  Find the condition, deleting which would most improve
	    the pessimistic accuracy of the rule.
	    Note: d = 0 means all conditions are satisfied  */

	Context->rule_build.best_condition = id = 0;

	Gain = Base - TI(Context->rule_build.condition_totals[0]-Context->rule_build.condition_errors[0], Context->rule_build.condition_errors[0])
		    - TI(Context->training.class_frequencies[Context->rule_build.target_class]-Context->rule_build.condition_totals[0]+Context->rule_build.condition_errors[0],
			 Context->cases.max_case+1-Context->training.class_frequencies[Context->rule_build.target_class]-Context->rule_build.condition_errors[0]);

	Verbosity(1,
	    fprintf(Of, "\n       Err   Used   Pess\tAbsent condition\n"))

	ForEach(d, 0, Context->rule_build.condition_count)
	{
	    if ( Context->rule_build.deleted_conditions[d] ) continue;

	    if ( Context->rule_build.condition_errors[d] > Context->rule_build.condition_totals[d] ) Context->rule_build.condition_errors[d] = Context->rule_build.condition_totals[d];

	    Context->rule_build.pessimistic_errors[d] = ( Context->rule_build.condition_totals[d] < Epsilon ? 0.5 :
			       (Context->rule_build.condition_errors[d] + 1) / (Context->rule_build.condition_totals[d] + 2.0) );

	    Verbosity(1,
		fprintf(Of, "   %7.1f%7.1f  %4.1f%%",
		       Context->rule_build.condition_errors[d], Context->rule_build.condition_totals[d], 100 * Context->rule_build.pessimistic_errors[d]))

	    if ( ! d )
	    {
		Verbosity(1,
		    fprintf(Of, "\t<base> %.1f/%.1f bits\n", Gain, Cost))
	    }
	    else
	    {
		id++;

		Verbosity(1, PrintCondition(Context, Cond[d]))

		/*  Context->rule_build.best_condition identifies the condition with lowest pessimistic
		    error  estimate  */

		if ( ! Context->rule_build.best_condition || Context->rule_build.pessimistic_errors[d] <= Context->rule_build.pessimistic_errors[Context->rule_build.best_condition] )
		{
		    Context->rule_build.best_condition  = d;
		    Bestid = id;
		}
	    }
	}

	if ( Remaining == 1 || ! Context->rule_build.best_condition ||
	     ( THEORYFRAC * Cost <= Gain &&
	       Context->rule_build.pessimistic_errors[Context->rule_build.best_condition] > Context->rule_build.pessimistic_errors[0] ) )
	{
	    break;
	}

	Verbosity(1, fprintf(Of, "\teliminate test %d\n", Bestid))

	Context->rule_build.deleted_conditions[Context->rule_build.best_condition] = true;
	Remaining--;
	Cost -= Context->rule_build.condition_costs[Context->rule_build.best_condition] - Context->rule_build.log_factorial[Remaining+1] + Context->rule_build.log_factorial[Remaining];

	ForEach(d, 1, Context->rule_build.condition_count)
	{
	    if ( d != Context->rule_build.best_condition )
	    {
		Context->rule_build.condition_totals[d]  += Context->rule_build.condition_totals[Context->rule_build.best_condition] - Context->rule_build.condition_totals[0];
		Context->rule_build.condition_errors[d] += Context->rule_build.condition_errors[Context->rule_build.best_condition] - Context->rule_build.condition_errors[0];
	    }
	}
	Context->rule_build.condition_totals[0]  = Context->rule_build.condition_totals[Context->rule_build.best_condition];
	Context->rule_build.condition_errors[0] = Context->rule_build.condition_errors[Context->rule_build.best_condition];

	ProcessLists(Context);
    }

    if ( Remaining && Context->rule_build.condition_totals[0] > 0.99 && THEORYFRAC * Cost <= Gain )
    {
	Prior = Context->training.class_frequencies[Context->rule_build.target_class] / (Context->cases.max_case+1.0);

	/*  Find list of cases covered by this rule and adjust coverage
	    if using costs  */

	if ( ! Context->costs.matrix )
	{
	    RealTotal   = Context->rule_build.condition_totals[0];
	    RealCorrect = Context->rule_build.condition_totals[0] - Context->rule_build.condition_errors[0];

	    for ( i = Context->rule_build.fail_zero ; i >= 0 ; i = Context->rule_build.successors[i] )
	    {
		Context->rule_build.list[++LL] = i;
	    }
	}
	else
	if ( Context->costs.weighted )
	{
	    /*  Adjust distributions to reverse case weighting  */

	    Prior /= Context->costs.weight_multipliers[Context->rule_build.target_class];

	    RealTotal = 0;
	    for ( i = Context->rule_build.fail_zero ; i >= 0 ; i = Context->rule_build.successors[i] )
	    {
		RealTotal += Weight(Context->cases.records[i]) / Context->costs.weight_multipliers[Class(Context->cases.records[i])];
		Context->rule_build.list[++LL] = i;
	    }
	    RealCorrect = (Context->rule_build.condition_totals[0] - Context->rule_build.condition_errors[0]) / Context->costs.weight_multipliers[Context->rule_build.target_class];
	}
	else
	{
	    /*  Context->rule_build.condition_errors have been weighted by Context->costs.normalized_matrix -- undo  */

	    RealTotal   = Context->rule_build.condition_totals[0];
	    RealCorrect = 0;
	    for ( i = Context->rule_build.fail_zero ; i >= 0 ; i = Context->rule_build.successors[i] )
	    {
		RealCorrect += Weight(Context->cases.records[i]) *
			       (Class(Context->cases.records[i]) == Context->rule_build.target_class);
		Context->rule_build.list[++LL] = i;
	    }
	}
	Context->rule_build.list[0] = LL;

	if ( (RealCorrect + 1) / ((RealTotal + 2) * Prior) >= 0.95 )
	{
	    NewRule(Context, Cond, Context->rule_build.condition_count, Context->rule_build.target_class, Context->rule_build.deleted_conditions, Nil,
		    RealTotal, RealCorrect, Prior);
	}
    }
}



/*************************************************************************/
/*								  	 */
/*	Change Context->rule_build.fail_zero, Context->rule_build.fail_one, and Context->rule_build.fail_many.				 */
/*	If Context->rule_build.best_condition has not been set, initialise the lists; otherwise	 */
/*	record the changes for deleting condition Context->rule_build.best_condition and reduce	 */
/*	Context->rule_build.local_failure_count for cases that do not satisfy condition Context->rule_build.best_condition	 */
/*								  	 */
/*************************************************************************/


void ProcessLists(c50_context *Context)
/*   ------------  */
{
    CaseNo	i, iNext, *Prev;
    int		d;

    if ( ! Context->rule_build.best_condition )
    {
	/*  Initialise the fail list */

	Context->rule_build.fail_zero = Context->rule_build.fail_one = Context->rule_build.fail_many = -1;

	ForEach(i, 0, Context->cases.max_case)
	{
	    if ( ! Context->rule_build.local_failure_count[i] )
	    {
		Increment(Context, 0, i, Context->rule_build.condition_totals, Context->rule_build.condition_errors);
		AddToList(Context, &Context->rule_build.fail_zero, i);
	    }
	    else
	    if ( Context->rule_build.local_failure_count[i] == 1 )
	    {
		d = SingleFail(Context, i);
		Increment(Context, d, i, Context->rule_build.condition_totals, Context->rule_build.condition_errors);
		AddToList(Context, &Context->rule_build.fail_one, i);
	    }
	    else
	    {
		AddToList(Context, &Context->rule_build.fail_many, i);
	    }
	}
    }
    else
    {
	/*  Change the fail list to remove condition Context->rule_build.best_condition  */

	/*  Promote cases from Context->rule_build.fail_one to Context->rule_build.fail_zero  */

	Prev = &Context->rule_build.fail_one;

	for ( i = Context->rule_build.fail_one ; i >= 0 ; )
	{
	    iNext = Context->rule_build.successors[i];
	    if ( Context->rule_build.condition_failed_by[Context->rule_build.best_condition][i] )
	    {
		DeleteFromList(Context, Prev, i);
		AddToList(Context, &Context->rule_build.fail_zero, i);
	    }
	    else
	    {
		Prev = &Context->rule_build.successors[i];
	    }
	    i = iNext;
	}

	/*  Check cases in Context->rule_build.fail_many  */

	Prev = &Context->rule_build.fail_many;

	for ( i = Context->rule_build.fail_many ; i >= 0 ; )
	{
	    iNext = Context->rule_build.successors[i];
	    if ( Context->rule_build.condition_failed_by[Context->rule_build.best_condition][i] && --Context->rule_build.local_failure_count[i] == 1 )
	    {
		d = SingleFail(Context, i);
		Increment(Context, d, i, Context->rule_build.condition_totals, Context->rule_build.condition_errors);

		DeleteFromList(Context, Prev, i);
		AddToList(Context, &Context->rule_build.fail_one, i);
	    }
	    else
	    {
		Prev = &Context->rule_build.successors[i];
	    }
	    i = iNext;
	}
    }
}



/*************************************************************************/
/*								  	 */
/*	Add case to list whose first case is *Context->rule_build.list			 */
/*								  	 */
/*************************************************************************/


void AddToList(c50_context *Context, CaseNo *head, CaseNo N)
/*   ---------  */
{
    Context->rule_build.successors[N] = *head;
    *head   = N;
}



/*************************************************************************/
/*								  	 */
/*	Delete case from list where previous case is *Before		 */
/*								  	 */
/*************************************************************************/


void DeleteFromList(c50_context *Context, CaseNo *Before, CaseNo N)
/*   --------------  */
{
    *Before = Context->rule_build.successors[N];
}



/*************************************************************************/
/*								  	 */
/*	Find single condition failed by a case				 */
/*								  	 */
/*************************************************************************/


int SingleFail(c50_context *Context, CaseNo i)
/*  ----------  */
{
    int		d;

    ForEach(d, 1, Context->rule_build.condition_count)
    {
	if ( ! Context->rule_build.deleted_conditions[d] && Context->rule_build.condition_failed_by[d][i] ) return d;
    }

    return 0;
}



/*************************************************************************/
/*								  	 */
/*	Case i covers all conditions except d; update Context->rule_build.condition_totals and Context->rule_build.condition_errors	 */
/*								  	 */
/*************************************************************************/


void Increment(c50_context *Context, int d, CaseNo i,
	       double *totals, double *errors)
/*   ---------  */
{
    totals[d] += Weight(Context->cases.records[i]);
    errors[d] += Weight(Context->cases.records[i]) *
	Context->costs.normalized_matrix[Context->rule_build.target_class]
					 [Class(Context->cases.records[i])];
}






/*************************************************************************/
/*								  	 */
/*	Free all data allocated for forming rules			 */
/*								  	 */
/*************************************************************************/


void FreeFormRuleData(c50_context *Context)
/*   ----------------  */
{
    if ( ! Context->rule_build.condition_failed_by ) return;

    FreeVector((void **) Context->rule_build.condition_failed_by, 0, Context->rule_build.max_rule_depth+1);	Context->rule_build.condition_failed_by = Nil;
    FreeVector((void **) Context->rule_build.condition_stack, 0, Context->rule_build.max_rule_depth+1);		Context->rule_build.condition_stack = Nil;
    Free(Context->rule_build.deleted_conditions);					Context->rule_build.deleted_conditions = Nil;
    Free(Context->rule_build.pessimistic_errors);					Context->rule_build.pessimistic_errors = Nil;
    Free(Context->rule_build.condition_costs);					Context->rule_build.condition_costs = Nil;
    Free(Context->rule_build.condition_totals);					Context->rule_build.condition_totals = Nil;
    Free(Context->rule_build.condition_errors);					Context->rule_build.condition_errors = Nil;
    Free(Context->rule_build.failure_count);					Context->rule_build.failure_count = Nil;
    Free(Context->rule_build.local_failure_count);					Context->rule_build.local_failure_count = Nil;
    Free(Context->rule_build.successors);						Context->rule_build.successors = Nil;
}
