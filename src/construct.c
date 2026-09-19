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
/*	Manage construction of classifiers, including boosting		 */
/*	------------------------------------------------------		 */
/*									 */
/*	C5.0 uses a modified form of boosting as follows:		 */
/*	*  Multiplicative weight adjustment is used for cases that are	 */
/*	   classified correctly, but misclassified cases use a form	 */
/*	   of additive weight adjustment.				 */
/*	*  In later boosting trials, cases that cannot possibly be	 */
/*	   classified correctly are dropped.  (This follows Freund's	 */
/*	   "Brown-Boost" approach, since the number of trials is known.) */
/*	*  The voting weight of a boosted classifier is determined by	 */
/*	   the confidence of its classification, based on the boosted	 */
/*	   weights of the training cases.				 */
/*									 */
/*	Variable misclassification costs are also supported.  When	 */
/*	there are two classes, the misclassification costs are used	 */
/*	to reweight the training cases, and the reweighting is reversed	 */
/*	after the classifier is constructed.  When classifying a case,	 */
/*	the estimated class probabilities and cost matrix are used to	 */
/*	determine the predicted class with lowest expected cost.	 */
/*									 */
/*************************************************************************/


#include "defns.i"
#include "extern.i"
#include "c50_api_internal.h"

/*************************************************************************/
/*									 */
/*	Grow single tree or sequence of boosted trees			 */
/*									 */
/*************************************************************************/


void ConstructClassifiers(c50_context *Context)
/*   --------------------  */
{
    CaseNo	i, Errs, Cases, Bp, Excl=0;
    double	ErrWt, ExclWt=0, OKWt, ExtraErrWt, NFact, MinWt=1.0, a, b;
    ClassNo	c, Pred, Real, Best;
    int		BaseLeaves;
    Boolean	NoStructure, CheckExcl;
    float	*BVote;

    /*  Clean up after possible interrupt  */

    FreeUnlessNil(Context->training.wrong_predictions);

    Context->training.wrong_predictions = Alloc(Context->cases.max_case+1, ClassNo);

    if ( Context->options.trials > 1 )
    {
	/*  Context->training.boost_vote_block contains each case's class votes  */

	Context->training.boost_vote_block = AllocZero((Context->cases.max_case+1) * (Context->schema.max_class+1), float);
    }

    /*  Preserve original case order  */

    Context->cases.saved_records = Alloc(Context->cases.max_case+1, DataRec);
    memcpy(Context->cases.saved_records, Context->cases.records,
	   (Context->cases.max_case+1) * sizeof(DataRec));

    /*  If using case weighting, find average  */

    if ( Context->schema.case_weight_attribute )
    {
	SetAvCWt(Context);
    }

    InitialiseWeights(Context);

    /*  Adjust minimum weight if using cost weighting  */

    if ( Context->costs.weighted )
    {
	ForEach(c, 1, Context->schema.max_class)
	{
	    if ( Context->costs.weight_multipliers[c] < MinWt ) MinWt = Context->costs.weight_multipliers[c];
	}
    }

    Context->options.leaf_ratio = Bp = 0;
    SetMinGainThresh(Context);

    /*  Main loop for growing the sequence of boosted classifiers  */

    ForEach(Context->trees.trial, 0, Context->options.trials-1 )
    {
	if ( Context->options.trials > 1 )
	{
	    fprintf(Context->io.output, "\n-----  " F_Trial " %d:  -----\n", Context->trees.trial);
	}

	NotifyStage(Context, FORMTREE);
	Progress(Context, -(Context->cases.max_case+1.0));

	/*  Update count here in case tree construction is interrupted  */

	Context->trees.max_tree = Context->trees.trial;
	Context->trees.raw[Context->trees.max_tree] = Context->trees.pruned[Context->trees.max_tree] = Nil;
	if ( Context->options.rules ) Context->rules.sets[Context->trees.max_tree] = Nil;

	memset(Context->splits.tested_attributes, 0, Context->schema.max_attribute+1);		/* reset tested attributes */

	FormTree(Context, Bp, Context->cases.max_case, 0, &Context->trees.raw[Context->trees.trial]);

	/*  Prune the raw tree to minimise expected misclassification cost  */

	Verbosity(1, if ( ! Context->options.rules )
	    PrintTree(Context, Context->trees.raw[Context->trees.trial], "Before pruning:"))

	NotifyStage(Context, SIMPLIFYTREE);
	Progress(Context, -(Context->cases.max_case+1));

	/*  If still need raw tree, copy it; otherwise set initial
	    pruned tree to raw tree  */

	if ( Context->options.verbosity && ! Context->options.rules )
	{
	    Context->trees.pruned[Context->trees.trial] = CopyTree(Context, Context->trees.raw[Context->trees.trial]);
	    if ( Context->costs.matrix )
	    {
		RestoreDistribs(Context, Context->trees.raw[Context->trees.trial]);
	    }
	}
	else
	{
	    Context->trees.pruned[Context->trees.trial] = Context->trees.raw[Context->trees.trial];
	    Context->trees.raw[Context->trees.trial] = Nil;
	}

	memcpy(Context->cases.records, Context->cases.saved_records,
	       (Context->cases.max_case+1) * sizeof(DataRec)); /* restore */

	Prune(Context, Context->trees.pruned[Context->trees.trial]);

	AdjustAllThresholds(Context, Context->trees.pruned[Context->trees.trial]);

	/*  Record tree parameters for later  */

	if ( ! Context->trees.trial )
	{
	    BaseLeaves = ( Context->options.rules || Context->options.subset_splits ? TreeSize(Context->trees.pruned[0]) :
					     ExpandedLeafCount(Context,
						       Context->trees.pruned[0]) );
	}
	NoStructure = ! Context->trees.pruned[Context->trees.trial]->NodeType;

	if ( Context->options.probabilistic_thresholds )
	{
	    SoftenThresh(Context, Context->trees.pruned[Context->trees.trial]);
	}

	memcpy(Context->cases.records, Context->cases.saved_records,
	       (Context->cases.max_case+1) * sizeof(DataRec)); /* restore */

	if ( Context->options.rules )
	{
	    Context->rules.sets[Context->trees.trial] = FormRules(Context, Context->trees.pruned[Context->trees.trial]);
	    NoStructure |= ! Context->rules.sets[Context->trees.trial]->SNRules;

	    PrintRules(Context, Context->rules.sets[Context->trees.trial], T_Rules);
	    fprintf(Context->io.output, "\n" T_Default_class ": %s\n",
			Context->schema.class_names[Context->rules.sets[Context->trees.trial]->SDefault]);

	    FreeTree(Context->trees.pruned[Context->trees.trial]);			Context->trees.pruned[Context->trees.trial] = Nil;
	}
	else
	{
	    PrintTree(Context, Context->trees.pruned[Context->trees.trial], T_Tree);
	}

	if ( Context->trees.trial == Context->options.trials-1 ) continue;

	/*  Check errors, adjust boost voting, and shift dropped cases
	    to the front  */

	ErrWt = Errs = OKWt = Bp = 0;
	CheckExcl = ( Context->trees.trial+1 > Context->options.trials / 2.0 );

	ForEach(i, 0, Context->cases.max_case)
	{
	    /*  Has this case been dropped already?  */

	    if ( Weight(Context->cases.records[i]) <= 0 )
	    {
		Context->cases.records[i]  = Context->cases.records[Bp];
		Context->training.wrong_predictions[i] = Context->training.wrong_predictions[Bp];
		Bp++;
		continue;
	    }

	    Pred = ( Context->options.rules ? RuleClassify(Context, Context->cases.records[i], Context->rules.sets[Context->trees.trial]) :
		     TreeClassify(Context, Context->cases.records[i], Context->trees.pruned[Context->trees.trial]) );

	    Real = Class(Context->cases.records[i]);

	    /*  Update boosting votes for this case.  (Note that cases
		must have been reset to their original order.)  */

	    BVote = Context->training.boost_vote_block + i * (Context->schema.max_class+1);
	    BVote[Pred] += Context->confidence;

	    Best = BVote[0];
	    if ( BVote[Pred] > BVote[Best] ) BVote[0] = Best = Pred;

	    if ( CheckExcl )
	    {
		/*  Check whether this case should be dropped because
		    the vote for the correct class cannot be increased
		    sufficiently in the remaining trials  */

		if ( BVote[Best] > BVote[Real] + (Context->options.trials-1) - Context->trees.trial )
		{
		    Excl++;
		    ExclWt += Weight(Context->cases.records[i]);

		    Weight(Context->cases.records[i]) = 0;
		    Context->cases.records[i]  = Context->cases.records[Bp];
		    Context->training.wrong_predictions[i] = Context->training.wrong_predictions[Bp];
		    Bp++;

		    continue;
		}
	    }

	    if ( Pred != Real )
	    {
		Context->training.wrong_predictions[i] = Pred;
		ErrWt   += Weight(Context->cases.records[i]);
		Errs++;
	    }
	    else
	    {
		Context->training.wrong_predictions[i] = 0;
		OKWt    += Weight(Context->cases.records[i]);
	    }
	}

	Cases  = (Context->cases.max_case+1) - Excl;

	/*  Special termination conditions  */

	if ( ErrWt < 0.1 )
	{
	    Context->options.trials = Context->trees.trial + 1;
	    fprintf(Context->io.output, TX_Reduced1(Context->options.trials), Context->options.trials);
	}
	else
	if ( ( Context->trees.trial && NoStructure ) || ErrWt / Cases >= 0.49 )
	{
	    Context->options.trials = ( Context->trees.trial ? Context->trees.trial : 1 );
	    fprintf(Context->io.output, TX_Reduced2(Context->options.trials), Context->options.trials);
	}
	else
	{
	    /*  Adjust case weights.  Total weight of misclassified cases
		set to midway between current weight and half total weight.
		Take account of any dropped cases  */

	    ExtraErrWt = 0.25 * (OKWt - ErrWt);		/* half */
	    a = (OKWt - ExtraErrWt) / OKWt;
	    b = ExtraErrWt / Errs;

	    /*  Normalise so that sum of weights equals number of cases  */

	    NFact = Cases / (OKWt + ErrWt);

	    MinWt *= a * NFact;

	    ForEach(i, Bp, Context->cases.max_case)
	    {
		if ( Context->training.wrong_predictions[i] )
		{
		    Weight(Context->cases.records[i]) = NFact * (Weight(Context->cases.records[i]) + b);
		}
		else
		{
		    Weight(Context->cases.records[i]) *= NFact * a;

		    /*  Necessary for accumulated arithmetic errors  */

		    if ( Weight(Context->cases.records[i]) < 1E-3 ) Weight(Context->cases.records[i]) = 1E-3;
		}
	    }

	    /*  Set the leaf ratio for subsequent boosting trials.
		The idea is to constrain the tree to roughly the size
		of the initial tree by limiting the number of leaves
		per training case.  This limitation is not strict
		since even a tiny number of cases can give a leaf  */

	    if ( Context->trees.trial == 0 )
	    {
		Context->options.leaf_ratio = 1.1 * BaseLeaves / (Context->cases.max_case + 1.0);
	    }

	    /*  Trim cases for larger datasets  */

	    if ( Context->cases.max_case > 4000 && MinWt <= 0.2 )
	    {
		a = 0;
		ForEach(i, Bp, Context->cases.max_case)
		{
		    if ( Weight(Context->cases.records[i]) <= MinWt + 1E-3 )
		    {
			a += Weight(Context->cases.records[i]);
			Swap(i, Bp);
			Bp++;
		    }
		}
	    }
	}

	Context->costs.unit_weights = false;
    }

    FreeUnlessNil(Context->cases.saved_records);				Context->cases.saved_records = Nil;

    /*  Decide whether boosting should be abandoned  */

    if ( Context->options.boosting && Context->options.trials <= 2 )
    {
	fprintf(Context->io.output, T_Abandoned);
	Context->options.trials = 1;
    }

    /*  Save trees or rulesets  */

    if ( ! Context->options.cross_validation )
    {
	if ( ! Context->options.rules )
	{
	    ForEach(Context->trees.trial, 0, Context->options.trials-1)
	    {
		SaveTree(Context, Context->trees.pruned[Context->trees.trial], ".tree");
	    }
	}
	else
	{
	    ForEach(Context->trees.trial, 0, Context->options.trials-1)
	    {
		SaveRules(Context, Context->rules.sets[Context->trees.trial], ".rules");
	    }
	}

	fclose(Context->io.model_file);
    }
    Context->io.model_file = 0;

    Free(Context->training.wrong_predictions);					Context->training.wrong_predictions = Nil;
    FreeUnlessNil(Context->training.boost_vote_block);				Context->training.boost_vote_block = Nil;
}



/*************************************************************************/
/*								 	 */
/*	Initialise the weight of each case				 */
/*								 	 */
/*************************************************************************/


void InitialiseWeights(c50_context *Context)
/*   -----------------  */
{
    CaseNo	i;

    if ( Context->costs.weighted )
    {
	/*  Make weights proportional to average error cost  */

	ForEach(i, 0, Context->cases.max_case)
	{
	    Weight(Context->cases.records[i]) = Context->costs.weight_multipliers[Class(Context->cases.records[i])];
	}
	Context->costs.unit_weights = false;
    }
    else
    {
	ForEach(i, 0, Context->cases.max_case)
	{
	    Weight(Context->cases.records[i]) = 1.0;
	}
	Context->costs.unit_weights = true;
    }

    /*  Adjust when using case weights  */

    if ( Context->schema.case_weight_attribute )
    {
	ForEach(i, 0, Context->cases.max_case)
	{
	    Weight(Context->cases.records[i]) *= RelCWt(Context, Context->cases.records[i]);
	}
	Context->costs.unit_weights = false;
    }
}



/*************************************************************************/
/*								 	 */
/*	Determine average case weight, ignoring cases with unknown,	 */
/*	non-applicable, or negative values of Context->schema.case_weight_attribute.			 */
/*								 	 */
/*************************************************************************/


void SetAvCWt(c50_context *Context)
/*   --------  */
{
    CaseNo	i, NCWt=0;
    ContValue	CWt;

    Context->average_case_weight = 0;
    ForEach(i, 0, Context->cases.max_case)
    {
	if ( ! NotApplic(Context, Context->cases.records[i], Context->schema.case_weight_attribute) && ! Unknown(Context->cases.records[i], Context->schema.case_weight_attribute) &&
	     (CWt = CVal(Context->cases.records[i], Context->schema.case_weight_attribute)) > 0 )
	{
	    NCWt++;
	    Context->average_case_weight += CWt;
	}
    }

    Context->average_case_weight = ( NCWt > 0 ? Context->average_case_weight / NCWt : 1 );
}



/*************************************************************************/
/*									 */
/*	Print report of errors for each of the trials			 */
/*									 */
/*************************************************************************/

void Evaluate(c50_context *Context, int Flags)
/*   --------  */
{
    if ( Context->options.trials == 1 )
    {
	EvaluateSingle(Context, Flags);
    }
    else
    {
	EvaluateBoost(Context, Flags);
    }
}



void EvaluateSingle(c50_context *Context, int Flags)
/*   --------------  */
{
    ClassNo	RealClass, PredClass;
    int		x, u, SaveUtility;
    CaseNo	*ConfusionMat, *Usage, i, RawErrs=0, Errs=0;
    double	ECost=0, Tests;
    Boolean	CMInfo, UsageInfo;
    const char	*StdR[] = { "   Before Pruning   ",
			    "  ----------------  ", "  " F_SizeErrors "  " };
    const char	*StdP[] = { "  " F_DecisionTree16 "  ",
			    "  ----------------  ", "  " F_SizeErrors "  " };
    const char	*StdPC[] = { "  " F_DecisionTree23 "  ",
			     "  -----------------------  ",
			     "  " F_SizeErrorsCost "  " };
    const char	*Extra[] = { "  " F_Rules16, "  ----------------",
			     "  " F_NoErrors };
    const char	*ExtraC[] = { "  " F_Rules23,
			      "  -----------------------",
			      "  " F_NoErrorsCost };

    (void) RawErrs;  /* Used only when VerbOpt is enabled. */

    CMInfo    = Flags & CMINFO;
    UsageInfo = Flags & USAGEINFO;

    if ( CMInfo )
    {
	ConfusionMat = AllocZero((Context->schema.max_class+1)*(Context->schema.max_class+1), CaseNo);
    }

    if ( UsageInfo )
    {
	Usage = AllocZero(Context->schema.max_attribute+1, CaseNo);
    }

    Tests = Max(Context->cases.max_case+1, 1);	/* in case no useful test data! */

    if ( Context->options.utility_bands && Context->options.rules )
    {
	SaveUtility = Context->options.utility_bands;

	Context->options.utility_bands = Min(Context->options.utility_bands, Context->rules.sets[0]->SNRules);

	Context->evaluation.utility_errors  = AllocZero(Context->options.utility_bands, int);
	Context->evaluation.utility_bands = Alloc(Context->options.utility_bands, int);
	if ( Context->costs.matrix )
	{
	    Context->evaluation.utility_costs = AllocZero(Context->options.utility_bands, double);
	}

	ForEach(u, 1, Context->options.utility_bands-1)
	{
	    Context->evaluation.utility_bands[u] = rint(Context->rules.sets[0]->SNRules * u / (float) Context->options.utility_bands);
	}
    }
	    
    fprintf(Context->io.output, "\n");
    ForEach(x, 0, 2)
    {
	putc('\t', Context->io.output);
	if ( Context->options.rules )
	{
	    fprintf(Context->io.output, "%s", ( Context->costs.matrix ? ExtraC[x] : Extra[x] ));
	}
	else
	{
	    Verbosity(1, fprintf(Context->io.output, "%s", StdR[x]))
	    fprintf(Context->io.output, "%s", ( Context->costs.matrix ? StdPC[x] : StdP[x] ));
	}
	putc('\n', Context->io.output);
    }
    putc('\n', Context->io.output);

    ForEach(i, 0, Context->cases.max_case)
    {
	RealClass = Class(Context->cases.records[i]);
	assert(RealClass > 0 && RealClass <= Context->schema.max_class);

	memset(Context->splits.tested_attributes, 0, Context->schema.max_attribute+1);	/* for usage */

	if ( Context->options.rules )
	{
	    PredClass = RuleClassify(Context, Context->cases.records[i], Context->rules.sets[0]);
	}
	else
	{
	    Verbosity(1,
		PredClass = TreeClassify(Context, Context->cases.records[i], Context->trees.raw[0]);
		if ( PredClass != RealClass )
		{
		    RawErrs++;
		})

	    PredClass = TreeClassify(Context, Context->cases.records[i], Context->trees.pruned[0]);
	}
	assert(PredClass > 0 && PredClass <= Context->schema.max_class);

	if ( PredClass != RealClass )
	{
	    Errs++;
	    if ( Context->costs.matrix ) ECost += Context->costs.matrix[PredClass][RealClass];
	}

	if ( CMInfo )
	{
	    ConfusionMat[RealClass*(Context->schema.max_class+1)+PredClass]++;
	}

	if ( UsageInfo )
	{
	    RecordAttUsage(Context, Context->cases.records[i], Usage);
	}
    }

    putc('\t', Context->io.output);

    if ( Context->options.rules )
    {
	fprintf(Context->io.output, "  %4d %4d(%4.1f%%)",
	       Context->rules.sets[0]->SNRules, Errs, 100 * Errs / Tests);
    }
    else
    {
	/*  Results for unpruned tree  */

	Verbosity(1,
	{
	    fprintf(Context->io.output, "  %4d %4d(%4.1f%%)  ",
		   TreeSize(Context->trees.raw[0]), RawErrs, 100 * RawErrs / Tests);
	})

	/*  Results for pruned tree  */

	fprintf(Context->io.output, "  %4d %4d(%4.1f%%)",
	       TreeSize(Context->trees.pruned[0]), Errs, 100 * Errs / Tests);
    }

    if ( Context->costs.matrix )
    {
	fprintf(Context->io.output, "%7.2f", ECost / Tests);
    }

    fprintf(Context->io.output, "   <<\n");

    if ( CMInfo )
    {
	PrintConfusionMatrix(Context, ConfusionMat);
	Free(ConfusionMat);
    }

    if ( UsageInfo )
    {
	PrintUsageInfo(Context, Usage);
	Free(Usage);
    }

    if ( Context->evaluation.utility_errors )
    {
	if ( ! Context->options.cross_validation )
	{
	    fprintf(Context->io.output, "\n" T_Rule_utility_summary ":\n\n"
			"\t" F_Rules "\t      " F_Errors "%s\n"
			"\t" F_URules "\t      " F_UErrors "%s\n",
			    ( Context->costs.matrix ? "   " F_Cost : "" ),
			    ( Context->costs.matrix ? "   " F_UCost : "" ));

	    ForEach(u, 1, Context->options.utility_bands-1)
	    {
		fprintf(Context->io.output, "\t%s%d\t %4d(%4.1f%%)",
			    ( Context->evaluation.utility_bands[u] == 1 ? "" : "1-" ), Context->evaluation.utility_bands[u],
			    Context->evaluation.utility_errors[u], 100 * Context->evaluation.utility_errors[u] / Tests);
		if ( Context->costs.matrix )
		{
		    fprintf(Context->io.output, "%7.2f", Context->evaluation.utility_costs[u] / Tests);
		}
		fprintf(Context->io.output, "\n");
	    }
	}

	Free(Context->evaluation.utility_errors);					Context->evaluation.utility_errors = Nil;
	FreeUnlessNil(Context->evaluation.utility_costs);			Context->evaluation.utility_costs = Nil;
	Free(Context->evaluation.utility_bands);					Context->evaluation.utility_bands = Nil;

	Context->options.utility_bands = SaveUtility;
    }
}



void EvaluateBoost(c50_context *Context, int Flags)
/*   -------------  */
{
    ClassNo	RealClass, PredClass;
    int		t;
    CaseNo	*ConfusionMat, *Usage, i, *Errs, BoostErrs=0;
    double	*ECost, BoostECost=0, Tests;
    Boolean	CMInfo, UsageInfo;
    const char	*Multi[] = { F_Trial, F_UTrial, "" };
    const char	*StdP[] = { "  " F_DecisionTree16 "  ",
			    "  ----------------  ", "  " F_SizeErrors "  " };
    const char	*StdPC[] = { "  " F_DecisionTree23 "  ",
			     "  -----------------------  ",
			     "  " F_SizeErrorsCost "  " };
    const char	*Extra[] = { "  " F_Rules16, "  ----------------",
			     "  " F_NoErrors };
    const char	*ExtraC[] = { "  " F_Rules23,
			      "  -----------------------",
			      "  " F_NoErrorsCost };

    CMInfo    = Flags & CMINFO;
    UsageInfo = Flags & USAGEINFO;

    if ( CMInfo )
    {
	ConfusionMat = AllocZero((Context->schema.max_class+1)*(Context->schema.max_class+1), CaseNo);
    }

    if ( UsageInfo )
    {
	Usage = AllocZero(Context->schema.max_attribute+1, CaseNo);
    }

    Tests = Max(Context->cases.max_case+1, 1);	/* in case no useful test data! */
    Errs = AllocZero(Context->options.trials, CaseNo);
    ECost = AllocZero(Context->options.trials, double);

    fprintf(Context->io.output, "\n");
    ForEach(t, 0, 2)
    {
	fprintf(Context->io.output, "%s\t", Multi[t]);
	if ( Context->options.rules )
	{
	    fprintf(Context->io.output, "%s", ( Context->costs.matrix ? ExtraC[t] : Extra[t] ));
	}
	else
	{
	    fprintf(Context->io.output, "%s", ( Context->costs.matrix ? StdPC[t] : StdP[t] ));
	}
	putc('\n', Context->io.output);
    }
    putc('\n', Context->io.output);

    /*  Set global default class for boosting  */

    Context->default_class =
	( Context->options.rules ? Context->rules.sets[0]->SDefault : Context->trees.pruned[0]->Leaf );

    ForEach(i, 0, Context->cases.max_case)
    {
	RealClass = Class(Context->cases.records[i]);

	memset(Context->splits.tested_attributes, 0, Context->schema.max_attribute+1);	/* for usage */

	PredClass = BoostClassify(Context, Context->cases.records[i], Context->options.trials-1);
	if ( PredClass != RealClass )
	{
	    BoostErrs++;
	    if ( Context->costs.matrix ) BoostECost += Context->costs.matrix[PredClass][RealClass];
	}

	if ( CMInfo )
	{
	    ConfusionMat[RealClass*(Context->schema.max_class+1)+PredClass]++;
	}

	if ( UsageInfo )
	{
	    RecordAttUsage(Context, Context->cases.records[i], Usage);
	}

	/*  Keep track of results for each trial  */

	ForEach(t, 0, Context->options.trials-1)
	{
	    if ( Context->trial_predictions[t] != RealClass )
	    {
		Errs[t]++;
		if ( Context->costs.matrix ) ECost[t] += Context->costs.matrix[Context->trial_predictions[t]][RealClass];
	    }
	}
    }

    /*  Print results for individual trials  */

    ForEach(t, 0, Context->options.trials-1)
    {
	fprintf(Context->io.output, "%4d\t", t);

	if ( Context->options.rules )
	{
	    fprintf(Context->io.output, "  %4d %4d(%4.1f%%)",
		   Context->rules.sets[t]->SNRules, Errs[t], 100 * Errs[t] / Tests);
	}
	else
	{
	    fprintf(Context->io.output, "  %4d %4d(%4.1f%%)",
		   TreeSize(Context->trees.pruned[t]), Errs[t], 100 * Errs[t] / Tests);
	}

	if ( Context->costs.matrix )
	{
	    fprintf(Context->io.output, "%7.2f", ECost[t] / Tests);
	}

	putc('\n', Context->io.output);
    }

    /*  Print boosted results  */

    if ( Context->options.rules )
    {
	fprintf(Context->io.output, F_Boost "\t  %9d(%4.1f%%)",
	    BoostErrs, 100 * BoostErrs / Tests);
    }
    else
    {
	fprintf(Context->io.output, F_Boost "\t       %4d(%4.1f%%)",
		BoostErrs, 100 * BoostErrs / Tests);
    }

    if ( Context->costs.matrix )
    {
	fprintf(Context->io.output, "%7.2f", BoostECost / Tests);
    }

    fprintf(Context->io.output, "   <<\n");

    if ( CMInfo )
    {
	PrintConfusionMatrix(Context, ConfusionMat);
	Free(ConfusionMat);
    }

    if ( UsageInfo )
    {
	PrintUsageInfo(Context, Usage);
	Free(Usage);
    }

    Free(Errs);
    Free(ECost);
}



/*************************************************************************/
/*								 	 */
/*	Record atts used when classifying last case			 */
/*								 	 */
/*************************************************************************/


void RecordAttUsage(c50_context *Context, DataRec Case, int *Usage)
/*   --------------  */
{
    Attribute	Att;
    int		i;

    /*  Scan backwards to allow for information from defined attributes  */

    for ( Att = Context->schema.max_attribute ; Att > 0 ; Att-- )
    {
	if ( Context->splits.tested_attributes[Att] && ! Unknown(Case, Att) )
	{
	    Usage[Att]++;

	    if ( Context->schema.attribute_definitions[Att] )
	    {
		ForEach(i, 1, Context->schema.attribute_definition_uses[Att][0])
		{
		    Context->splits.tested_attributes[Context->schema.attribute_definition_uses[Att][i]] = true;
		}
	    }
	}
    }
}
