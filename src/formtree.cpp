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
/*								 	 */
/*    Central tree-forming algorithm					 */
/*    ------------------------------					 */
/*								 	 */
/*************************************************************************/


#include "defns.i"
#include "extern.i"
#include "c50_api_internal.h"

#include <algorithm>
#include <array>
#include <exception>
#include <future>
#include <limits>
#include <vector>

#ifdef C50_TEST_SPLIT_FAILURE
#include "split_failure_hooks.hpp"
#else
#define C50_SPLIT_TEST_POINT(Event, Worker) ((void) 0)
#define C50_SPLIT_TEST_WORKER(Worker) ((void) 0)
#endif


#define		SAMPLEUNIT	2000


void c50_split_workspace_deleter::operator()(
    SplitWorkspace *Workspace) const noexcept
{
    if ( ! Workspace ) return;
    C50_SPLIT_TEST_POINT(workspace_release, 0);

    const DiscrValue MaxValue = Workspace->MaxDiscrValue;
    FreeVector((void **) Workspace->Freq, 0, Max(3, MaxValue+1));
    free(Workspace->ValFreq);
    free(Workspace->ClassFreq);
    free(Workspace->SRec);
    free(Workspace->SortScratch);
    free(Workspace->SortKeys);
    free(Workspace->SortKeyScratch);
    free(Workspace->SubsetInfo);
    free(Workspace->SubsetEntr);
    FreeVector((void **) Workspace->MergeInfo, 1, MaxValue);
    FreeVector((void **) Workspace->MergeEntr, 1, MaxValue);
    FreeVector((void **) Workspace->WSubset, 1, MaxValue);
    delete Workspace;
}



/*************************************************************************/
/*								 	 */
/*	Allocate space for tree tables				 	 */
/*								 	 */
/*************************************************************************/


using SplitWorkspacePtr =
    std::unique_ptr<SplitWorkspace, c50_split_workspace_deleter>;


static SplitWorkspacePtr MakeSplitWorkspace(c50_context *Context)
{
    DiscrValue v;
    if ( Context->schema.max_discrete_value < 3 ||
         Context->schema.max_discrete_value >
             std::numeric_limits<DiscrValue>::max() - 2 ||
         Context->schema.max_class < 1 ||
         Context->schema.max_class ==
             std::numeric_limits<ClassNo>::max() )
    {
        Error(Context, NOMEM, "", "");
    }

    SplitWorkspacePtr Workspace(new SplitWorkspace{});
    Workspace->MaxDiscrValue = Context->schema.max_discrete_value;
    DiscrValue vMax = Max(3, Context->schema.max_discrete_value + 1);

    Workspace->Freq = Alloc(vMax + 1, double *);
    ForEach(v, 0, vMax)
    {
        Workspace->Freq[v] = Alloc(Context->schema.max_class + 1, double);
    }
    Workspace->ValFreq = Alloc(vMax, double);
    Workspace->ClassFreq = Alloc(Context->schema.max_class + 1, double);

    size_t NoCases = Context->cases.max_case < 0
                         ? 0
                         : static_cast<size_t>(Context->cases.max_case) + 1;
    if ( NoCases > std::numeric_limits<size_t>::max() / sizeof(SortRec) )
    {
        Error(Context, NOMEM, "", "");
    }
    Workspace->SRec = Alloc(NoCases, SortRec);
    if ( Context->options.stable_ties )
    {
	Workspace->SortScratch = Alloc(NoCases, SortRec);
	Workspace->SortKeys = Alloc(NoCases, C50SortKey);
	Workspace->SortKeyScratch = Alloc(NoCases, C50SortKey);
    }

    if ( Context->options.subset_splits )
    {
        DiscrValue MaxValue = Context->schema.max_discrete_value;
        Workspace->SubsetInfo = Alloc(MaxValue + 1, double);
        Workspace->SubsetEntr = Alloc(MaxValue + 1, double);
        Workspace->MergeInfo = Alloc(MaxValue + 1, double *);
        Workspace->MergeEntr = Alloc(MaxValue + 1, double *);
        Workspace->WSubset = Alloc(MaxValue + 1, Set);
        ForEach(v, 1, MaxValue)
        {
            Workspace->MergeInfo[v] = Alloc(MaxValue + 1, double);
            Workspace->MergeEntr[v] = Alloc(MaxValue + 1, double);
            Workspace->WSubset[v] = Alloc((MaxValue >> 3) + 1, Byte);
        }
    }
    return Workspace;
}


void InitialiseTreeData(c50_context *Context)
/*   ------------------  */
{
    DiscrValue	v;
    Attribute	Att;
    size_t	NoAttributes;

    Context->trees.raw	     = AllocZero(Context->options.trials+1, Tree);
    Context->trees.pruned   = AllocZero(Context->options.trials+1, Tree);

    Context->splits.tested_attributes   = AllocZero(Context->schema.max_attribute+1, Byte);

    Context->splits.gain     = AllocZero(Context->schema.max_attribute+1, float);
    Context->splits.information     = AllocZero(Context->schema.max_attribute+1, float);
    Context->splits.thresholds      = AllocZero(Context->schema.max_attribute+1, ContValue);

    Context->splits.estimated_max_gain_ratio = AllocZero(Context->schema.max_attribute+1, float);

    /*  Data for subsets  */

    if ( Context->options.subset_splits )
    {
	InitialiseBellNumbers(Context);
	Context->splits.subsets = Alloc(Context->schema.max_attribute+1, Set *);

	ForEach(Att, 1, Context->schema.max_attribute)
	{
	    if ( Discrete(Att) && Att != Context->schema.class_attribute && ! Skip(Att) )
	    {
		Context->splits.subsets[Att] = AllocZero(Context->schema.max_attribute_value[Att]+1, Set);
		ForEach(v, 0, Context->schema.max_attribute_value[Att])
		{
		    Context->splits.subsets[Att][v] = Alloc((Context->schema.max_attribute_value[Att]>>3)+1, Byte);
		}
	    }
	}
	Context->splits.subset_counts = AllocZero(Context->schema.max_attribute+1, int);
    }

    NoAttributes = ( Context->schema.max_attribute < 1 ? 0 : (size_t) Context->schema.max_attribute );
    Context->splits.discrete_attributes  = Alloc(NoAttributes, Attribute);
    Context->splits.discrete_attribute_count = 0;

    Context->splits.discrete_frequencies = AllocZero(Context->schema.max_attribute+1, double *);
    ForEach(Att, 1, Context->schema.max_attribute)
    {
	if ( Att == Context->schema.class_attribute || Skip(Att) || ! Discrete(Att) ) continue;

	Context->splits.discrete_attributes[Context->splits.discrete_attribute_count++] = Att;

	Context->splits.discrete_frequencies[Att] = Alloc(static_cast<size_t>(Context->schema.max_class) *
	                                                 static_cast<size_t>(Context->schema.max_attribute_value[Att] + 1), double);
    }

    Context->training.class_frequencies = AllocZero(Context->schema.max_class+1, double);
    Context->class_sum  = Alloc(Context->schema.max_class+1, float);

    if ( Context->options.boosting )
    {
	Context->votes      = Alloc(Context->schema.max_class+1, float);
	Context->trial_predictions = Alloc(Context->options.trials, ClassNo);
    }

    if ( Context->options.rules )
    {
	Context->most_specific_rules     = Alloc(Context->schema.max_class+1, CRule);
	Context->splits.possible_cuts = Alloc(Context->schema.max_attribute+1, int);
    }

    /*  Check whether all attributes have many discrete values  */

    Context->splits.all_attributes_multi_valued = true;
    if ( ! Context->options.subset_splits )
    {
	for ( Att = 1 ; Context->splits.all_attributes_multi_valued && Att <= Context->schema.max_attribute ; Att++ )
	{
	    if ( ! Skip(Att) && Att != Context->schema.class_attribute )
	    {
		Context->splits.all_attributes_multi_valued = Context->schema.max_attribute_value[Att] >= 0.3 * (Context->cases.max_case + 1);
	    }
	}
    }

    /*  See whether there are continuous attributes for subsampling  */

    Context->splits.use_subsampling = false;

    /*  Set parameters for RawExtraErrs() */

    InitialiseExtraErrs(Context);

    /*  Set up environment  */

    Context->training.environment = MakeSplitWorkspace(Context);
    Context->splits.waiting_attributes = Alloc(Context->schema.max_attribute+1, Attribute);
}


void FreeTreeData(c50_context *Context)
/*   ------------  */
{
    Attribute	Att;

    FreeUnlessNil(Context->trees.raw);					Context->trees.raw = Nil;
    FreeUnlessNil(Context->trees.pruned);				Context->trees.pruned = Nil;

    FreeUnlessNil(Context->splits.tested_attributes);				Context->splits.tested_attributes = Nil;

    FreeUnlessNil(Context->splits.gain);				Context->splits.gain = Nil;
    FreeUnlessNil(Context->splits.information);				Context->splits.information = Nil;
    FreeUnlessNil(Context->splits.thresholds);					Context->splits.thresholds = Nil;

    FreeUnlessNil(Context->splits.estimated_max_gain_ratio);				Context->splits.estimated_max_gain_ratio = Nil;

    if ( Context->options.subset_splits )
    {
	FreeVector((void **) Context->splits.bell_numbers, 1, Context->schema.max_discrete_value);	Context->splits.bell_numbers = Nil;

	if ( Context->splits.subsets )
	{
	    ForEach(Att, 1, Context->schema.max_attribute)
	    {
		if ( Context->splits.subsets[Att] )
		{
		    FreeVector((void **) Context->splits.subsets[Att], 0, Context->schema.max_attribute_value[Att]);
		}
	    }
	    Free(Context->splits.subsets);				Context->splits.subsets = Nil;
	    Free(Context->splits.subset_counts);				Context->splits.subset_counts = Nil;
	}
    }

    FreeUnlessNil(Context->splits.discrete_attributes);				Context->splits.discrete_attributes = Nil;

    if ( Context->splits.discrete_frequencies )
    {
	ForEach(Att, 1, Context->schema.max_attribute)
	{
	    FreeUnlessNil(Context->splits.discrete_frequencies[Att]);
	}

	Free(Context->splits.discrete_frequencies);					Context->splits.discrete_frequencies = Nil;
    }

    FreeUnlessNil(Context->training.class_frequencies);				Context->training.class_frequencies = Nil;
    FreeUnlessNil(Context->splits.possible_cuts);			Context->splits.possible_cuts = Nil;

    Context->training.environment.reset();

    FreeUnlessNil(Context->splits.waiting_attributes);				Context->splits.waiting_attributes = Nil;
}



/*************************************************************************/
/*									 */
/*	Set threshold on minimum gain as follows:			 */
/*	  * when forming winnowing tree: no minimum			 */
/*	  * for small problems, AvGain (usual Context->splits.gain Ratio)		 */
/*	  * for large problems, discounted MDL				 */
/*	  * for intermediate problems, interpolated			 */
/*									 */
/*************************************************************************/


void SetMinGainThresh(c50_context *Context)
/*   ----------------  */
{
    float	Frac;

    /*  Set Context->splits.average_gain_weight and Context->splits.mdl_weight  */

    if ( Context->progress.stage == WINNOWATTS )
    {
	Context->splits.average_gain_weight = Context->splits.mdl_weight = 0.0;
    }
    else
    if ( (Context->cases.max_case+1) / Context->schema.max_class <= 500 )
    {
	Context->splits.average_gain_weight = 1.0;
	Context->splits.mdl_weight    = 0.0;
    }
    else
    if ( (Context->cases.max_case+1) / Context->schema.max_class >= 1000 )
    {
	Context->splits.average_gain_weight = 0.0;
	Context->splits.mdl_weight    = 0.9;
    }
    else
    {
	Frac = ((Context->cases.max_case+1) / Context->schema.max_class - 500) / 500.0;

	Context->splits.average_gain_weight = 1 - Frac;
	Context->splits.mdl_weight    = 0.9 * Frac;
    }
}



/*************************************************************************/
/*								 	 */
/*	Build a decision tree for the cases Fp through Lp	 	 */
/*								 	 */
/*    - if all cases are of the same class, the tree is a leaf labelled	 */
/*	with this class						 	 */
/*								 	 */
/*    - for each attribute, calculate the potential information provided */
/*	by a test on the attribute (based on the probabilities of each	 */
/*	case having a particular value for the attribute), and the gain	 */
/*	in information that would result from a test on the attribute	 */
/*	(based on the probabilities of each case with a particular	 */
/*	value for the attribute being of a particular class)		 */
/*								 	 */
/*    - on the basis of these figures, and depending on the current	 */
/*	selection criterion, find the best attribute to branch on. 	 */
/*	Note:  this version will not allow a split on an attribute	 */
/*	unless two or more subsets have at least Context->options.minimum_cases cases 	 */
/*								 	 */
/*    - try branching and test whether the resulting tree is better than */
/*	forming a leaf						 	 */
/*								 	 */
/*************************************************************************/


void FormTree(c50_context *Context, CaseNo Fp, CaseNo Lp, int Level,
	      Tree *Result)
/*   --------  */
{
    CaseCount	Cases=0, TreeErrs=0;
    Attribute	BestAtt;
    ClassNo	c, BestLeaf=1, Least=1;
    Tree	Node;
    DiscrValue	v;


    assert(Fp >= 0 && Lp >= Fp && Lp <= Context->cases.max_case);

    /*  Make a single pass through the cases to determine class frequencies
	and value/class frequencies for all discrete attributes  */

    FindAllFreq(Context, Fp, Lp);

    /*  Choose the best leaf and the least prevalent class  */

    ForEach(c, 2, Context->schema.max_class)
    {
	if ( Context->training.class_frequencies[c] > Context->training.class_frequencies[BestLeaf] )
	{
	    BestLeaf = c;
	}
	else
	if ( Context->training.class_frequencies[c] > 0.1 && Context->training.class_frequencies[c] < Context->training.class_frequencies[Least] )
	{
	    Least = c;
	}
    }

    ForEach(c, 1, Context->schema.max_class)
    {
	Cases += Context->training.class_frequencies[c];
    }

    Context->splits.max_leaves = ( Context->options.leaf_ratio > 0 ? rint(Context->options.leaf_ratio * Cases) : 1E6 );

    *Result = Node = Leaf(Context, Context->training.class_frequencies, BestLeaf, Cases,
			 Cases - Context->training.class_frequencies[BestLeaf]);

    Verbosity(1,
	c50_diagnostic_printf(Context->io.output, "\n<%d> %d cases", Level, No(Fp,Lp));
	if ( fabs(No(Fp,Lp) - Cases) >= 0.1 )
	{
	    c50_diagnostic_printf(Context->io.output, ", total weight %.1f", Cases);
	}
	c50_diagnostic_printf(Context->io.output, "\n"))

    /*  Do not try to split if:
	- all cases are of the same class
	- there are not enough cases to split  */

    if ( Context->training.class_frequencies[BestLeaf] >= 0.999 * Cases  ||
	 Cases < 2 * Context->options.minimum_cases ||
	 Context->splits.max_leaves < 2 )
    {
	if ( Context->progress.stage == FORMTREE ) Progress(Context, Cases);
	return;
    }

    /*  Calculate base information  */

    Context->splits.base_information = TotalInfo(Context->training.class_frequencies, 1, Context->schema.max_class) / Cases;

    /*  Perform preliminary evaluation if using subsampling.
	Must expect at least 10 of least prevalent class  */

    Context->splits.value_threshold = 0;
    if ( Context->splits.use_subsampling && No(Fp, Lp) > 5 * Context->schema.max_class * SAMPLEUNIT &&
	 (Context->training.class_frequencies[Least] * Context->schema.max_class * SAMPLEUNIT) / No(Fp, Lp) >= 10 )
    {
	SampleEstimate(Context, Fp, Lp, Cases);
	Context->splits.sampled   = true;
    }
    else
    {
	Context->splits.sampled = false;
    }

    BestAtt = ChooseSplit(Context, Fp, Lp, Cases, Context->splits.sampled);

    /*  Decide whether to branch or not  */

    if ( BestAtt == None )
    {
	Verbosity(1, c50_diagnostic_printf(Context->io.output, "\tno sensible splits\n"))
	if ( Context->progress.stage == FORMTREE ) Progress(Context, Cases);
    }
    else
    {
	Verbosity(1,
	    c50_diagnostic_printf(Context->io.output, "\tbest attribute %s", Context->schema.attribute_names[BestAtt]);
	    if ( Continuous(BestAtt) )
	    {
		c50_diagnostic_printf(Context->io.output, " cut %.3f", Context->splits.thresholds[BestAtt]);
	    }
	    c50_diagnostic_printf(Context->io.output, " inf %.3f gain %.3f val %.3f\n",
		   Context->splits.information[BestAtt], Context->splits.gain[BestAtt], Context->splits.gain[BestAtt] / Context->splits.information[BestAtt]))

	/*  Build a node of the selected test  */

	if ( Discrete(BestAtt) )
	{
	    if ( Context->options.subset_splits && Context->schema.max_attribute_value[BestAtt] > 3 && ! Ordered(BestAtt) )
	    {
		SubsetTest(Context, Node, BestAtt);
	    }
	    else
	    {
		DiscreteTest(Context, Node, BestAtt);
	    }
	}
	else
	{
	    ContinTest(Context, Node, BestAtt);
	}

	/*  Carry out the recursive divide-and-conquer  */

	++Context->splits.tested_attributes[BestAtt];

	Divide(Context, Node, Fp, Lp, Level);

	--Context->splits.tested_attributes[BestAtt];

	/*  See whether we would have been no worse off with a leaf  */

	ForEach(v, 1, Node->Forks)
	{
	    TreeErrs += Node->Branch[v]->Errors;
	}

	if ( TreeErrs >= 0.999 * Node->Errors )
	{
	    Verbosity(1,
		c50_diagnostic_printf(Context->io.output, "<%d> Collapse tree for %d cases to leaf %s\n",
			    Level, No(Fp,Lp), Context->schema.class_names[BestLeaf]))

	    UnSprout(Node);
	}
	else
	{
	    Node->Errors = TreeErrs;
	}
    }
}



/*************************************************************************/
/*								 	 */
/*	Estimate Context->splits.gain[] and Context->splits.information[] using sample				 */
/*								 	 */
/*************************************************************************/


void SampleEstimate(c50_context *Context, CaseNo Fp, CaseNo Lp,
		    CaseCount Cases)
/*   --------------  */
{
    CaseNo	SLp, SampleSize;
    CaseCount	NewCases;
    Attribute	Att;
    float	GR;

    /*  Phase 1: evaluate all discrete attributes and record best GR  */

    ForEach(Att, 1, Context->schema.max_attribute)
    { 
	Context->splits.gain[Att] = None;

	if ( Discrete(Att) )
	{
	    EvalDiscrSplit(Context, Att, Cases);

	    if ( Context->splits.information[Att] > Epsilon &&
		 (GR = Context->splits.gain[Att] / Context->splits.information[Att]) > Context->splits.value_threshold )
	    {
		Context->splits.value_threshold = GR;
	    }
	}
    }

    /*  Phase 2: generate sample  */

    SampleSize = Context->schema.max_class * SAMPLEUNIT;
    Sample(Context, Fp, Lp, SampleSize);
    SLp = Fp + SampleSize - 1;

    /*  Phase 3: evaluate continuous attributes using sample  */

    NewCases   = CountCases(Context, Fp, SLp);
    Context->splits.sample_fraction = NewCases / Cases;
    Context->splits.waiting_count   = 0;

    ForEach(Att, 1, Context->schema.max_attribute)
    { 
	if ( Continuous(Att) )
	{
	    Context->splits.waiting_attributes[Context->splits.waiting_count++] = Att;
	}
    } 

    ProcessQueue(Context, Fp, SLp, NewCases);

    Context->splits.sample_fraction = 1.0;
}



/*************************************************************************/
/*								 	 */
/*	Sample N cases from cases Fp through Lp				 */
/*								 	 */
/*************************************************************************/


void Sample(c50_context *Context, CaseNo Fp, CaseNo Lp, CaseNo N)
/*   ------  */
{
    CaseNo	i, j;
    double	Interval;

    Interval = No(Fp, Lp) / (double) N;

    ForEach(i, 0, N-1)
    {
	j = (i + 0.5) * Interval;

	assert(j >= 0 && Fp + j <= Lp);

	Swap(Fp + i, Fp + j);
    }
}



/*************************************************************************/
/*								 	 */
/*	Evaluate splits and choose best attribute to split on.		 */
/*	If Context->splits.sampled, Context->splits.gain[] and Context->splits.information[] have been estimated on		 */
/*	sample and unlikely candidates are not evaluated on all cases	 */
/*								 	 */
/*************************************************************************/


Attribute ChooseSplit(c50_context *Context, CaseNo Fp, CaseNo Lp,
		      CaseCount Cases, Boolean sampled)
/*        -----------  */
{
    Attribute	Att;
    int		i, j;


    /*  For each available attribute, find the information and gain  */

    Context->splits.waiting_count = 0;

    if ( sampled )
    {
	/*  If samples have been used, do not re-evaluate discrete atts
	    or atts that have low GR  */

	for ( Att = Context->schema.max_attribute ; Att > 0 ; Att-- )
	{
	    if ( ! Continuous(Att) ) continue;

	    if ( Context->splits.estimated_max_gain_ratio[Att] >= Context->splits.value_threshold )
	    {
		/*  Add attributes in reverse order of estimated max GR  */

		for ( i = 0 ;
		      i < Context->splits.waiting_count && Context->splits.estimated_max_gain_ratio[Context->splits.waiting_attributes[i]] < Context->splits.estimated_max_gain_ratio[Att] ;
		      i++ )
		    ;

		for ( j = Context->splits.waiting_count-1 ; j >= i ; j-- )
		{
		    Context->splits.waiting_attributes[j+1] = Context->splits.waiting_attributes[j];
		}
		Context->splits.waiting_count++;

		Context->splits.waiting_attributes[i] = Att;
	    }
	    else
	    {
		/*  Don't use -- attribute hasn't been fully evaluated.
		    Leave Context->splits.gain unchanged to get correct count for Possible  */

		Context->splits.information[Att] = -1E6;	/* negative so GR also negative */
	    }
	}
    }
    else
    {
	for ( Att = Context->schema.max_attribute ; Att > 0 ; Att-- )
	{
	    Context->splits.gain[Att] = None;

	    if ( Skip(Att) || Att == Context->schema.class_attribute )
	    {
		continue;
	    }

	    Context->splits.waiting_attributes[Context->splits.waiting_count++] = Att;
	}
    }

    ProcessQueue(Context, Fp, Lp, Cases);

    return FindBestAtt(Context, Cases);
}



static SplitResult ReadSplitResult(c50_context *Context, Attribute Att)
{
    SplitResult Result{};
    Result.Gain = Context->splits.gain[Att];
    Result.Information = Context->splits.information[Att];
    Result.EstimatedMaxGR = Context->splits.estimated_max_gain_ratio[Att];
    Result.Threshold = Context->splits.thresholds[Att];
    if ( Context->splits.subset_counts )
    {
        Result.SubsetCount = Context->splits.subset_counts[Att];
        Result.Subsets = Context->splits.subsets[Att];
    }
    return Result;
}


static void PublishSplitResult(c50_context *Context, Attribute Att,
                               const SplitResult &Result)
{
    Context->splits.gain[Att] = Result.Gain;
    Context->splits.information[Att] = Result.Information;
    Context->splits.estimated_max_gain_ratio[Att] = Result.EstimatedMaxGR;
    Context->splits.thresholds[Att] = Result.Threshold;
    if ( Context->splits.subset_counts )
    {
        Context->splits.subset_counts[Att] = Result.SubsetCount;
    }
}


static bool CanEvaluateQueueInParallel(c50_context *Context,
                                       CaseNo WFp, CaseNo WLp)
{
    constexpr size_t MinimumCases = 10000;
    return Context->split_worker_count > 1 &&
           Context->splits.waiting_count > 1 &&
           ! Context->splits.sampled &&
           Context->splits.sample_fraction == 1 &&
           Context->options.verbosity == 0 &&
           WFp >= 0 && WLp >= WFp &&
           static_cast<size_t>(WLp - WFp) + 1 >= MinimumCases;
}


static void EvaluateQueueInParallel(c50_context *Context, CaseNo WFp,
                                    CaseNo WLp, CaseCount WCases)
{
    std::vector<Attribute> Queue;
    std::vector<SplitResult> Results;
    Queue.reserve(Context->splits.waiting_count);
    Results.reserve(Context->splits.waiting_count);
    while ( Context->splits.waiting_count > 0 )
    {
        Attribute Att = Context->splits.waiting_attributes[
            --Context->splits.waiting_count];
        Queue.push_back(Att);
        Results.push_back(ReadSplitResult(Context, Att));
    }

    const size_t Workers = std::min<size_t>(Context->split_worker_count,
                                            Queue.size());
    std::array<SplitWorkspace *, 8> Workspaces{};
    std::vector<SplitWorkspacePtr> ExtraWorkspaces;
    ExtraWorkspaces.reserve(Workers - 1);
    Workspaces[0] = Context->training.environment.get();
    for ( size_t Worker = 1; Worker < Workers; ++Worker )
    {
        C50_SPLIT_TEST_POINT(workspace_begin, Worker);
        ExtraWorkspaces.push_back(MakeSplitWorkspace(Context));
        C50_SPLIT_TEST_POINT(workspace_ready, Worker);
        Workspaces[Worker] = ExtraWorkspaces.back().get();
    }

    auto EvaluateWorker = [&](size_t Worker)
    {
        C50_SPLIT_TEST_WORKER(Worker);
        C50_SPLIT_TEST_POINT(worker_begin, Worker);
        SplitWorkspace &Workspace = *Workspaces[Worker];
        for ( size_t Index = Worker; Index < Queue.size(); Index += Workers )
        {
            Attribute Att = Queue[Index];
            if ( Discrete(Att) )
            {
                EvalDiscrSplit(Context, Workspace, Results[Index], Att, WCases);
            }
            else
            {
                EvalContinuousAtt(Context, Workspace, Results[Index], Att,
                                  WFp, WLp);
            }
        }
    };

    std::vector<std::future<void>> Tasks;
    Tasks.reserve(Workers - 1);
    for ( size_t Worker = 1; Worker < Workers; ++Worker )
    {
        C50_SPLIT_TEST_POINT(task_launch, Worker);
        Tasks.push_back(std::async(std::launch::async, EvaluateWorker, Worker));
    }

    std::exception_ptr Failure;
    try
    {
        EvaluateWorker(0);
    }
    catch ( ... )
    {
        Failure = std::current_exception();
    }
    for ( auto &Task : Tasks )
    {
        try
        {
            Task.get();
        }
        catch ( ... )
        {
            if ( ! Failure ) Failure = std::current_exception();
        }
    }
    if ( Failure ) std::rethrow_exception(Failure);

    // Preserve the serial queue order for publication and tie selection.
    for ( size_t Index = 0; Index < Queue.size(); ++Index )
    {
        PublishSplitResult(Context, Queue[Index], Results[Index]);
    }
}


void ProcessQueue(c50_context *Context, CaseNo WFp, CaseNo WLp,
                  CaseCount WCases)
{
    if ( CanEvaluateQueueInParallel(Context, WFp, WLp) )
    {
        EvaluateQueueInParallel(Context, WFp, WLp, WCases);
        return;
    }

    float GR;

    for ( ; Context->splits.waiting_count > 0 ; )
    {
        Attribute Att =
            Context->splits.waiting_attributes[--Context->splits.waiting_count];
        SplitResult Result = ReadSplitResult(Context, Att);

        if ( Discrete(Att) )
        {
            EvalDiscrSplit(Context, *Context->training.environment, Result,
                           Att, WCases);
        }
        else if ( Context->splits.sample_fraction < 1 )
        {
            EstimateMaxGR(Context, *Context->training.environment, Result,
                          Att, WFp, WLp);
        }
        else if ( Context->splits.sampled )
        {
            Result.Information = -1E16;
            if ( Result.EstimatedMaxGR > Context->splits.value_threshold )
            {
                EvalContinuousAtt(Context, *Context->training.environment,
                                  Result, Att, WFp, WLp);
                if ( Result.Information > Epsilon &&
                     (GR = Result.Gain / Result.Information) >
                         Context->splits.value_threshold )
                {
                    Context->splits.value_threshold = GR;
                }
            }
        }
        else
        {
            EvalContinuousAtt(Context, *Context->training.environment, Result,
                              Att, WFp, WLp);
        }

        PublishSplitResult(Context, Att, Result);
    }
}



/*************************************************************************/
/*								 	 */
/*	Adjust each attribute's gain to reflect choice and		 */
/*	select att with maximum GR					 */
/*								 	 */
/*************************************************************************/


Attribute FindBestAtt(c50_context *Context, CaseCount Cases)
/*	  -----------  */
{
    double	BestVal, Val, MinGain=1E6, AvGain=0, MDL;
    Attribute	Att, BestAtt, Possible=0;
    DiscrValue	NBr, BestNBr=Context->schema.max_discrete_value+1;

    ForEach(Att, 1, Context->schema.max_attribute)
    {
	/*  Update the number of possible attributes for splitting and
	    average gain (unless very many values)  */

	if ( Context->splits.gain[Att] >= Epsilon &&
	     ( Context->splits.all_attributes_multi_valued || Context->schema.max_attribute_value[Att] < 0.3 * (Context->cases.max_case + 1) ) )
	{
	    Possible++;
	    AvGain += Context->splits.gain[Att];
	}
	else
	{
	    Context->splits.gain[Att] = None;
	}
    }

    /*  Set threshold on minimum gain  */

    if ( ! Possible ) return None;

    AvGain /= Possible;
    MDL     = Log(Possible) / Cases;
    MinGain = AvGain * Context->splits.average_gain_weight + MDL * Context->splits.mdl_weight;

    Verbosity(2,
	c50_diagnostic_printf(Context->io.output, "\tav gain=%.3f, MDL (%d) = %.3f, min=%.3f\n",
		    AvGain, Possible, MDL, MinGain))

    /*  Find best attribute according to Context->splits.gain Ratio criterion subject
	to threshold on minimum gain  */

    BestVal = -Epsilon;
    BestAtt = None;

    ForEach(Att, 1, Context->schema.max_attribute)
    {
	if ( Context->splits.gain[Att] >= 0.999 * MinGain && Context->splits.information[Att] > 0 )
	{
	    Val = Context->splits.gain[Att] / Context->splits.information[Att];
	    NBr = ( Context->schema.max_attribute_value[Att] <= 3 || Ordered(Att) ? 3 :
		    Context->options.subset_splits ? Context->splits.subset_counts[Att] : Context->schema.max_attribute_value[Att] );

	    if ( Val > BestVal ||
		 ( Val > 0.999 * BestVal &&
		   ( NBr < BestNBr ||
		     ( NBr == BestNBr && Context->splits.gain[Att] > Context->splits.gain[BestAtt] ) ) ) )
	    {
		BestAtt = Att;
		BestVal = Val;
		BestNBr = NBr;
	    }
	}
    }

    return BestAtt;
}



/*************************************************************************/
/*								 	 */
/*	Evaluate split on Att						 */
/*								 	 */
/*************************************************************************/


void EvalDiscrSplit(c50_context *Context, SplitWorkspace &Workspace, SplitResult &Result, Attribute Att, CaseCount Cases)
/*   --------------  */
{
    DiscrValue	v, NBr;

    Result.Gain = None;

    if ( Skip(Att) || Att == Context->schema.class_attribute ) return;

    if ( Ordered(Att) )
    {
	EvalOrderedAtt(Context, Workspace, Result, Att, Cases);
	NBr = ( Workspace.ValFreq[1] > 0.5 ? 3 : 2 );
    }
    else
    if ( Context->options.subset_splits && Context->schema.max_attribute_value[Att] > 3 )
    {
	EvalSubset(Context, Workspace, Result, Att, Cases);
	NBr = Result.SubsetCount;
    }
    else
    if ( ! Context->splits.tested_attributes[Att] )
    {
	EvalDiscreteAtt(Context, Workspace, Result, Att, Cases);

	NBr = 0;
	ForEach(v, 1, Context->schema.max_attribute_value[Att])
	{
	    if ( Workspace.ValFreq[v] > 0.5 ) NBr++;
	}
    }
    else
    {
	NBr = 0;
    }

    /*  Check that this test will not give too many leaves  */

    if ( NBr > Context->splits.max_leaves + 1 )
    {
	Verbosity(2,
	    c50_diagnostic_printf(Context->io.output, "\t(cancelled -- %d leaves, max %d)\n", NBr, Context->splits.max_leaves))

	Result.Gain = None;
    }
}


void EvalDiscrSplit(c50_context *Context, Attribute Att, CaseCount Cases)
{
    SplitResult Result = ReadSplitResult(Context, Att);
    EvalDiscrSplit(Context, *Context->training.environment, Result, Att, Cases);
    PublishSplitResult(Context, Att, Result);
}



/*************************************************************************/
/*								 	 */
/*	Form the subtrees for the given node				 */
/*								 	 */
/*************************************************************************/


void Divide(c50_context *Context, Tree T, CaseNo Fp, CaseNo Lp, int Level)
/*   ------  */
{
    CaseNo	Bp, Ep, Missing, Cases, i;
    CaseCount	KnownCases, MissingCases, BranchCases;
    Attribute	Att;
    double	Factor;
    DiscrValue	v;
    Boolean	PrevUnitWeights;

    PrevUnitWeights = Context->costs.unit_weights;

    Att = T->Tested;
    Missing = (Ep = Group(Context, 0, Fp, Lp, T)) - Fp + 1;

    KnownCases = T->Cases - (MissingCases = CountCases(Context, Fp, Ep));

    if ( Missing )
    {
	Context->costs.unit_weights = false;

	/*  If using costs, must adjust branch factors to undo effects of
	    reweighting cases  */

	if ( Context->costs.weighted )
	{
	    KnownCases = SumNocostWeights(Context, Ep+1, Lp);
	}

	/*  If there are many cases with missing values and many branches,
	    skip cases whose weight < 0.1  */

	if ( (Cases = No(Fp,Lp)) > 1000 &&
	     Missing > 0.5 * Cases &&
	     T->Forks >= 10 )
	{
	    ForEach(i, Fp, Ep)
	    {
		if ( Weight(Context->cases.records[i]) < 0.1 )
		{
		    Missing--;
		    MissingCases -= Weight(Context->cases.records[i]);
		    Swap(Fp, i);
		    Fp++;
		}
	    }

	    assert(Missing >= 0);
	}
    }

    Bp = Fp;
    ForEach(v, 1, T->Forks)
    {
	Ep = Group(Context, v, Bp + Missing, Lp, T);

	assert(Bp + Missing <= Lp+1 && Ep <= Lp);

	/*  Bp -> first value in missing + remaining values
	    Ep -> last value in missing + current group  */

	BranchCases = CountCases(Context, Bp + Missing, Ep);

	Factor = ( ! Missing ? 0 :
		   ! Context->costs.weighted ? BranchCases / KnownCases :
		   SumNocostWeights(Context, Bp + Missing, Ep) / KnownCases );

	if ( BranchCases + Factor * MissingCases >= MinLeaf )
	{
	    if ( Missing )
	    {
		/*  Adjust weights of cases with missing values  */

		ForEach(i, Bp, Bp + Missing - 1)
		{
		    Weight(Context->cases.records[i]) *= Factor;
		}
	    }

	    FormTree(Context, Bp, Ep, Level+1, &T->Branch[v]);

	    /*  Restore weights if changed  */

	    if ( Missing )
	    {
		for ( i = Ep ; i >= Bp ; i-- )
		{
		    if ( Unknown(Context->cases.records[i], Att) )
		    {
			Weight(Context->cases.records[i]) /= Factor;
			Swap(i, Ep);
			Ep--;
		    }
		}
	    }

	    Bp = Ep+1;
	}
	else
	{
	    T->Branch[v] = Leaf(Context, Nil, T->Leaf, 0.0, 0.0);
	}
    }

    Context->costs.unit_weights = PrevUnitWeights;
}



/*************************************************************************/
/*								 	 */
/*	Group together the cases corresponding to branch V of a test 	 */
/*	and return the index of the last such			 	 */
/*								 	 */
/*	Note: if V equals zero, group the unknown values	 	 */
/*								 	 */
/*************************************************************************/


CaseNo Group(c50_context *Context, DiscrValue V, CaseNo Bp, CaseNo Ep,
	     Tree TestNode)
/*     -----  */
{
    CaseNo	i;
    Attribute	Att;
    ContValue	Thresh;
    Set		SS;

    Att = TestNode->Tested;

    if ( ! V )
    {
	/*  Group together unknown values (if any)  */

	if ( Context->cases.some_missing[Att] )
	{
	    ForEach(i, Bp, Ep)
	    {
		if ( Unknown(Context->cases.records[i], Att) )
		{
		    Swap(Bp, i);
		    Bp++;
		}
	    }
	}
    }
    else				/* skip non-existant N/A values */
    if ( V != 1 || TestNode->NodeType == BrSubset || Context->cases.some_not_applicable[Att] )
    {
	/*  Group cases on the value of attribute Att, and depending
	    on the type of branch  */

	switch ( TestNode->NodeType )
	{
	    case BrDiscr:

		ForEach(i, Bp, Ep)
		{
		    if ( DVal(Context->cases.records[i], Att) == V )
		    {
			Swap(Bp, i);
			Bp++;
		    }
		}
		break;

	    case BrThresh:

		Thresh = TestNode->Cut;
		ForEach(i, Bp, Ep)
		{
		    if ( V == 1 ? NotApplic(Context, Context->cases.records[i], Att) :
			 (CVal(Context->cases.records[i], Att) <= Thresh) == (V == 2) )
		    {
			Swap(Bp, i);
			Bp++;
		    }
		}
		break;

	    case BrSubset:

		SS = TestNode->Subset[V];
		ForEach(i, Bp, Ep)
		{
		    if ( In(XDVal(Context->cases.records[i], Att), SS) )
		    {
			Swap(Bp, i);
			Bp++;
		    }
		}
		break;
	}
    }

    return Bp - 1;
}



/*************************************************************************/
/*								 	 */
/*	Return the total weight of cases from Fp to Lp		 	 */
/*								 	 */
/*************************************************************************/


CaseCount SumWeights(c50_context *Context, CaseNo Fp, CaseNo Lp)
/*        ----------  */
{
    double	Sum=0.0;
    CaseNo	i;

    assert(Fp >= 0 && Lp >= Fp-1 && Lp <= Context->cases.max_case);

    ForEach(i, Fp, Lp)
    {
	Sum += Weight(Context->cases.records[i]);
    }

    return Sum;
}



/*************************************************************************/
/*								 	 */
/*	Special version to undo the weightings associated with costs	 */
/*								 	 */
/*************************************************************************/


CaseCount SumNocostWeights(c50_context *Context, CaseNo Fp, CaseNo Lp)
/*        ----------------  */
{
    double	Sum=0.0;
    CaseNo	i;

    assert(Fp >= 0 && Lp >= Fp-1 && Lp <= Context->cases.max_case);

    ForEach(i, Fp, Lp)
    {
	Sum += Weight(Context->cases.records[i]) / Context->costs.weight_multipliers[Class(Context->cases.records[i])];
    }

    return Sum;
}



/*************************************************************************/
/*                                                               	 */
/*	Generate class frequency distribution				 */
/*									 */
/*************************************************************************/


void FindClassFreq(c50_context *Context, double *Frequencies,
		   CaseNo Fp, CaseNo Lp)
/*   -------------  */
{
    ClassNo	c;
    CaseNo	i;

    assert(Fp >= 0 && Lp >= Fp && Lp <= Context->cases.max_case);

    ForEach(c, 0, Context->schema.max_class)
    {
	Frequencies[c] = 0;
    }

    ForEach(i, Fp, Lp)
    {
	assert(Class(Context->cases.records[i]) >= 1 && Class(Context->cases.records[i]) <= Context->schema.max_class);

	Frequencies[Class(Context->cases.records[i])] +=
	    Weight(Context->cases.records[i]);
    }
}



/*************************************************************************/
/*                                                               	 */
/*	Find all discrete frequencies					 */
/*									 */
/*************************************************************************/


void FindAllFreq(c50_context *Context, CaseNo Fp, CaseNo Lp)
/*   -----------  */
{
    ClassNo	c;
    CaseNo	i;
    Attribute	Att, a;
    CaseCount	w;
    int		x;

    /*  Zero all values  */

    ForEach(c, 0, Context->schema.max_class)
    {
	Context->training.class_frequencies[c] = 0;
    }

    for ( a = 0 ; a < Context->splits.discrete_attribute_count ; a++ )
    {
	Att = Context->splits.discrete_attributes[a];
	for ( x = Context->schema.max_class * (Context->schema.max_attribute_value[Att]+1) - 1 ; x >= 0 ; x-- )
	{
	    Context->splits.discrete_frequencies[Att][x] = 0;
	}
    }

    /*  Scan cases  */

    ForEach(i, Fp, Lp)
    {
	Context->training.class_frequencies[ (c=Class(Context->cases.records[i])) ] += (w=Weight(Context->cases.records[i]));

	for ( a = 0 ; a < Context->splits.discrete_attribute_count ; a++ )
	{
	    Att = Context->splits.discrete_attributes[a];
	    Context->splits.discrete_frequencies[Att][ Context->schema.max_class * XDVal(Context->cases.records[i], Att) + (c-1) ] += w;
	}
    }
}
