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
/*								 	 */
/*    Central tree-forming algorithm					 */
/*    ------------------------------					 */
/*								 	 */
/*************************************************************************/


#include "defns.i"
#include "extern.i"
#include "c50_api_internal.h"


Boolean		MultiVal,	/* all atts have many values */
		Subsample;	/* use subsampling */
float		AvGainWt,	/* weight of average gain in gain threshold */
		MDLWt;		/* weight of MDL threshold ditto */

Attribute	*DList=Nil;	/* list of discrete atts */
int		NDList;		/* number in list */

DiscrValue	MaxLeaves;	/* target maximum tree size */

#define		SAMPLEUNIT	2000

float		ValThresh;	/* minimum GR when evaluating sampled atts */
Boolean		Sampled;	/* true if sampling used */

Attribute	*Waiting=Nil,	/* attribute wait list */
		NWaiting=0;




/*************************************************************************/
/*								 	 */
/*	Allocate space for tree tables				 	 */
/*								 	 */
/*************************************************************************/


void InitialiseTreeData(c50_context *Context)
/*   ------------------  */
{
    DiscrValue	v;
    Attribute	Att;
    DiscrValue	vMax;
    size_t	NoAttributes;
    size_t	NoCases;

    Raw	     = AllocZero(TRIALS+1, Tree);
    Pruned   = AllocZero(TRIALS+1, Tree);

    Tested   = AllocZero(Context->schema.max_attribute+1, Byte);

    Gain     = AllocZero(Context->schema.max_attribute+1, float);
    Info     = AllocZero(Context->schema.max_attribute+1, float);
    Bar      = AllocZero(Context->schema.max_attribute+1, ContValue);

    EstMaxGR = AllocZero(Context->schema.max_attribute+1, float);

    /*  Data for subsets  */

    if ( SUBSET )
    {
	InitialiseBellNumbers(Context);
	Subset = Alloc(Context->schema.max_attribute+1, Set *);

	ForEach(Att, 1, Context->schema.max_attribute)
	{
	    if ( Discrete(Att) && Att != Context->schema.class_attribute && ! Skip(Att) )
	    {
		Subset[Att] = AllocZero(Context->schema.max_attribute_value[Att]+1, Set);
		ForEach(v, 0, Context->schema.max_attribute_value[Att])
		{
		    Subset[Att][v] = Alloc((Context->schema.max_attribute_value[Att]>>3)+1, Byte);
		}
	    }
	}
	Subsets = AllocZero(Context->schema.max_attribute+1, int);
    }

    NoAttributes = ( Context->schema.max_attribute < 1 ? 0 : (size_t) Context->schema.max_attribute );
    DList  = Alloc(NoAttributes, Attribute);
    NDList = 0;

    DFreq = AllocZero(Context->schema.max_attribute+1, double *);
    ForEach(Att, 1, Context->schema.max_attribute)
    {
	if ( Att == Context->schema.class_attribute || Skip(Att) || ! Discrete(Att) ) continue;

	DList[NDList++] = Att;

	DFreq[Att] = Alloc(Context->schema.max_class * (Context->schema.max_attribute_value[Att]+1), double);
    }

    ClassFreq = AllocZero(Context->schema.max_class+1, double);
    Context->class_sum  = Alloc(Context->schema.max_class+1, float);

    if ( BOOST )
    {
	Context->votes      = Alloc(Context->schema.max_class+1, float);
	Context->trial_predictions = Alloc(TRIALS, ClassNo);
    }

    if ( RULES )
    {
	Context->most_specific_rules     = Alloc(Context->schema.max_class+1, CRule);
	PossibleCuts = Alloc(Context->schema.max_attribute+1, int);
    }

    /*  Check whether all attributes have many discrete values  */

    MultiVal = true;
    if ( ! SUBSET )
    {
	for ( Att = 1 ; MultiVal && Att <= Context->schema.max_attribute ; Att++ )
	{
	    if ( ! Skip(Att) && Att != Context->schema.class_attribute )
	    {
		MultiVal = Context->schema.max_attribute_value[Att] >= 0.3 * (MaxCase + 1);
	    }
	}
    }

    /*  See whether there are continuous attributes for subsampling  */

    Subsample = false;

    /*  Set parameters for RawExtraErrs() */

    InitialiseExtraErrs();

    /*  Set up environment  */

    Waiting = Alloc(Context->schema.max_attribute+1, Attribute);

    vMax = Max(3, Context->schema.max_discrete_value+1);

    GEnv.Freq = Alloc(vMax+1, double *);
    ForEach(v, 0, vMax)
    {
	GEnv.Freq[v] = Alloc(Context->schema.max_class+1, double);
    }

    GEnv.ValFreq = Alloc(vMax, double);

    GEnv.ClassFreq = Alloc(Context->schema.max_class+1, double);

    NoCases = ( MaxCase < 0 ? 0 : (size_t) MaxCase + 1 );
    GEnv.SRec = Alloc(NoCases, SortRec);

    if ( SUBSET )
    {
	GEnv.SubsetInfo = Alloc(Context->schema.max_discrete_value+1, double);
	GEnv.SubsetEntr = Alloc(Context->schema.max_discrete_value+1, double);

	GEnv.MergeInfo = Alloc(Context->schema.max_discrete_value+1, double *);
	GEnv.MergeEntr = Alloc(Context->schema.max_discrete_value+1, double *);
	GEnv.WSubset   = Alloc(Context->schema.max_discrete_value+1, Set);
	ForEach(v, 1, Context->schema.max_discrete_value)
	{
	    GEnv.MergeInfo[v] = Alloc(Context->schema.max_discrete_value+1, double);
	    GEnv.MergeEntr[v] = Alloc(Context->schema.max_discrete_value+1, double);
	    GEnv.WSubset[v]   = Alloc((Context->schema.max_discrete_value>>3)+1, Byte);
	}
    }
}


void FreeTreeData(c50_context *Context)
/*   ------------  */
{
    Attribute	Att;
    DiscrValue	vMax;

    FreeUnlessNil(Raw);					Raw = Nil;
    FreeUnlessNil(Pruned);				Pruned = Nil;

    FreeUnlessNil(Tested);				Tested = Nil;

    FreeUnlessNil(Gain);				Gain = Nil;
    FreeUnlessNil(Info);				Info = Nil;
    FreeUnlessNil(Bar);					Bar = Nil;

    FreeUnlessNil(EstMaxGR);				EstMaxGR = Nil;

    if ( SUBSET )
    {
	FreeVector((void **) Bell, 1, Context->schema.max_discrete_value);	Bell = Nil;

	if ( Subset )
	{
	    ForEach(Att, 1, Context->schema.max_attribute)
	    {
		if ( Subset[Att] )
		{
		    FreeVector((void **) Subset[Att], 0, Context->schema.max_attribute_value[Att]);
		}
	    }
	    Free(Subset);				Subset = Nil;
	    Free(Subsets);				Subsets = Nil;
	}
    }

    FreeUnlessNil(DList);				DList = Nil;

    if ( DFreq )
    {
	ForEach(Att, 1, Context->schema.max_attribute)
	{
	    FreeUnlessNil(DFreq[Att]);
	}

	Free(DFreq);					DFreq = Nil;
    }

    FreeUnlessNil(ClassFreq);				ClassFreq = Nil;
    FreeUnlessNil(PossibleCuts);			PossibleCuts = Nil;

    vMax = Max(3, Context->schema.max_discrete_value+1);
    FreeVector((void **) GEnv.Freq, 0, vMax);
    Free(GEnv.ValFreq);
    Free(GEnv.ClassFreq);
    FreeUnlessNil(GEnv.SRec);

    if ( GEnv.SubsetInfo )
    {
	Free(GEnv.SubsetInfo);
	Free(GEnv.SubsetEntr);
	FreeVector((void **) GEnv.MergeInfo, 1, Context->schema.max_discrete_value);
	FreeVector((void **) GEnv.MergeEntr, 1, Context->schema.max_discrete_value);
	FreeVector((void **) GEnv.WSubset, 1, Context->schema.max_discrete_value);
    }

    FreeUnlessNil(Waiting);				Waiting = Nil;
}



/*************************************************************************/
/*									 */
/*	Set threshold on minimum gain as follows:			 */
/*	  * when forming winnowing tree: no minimum			 */
/*	  * for small problems, AvGain (usual Gain Ratio)		 */
/*	  * for large problems, discounted MDL				 */
/*	  * for intermediate problems, interpolated			 */
/*									 */
/*************************************************************************/


void SetMinGainThresh(c50_context *Context)
/*   ----------------  */
{
    float	Frac;

    /*  Set AvGainWt and MDLWt  */

    if ( Now == WINNOWATTS )
    {
	AvGainWt = MDLWt = 0.0;
    }
    else
    if ( (MaxCase+1) / Context->schema.max_class <= 500 )
    {
	AvGainWt = 1.0;
	MDLWt    = 0.0;
    }
    else
    if ( (MaxCase+1) / Context->schema.max_class >= 1000 )
    {
	AvGainWt = 0.0;
	MDLWt    = 0.9;
    }
    else
    {
	Frac = ((MaxCase+1) / Context->schema.max_class - 500) / 500.0;

	AvGainWt = 1 - Frac;
	MDLWt    = 0.9 * Frac;
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
/*	unless two or more subsets have at least MINITEMS cases 	 */
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


    assert(Fp >= 0 && Lp >= Fp && Lp <= MaxCase);

    /*  Make a single pass through the cases to determine class frequencies
	and value/class frequencies for all discrete attributes  */

    FindAllFreq(Context, Fp, Lp);

    /*  Choose the best leaf and the least prevalent class  */

    ForEach(c, 2, Context->schema.max_class)
    {
	if ( ClassFreq[c] > ClassFreq[BestLeaf] )
	{
	    BestLeaf = c;
	}
	else
	if ( ClassFreq[c] > 0.1 && ClassFreq[c] < ClassFreq[Least] )
	{
	    Least = c;
	}
    }

    ForEach(c, 1, Context->schema.max_class)
    {
	Cases += ClassFreq[c];
    }

    MaxLeaves = ( LEAFRATIO > 0 ? rint(LEAFRATIO * Cases) : 1E6 );

    *Result = Node = Leaf(Context, ClassFreq, BestLeaf, Cases,
			 Cases - ClassFreq[BestLeaf]);

    Verbosity(1,
    	fprintf(Of, "\n<%d> %d cases", Level, No(Fp,Lp));
	if ( fabs(No(Fp,Lp) - Cases) >= 0.1 )
	{
	    fprintf(Of, ", total weight %.1f", Cases);
	}
	fprintf(Of, "\n"))

    /*  Do not try to split if:
	- all cases are of the same class
	- there are not enough cases to split  */

    if ( ClassFreq[BestLeaf] >= 0.999 * Cases  ||
	 Cases < 2 * MINITEMS ||
	 MaxLeaves < 2 )
    {
	if ( Now == FORMTREE ) Progress(Cases);
	return;
    }

    /*  Calculate base information  */

    GlobalBaseInfo = TotalInfo(ClassFreq, 1, Context->schema.max_class) / Cases;

    /*  Perform preliminary evaluation if using subsampling.
	Must expect at least 10 of least prevalent class  */

    ValThresh = 0;
    if ( Subsample && No(Fp, Lp) > 5 * Context->schema.max_class * SAMPLEUNIT &&
	 (ClassFreq[Least] * Context->schema.max_class * SAMPLEUNIT) / No(Fp, Lp) >= 10 )
    {
	SampleEstimate(Context, Fp, Lp, Cases);
	Sampled   = true;
    }
    else
    {
	Sampled = false;
    }

    BestAtt = ChooseSplit(Context, Fp, Lp, Cases, Sampled);

    /*  Decide whether to branch or not  */

    if ( BestAtt == None )
    {
	Verbosity(1, fprintf(Of, "\tno sensible splits\n"))
	if ( Now == FORMTREE ) Progress(Cases);
    }
    else
    {
	Verbosity(1,
	    fprintf(Of, "\tbest attribute %s", Context->schema.attribute_names[BestAtt]);
	    if ( Continuous(BestAtt) )
	    {
		fprintf(Of, " cut %.3f", Bar[BestAtt]);
	    }
	    fprintf(Of, " inf %.3f gain %.3f val %.3f\n",
		   Info[BestAtt], Gain[BestAtt], Gain[BestAtt] / Info[BestAtt]))

	/*  Build a node of the selected test  */

	if ( Discrete(BestAtt) )
	{
	    if ( SUBSET && Context->schema.max_attribute_value[BestAtt] > 3 && ! Ordered(BestAtt) )
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
	    ContinTest(Node, BestAtt);
	}

	/*  Carry out the recursive divide-and-conquer  */

	++Tested[BestAtt];

	Divide(Context, Node, Fp, Lp, Level);

	--Tested[BestAtt];

	/*  See whether we would have been no worse off with a leaf  */

	ForEach(v, 1, Node->Forks)
	{
	    TreeErrs += Node->Branch[v]->Errors;
	}

	if ( TreeErrs >= 0.999 * Node->Errors )
	{
	    Verbosity(1,
		fprintf(Of, "<%d> Collapse tree for %d cases to leaf %s\n",
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
/*	Estimate Gain[] and Info[] using sample				 */
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
	Gain[Att] = None;

	if ( Discrete(Att) )
	{
	    EvalDiscrSplit(Context, Att, Cases);

	    if ( Info[Att] > Epsilon &&
		 (GR = Gain[Att] / Info[Att]) > ValThresh )
	    {
		ValThresh = GR;
	    }
	}
    }

    /*  Phase 2: generate sample  */

    SampleSize = Context->schema.max_class * SAMPLEUNIT;
    Sample(Fp, Lp, SampleSize);
    SLp = Fp + SampleSize - 1;

    /*  Phase 3: evaluate continuous attributes using sample  */

    NewCases   = CountCases(Fp, SLp);
    SampleFrac = NewCases / Cases;
    NWaiting   = 0;

    ForEach(Att, 1, Context->schema.max_attribute)
    { 
	if ( Continuous(Att) )
	{
	    Waiting[NWaiting++] = Att;
	}
    } 

    ProcessQueue(Context, Fp, SLp, NewCases);

    SampleFrac = 1.0;
}



/*************************************************************************/
/*								 	 */
/*	Sample N cases from cases Fp through Lp				 */
/*								 	 */
/*************************************************************************/


void Sample(CaseNo Fp, CaseNo Lp, CaseNo N)
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
/*	If Sampled, Gain[] and Info[] have been estimated on		 */
/*	sample and unlikely candidates are not evaluated on all cases	 */
/*								 	 */
/*************************************************************************/


Attribute ChooseSplit(c50_context *Context, CaseNo Fp, CaseNo Lp,
		      CaseCount Cases, Boolean Sampled)
/*        -----------  */
{
    Attribute	Att;
    int		i, j;


    /*  For each available attribute, find the information and gain  */

    NWaiting = 0;

    if ( Sampled )
    {
	/*  If samples have been used, do not re-evaluate discrete atts
	    or atts that have low GR  */

	for ( Att = Context->schema.max_attribute ; Att > 0 ; Att-- )
	{
	    if ( ! Continuous(Att) ) continue;

	    if ( EstMaxGR[Att] >= ValThresh )
	    {
		/*  Add attributes in reverse order of estimated max GR  */

		for ( i = 0 ;
		      i < NWaiting && EstMaxGR[Waiting[i]] < EstMaxGR[Att] ;
		      i++ )
		    ;

		for ( j = NWaiting-1 ; j >= i ; j-- )
		{
		    Waiting[j+1] = Waiting[j];
		}
		NWaiting++;

		Waiting[i] = Att;
	    }
	    else
	    {
		/*  Don't use -- attribute hasn't been fully evaluated.
		    Leave Gain unchanged to get correct count for Possible  */

		Info[Att] = -1E6;	/* negative so GR also negative */
	    }
	}
    }
    else
    {
	for ( Att = Context->schema.max_attribute ; Att > 0 ; Att-- )
	{
	    Gain[Att] = None;

	    if ( Skip(Att) || Att == Context->schema.class_attribute )
	    {
		continue;
	    }

	    Waiting[NWaiting++] = Att;
	}
    }

    ProcessQueue(Context, Fp, Lp, Cases);

    return FindBestAtt(Context, Cases);
}



void ProcessQueue(c50_context *Context, CaseNo WFp, CaseNo WLp,
		  CaseCount WCases)
/*   ------------  */
{
    Attribute	Att;
    float	GR;

    for ( ; NWaiting > 0 ; )
    {
	Att = Waiting[--NWaiting];

	if ( Discrete(Att) )
	{
	    EvalDiscrSplit(Context, Att, WCases);
	}
	else
	if ( SampleFrac < 1 )
	{
	    EstimateMaxGR(Context, Att, WFp, WLp);
	}
	else
	if ( Sampled )
	{
	    Info[Att] = -1E16;

	    if ( EstMaxGR[Att] > ValThresh )
	    {
		EvalContinuousAtt(Context, Att, WFp, WLp);

		if ( Info[Att] > Epsilon &&
		     (GR = Gain[Att] / Info[Att]) > ValThresh )
		{
		    if ( GR > ValThresh ) ValThresh = GR;
		}
	    }
	}
	else
	{
	    EvalContinuousAtt(Context, Att, WFp, WLp);
	}
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

	if ( Gain[Att] >= Epsilon &&
	     ( MultiVal || Context->schema.max_attribute_value[Att] < 0.3 * (MaxCase + 1) ) )
	{
	    Possible++;
	    AvGain += Gain[Att];
	}
	else
	{
	    Gain[Att] = None;
	}
    }

    /*  Set threshold on minimum gain  */

    if ( ! Possible ) return None;

    AvGain /= Possible;
    MDL     = Log(Possible) / Cases;
    MinGain = AvGain * AvGainWt + MDL * MDLWt;

    Verbosity(2,
	fprintf(Of, "\tav gain=%.3f, MDL (%d) = %.3f, min=%.3f\n",
		    AvGain, Possible, MDL, MinGain))

    /*  Find best attribute according to Gain Ratio criterion subject
	to threshold on minimum gain  */

    BestVal = -Epsilon;
    BestAtt = None;

    ForEach(Att, 1, Context->schema.max_attribute)
    {
	if ( Gain[Att] >= 0.999 * MinGain && Info[Att] > 0 )
	{
	    Val = Gain[Att] / Info[Att];
	    NBr = ( Context->schema.max_attribute_value[Att] <= 3 || Ordered(Att) ? 3 :
		    SUBSET ? Subsets[Att] : Context->schema.max_attribute_value[Att] );

	    if ( Val > BestVal ||
		 ( Val > 0.999 * BestVal &&
		   ( NBr < BestNBr ||
		     ( NBr == BestNBr && Gain[Att] > Gain[BestAtt] ) ) ) )
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


void EvalDiscrSplit(c50_context *Context, Attribute Att, CaseCount Cases)
/*   --------------  */
{
    DiscrValue	v, NBr;

    Gain[Att] = None;

    if ( Skip(Att) || Att == Context->schema.class_attribute ) return;

    if ( Ordered(Att) )
    {
	EvalOrderedAtt(Context, Att, Cases);
	NBr = ( GEnv.ValFreq[1] > 0.5 ? 3 : 2 );
    }
    else
    if ( SUBSET && Context->schema.max_attribute_value[Att] > 3 )
    {
	EvalSubset(Context, Att, Cases);
	NBr = Subsets[Att];
    }
    else
    if ( ! Tested[Att] )
    {
	EvalDiscreteAtt(Context, Att, Cases);

	NBr = 0;
	ForEach(v, 1, Context->schema.max_attribute_value[Att])
	{
	    if ( GEnv.ValFreq[v] > 0.5 ) NBr++;
	}
    }
    else
    {
	NBr = 0;
    }

    /*  Check that this test will not give too many leaves  */

    if ( NBr > MaxLeaves + 1 )
    {
	Verbosity(2,
	    fprintf(Of, "\t(cancelled -- %d leaves, max %d)\n", NBr, MaxLeaves))

	Gain[Att] = None;
    }
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

    PrevUnitWeights = UnitWeights;

    Att = T->Tested;
    Missing = (Ep = Group(Context, 0, Fp, Lp, T)) - Fp + 1;

    KnownCases = T->Cases - (MissingCases = CountCases(Fp, Ep));

    if ( Missing )
    {
	UnitWeights = false;

	/*  If using costs, must adjust branch factors to undo effects of
	    reweighting cases  */

	if ( CostWeights )
	{
	    KnownCases = SumNocostWeights(Ep+1, Lp);
	}

	/*  If there are many cases with missing values and many branches,
	    skip cases whose weight < 0.1  */

	if ( (Cases = No(Fp,Lp)) > 1000 &&
	     Missing > 0.5 * Cases &&
	     T->Forks >= 10 )
	{
	    ForEach(i, Fp, Ep)
	    {
		if ( Weight(Case[i]) < 0.1 )
		{
		    Missing--;
		    MissingCases -= Weight(Case[i]);
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

	BranchCases = CountCases(Bp + Missing, Ep);

	Factor = ( ! Missing ? 0 :
		   ! CostWeights ? BranchCases / KnownCases :
		   SumNocostWeights(Bp + Missing, Ep) / KnownCases );

	if ( BranchCases + Factor * MissingCases >= MinLeaf )
	{
	    if ( Missing )
	    {
		/*  Adjust weights of cases with missing values  */

		ForEach(i, Bp, Bp + Missing - 1)
		{
		    Weight(Case[i]) *= Factor;
		}
	    }

	    FormTree(Context, Bp, Ep, Level+1, &T->Branch[v]);

	    /*  Restore weights if changed  */

	    if ( Missing )
	    {
		for ( i = Ep ; i >= Bp ; i-- )
		{
		    if ( Unknown(Case[i], Att) )
		    {
			Weight(Case[i]) /= Factor;
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

    UnitWeights = PrevUnitWeights;
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

	if ( SomeMiss[Att] )
	{
	    ForEach(i, Bp, Ep)
	    {
		if ( Unknown(Case[i], Att) )
		{
		    Swap(Bp, i);
		    Bp++;
		}
	    }
	}
    }
    else				/* skip non-existant N/A values */
    if ( V != 1 || TestNode->NodeType == BrSubset || SomeNA[Att] )
    {
	/*  Group cases on the value of attribute Att, and depending
	    on the type of branch  */

	switch ( TestNode->NodeType )
	{
	    case BrDiscr:

		ForEach(i, Bp, Ep)
		{
		    if ( DVal(Case[i], Att) == V )
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
		    if ( V == 1 ? NotApplic(Context, Case[i], Att) :
			 (CVal(Case[i], Att) <= Thresh) == (V == 2) )
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
		    if ( In(XDVal(Case[i], Att), SS) )
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


CaseCount SumWeights(CaseNo Fp, CaseNo Lp)
/*        ----------  */
{
    double	Sum=0.0;
    CaseNo	i;

    assert(Fp >= 0 && Lp >= Fp-1 && Lp <= MaxCase);

    ForEach(i, Fp, Lp)
    {
	Sum += Weight(Case[i]);
    }

    return Sum;
}



/*************************************************************************/
/*								 	 */
/*	Special version to undo the weightings associated with costs	 */
/*								 	 */
/*************************************************************************/


CaseCount SumNocostWeights(CaseNo Fp, CaseNo Lp)
/*        ----------------  */
{
    double	Sum=0.0;
    CaseNo	i;

    assert(Fp >= 0 && Lp >= Fp-1 && Lp <= MaxCase);

    ForEach(i, Fp, Lp)
    {
	Sum += Weight(Case[i]) / WeightMul[Class(Case[i])];
    }

    return Sum;
}



/*************************************************************************/
/*                                                               	 */
/*	Generate class frequency distribution				 */
/*									 */
/*************************************************************************/


void FindClassFreq(c50_context *Context, double *CF, CaseNo Fp, CaseNo Lp)
/*   -------------  */
{
    ClassNo	c;
    CaseNo	i;

    assert(Fp >= 0 && Lp >= Fp && Lp <= MaxCase);

    ForEach(c, 0, Context->schema.max_class)
    {
	CF[c] = 0;
    }

    ForEach(i, Fp, Lp)
    {
	assert(Class(Case[i]) >= 1 && Class(Case[i]) <= Context->schema.max_class);

	CF[ Class(Case[i]) ] += Weight(Case[i]);
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
	ClassFreq[c] = 0;
    }

    for ( a = 0 ; a < NDList ; a++ )
    {
	Att = DList[a];
	for ( x = Context->schema.max_class * (Context->schema.max_attribute_value[Att]+1) - 1 ; x >= 0 ; x-- )
	{
	    DFreq[Att][x] = 0;
	}
    }

    /*  Scan cases  */

    ForEach(i, Fp, Lp)
    {
	ClassFreq[ (c=Class(Case[i])) ] += (w=Weight(Case[i]));

	for ( a = 0 ; a < NDList ; a++ )
	{
	    Att = DList[a];
	    DFreq[Att][ Context->schema.max_class * XDVal(Case[i], Att) + (c-1) ] += w;
	}
    }
}
