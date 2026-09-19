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
/*                                                              	 */
/*	Determine the class of a case from a decision tree or ruleset	 */
/*                                                              	 */
/*************************************************************************/


#include "defns.i"
#include "extern.i"
#include "c50_api_internal.h"



/*************************************************************************/
/*                                                              	 */
/*	Classify a case using a decision tree				 */
/*                                                              	 */
/*************************************************************************/


ClassNo TreeClassify(c50_context *Context, DataRec Case, Tree DecisionTree)
/*      ------------  */
{
    ClassNo	c;

    ForEach(c, 0, MaxClass)
    {
	Context->class_sum[c] = 0;
    }

    FindLeaf(Context, Case, DecisionTree, Nil, 1.0);

    return SelectClass(Context, 1, (Boolean)(MCost != Nil));
}



/*************************************************************************/
/*                                                              	 */
/*	Classify a case using the given subtree.			 */
/*	Adjust the class sum for each class				 */
/*                                                              	 */
/*************************************************************************/


void FindLeaf(c50_context *Context, DataRec Case, Tree T, Tree PT,
	      float Fraction)
/*   --------  */
{
    DiscrValue	v, Dv;
    ClassNo	c;
    float	NewFrac, BrWt[4];

    /*  Special case for winnowing cycles  */

    if ( T->NodeType && Skip(T->Tested) )
    {
	FollowAllBranches(Context, Case, T, Fraction);
	return;
    }

    if ( T->NodeType && Tested )
    {
	Tested[T->Tested] = true;	/* for usage */
    }

    switch ( T->NodeType )
    {
	case 0:  /* leaf */

	  LeafUpdate:

	    /*  Use parent node if effectively no cases at this node  */

	    if ( T->Cases < Epsilon )
	    {
		T = PT;
	    }

	    /*  Update from all classes  */

	    ForEach(c, 1, MaxClass)
	    {
		Context->class_sum[c] += Fraction * T->ClassDist[c] / T->Cases;
	    }

	    return;

	case BrDiscr:  /* test of discrete attribute */

	    Dv = DVal(Case, T->Tested);	/* > MaxAttVal if unknown */

	    if ( Dv <= T->Forks )	/*  Make sure not new discrete value  */
	    {
		FindLeaf(Context, Case, T->Branch[Dv], T, Fraction);
	    }
	    else
	    {
		FollowAllBranches(Context, Case, T, Fraction);
	    }

	    return;

	case BrThresh:  /* test of continuous attribute */

	    if ( Unknown(Case, T->Tested) )
	    {
		FollowAllBranches(Context, Case, T, Fraction);
	    }
	    else
	    if ( NotApplic(Case, T->Tested) )
	    {
		FindLeaf(Context, Case, T->Branch[1], T, Fraction);
	    }
	    else
	    {
		/*  Find weights for <= and > branches, interpolating if
		    probabilistic thresholds are used  */

		BrWt[2] = Interpolate(T, CVal(Case, T->Tested));
		BrWt[3] = 1 - BrWt[2];

		ForEach(v, 2, 3)
		{
		    if ( (NewFrac = Fraction * BrWt[v]) >= 0.01 )
		    {
			FindLeaf(Context, Case, T->Branch[v], T, NewFrac);
		    }
		}
	    }

	    return;

	case BrSubset:  /* subset test on discrete attribute  */

	    Dv = DVal(Case, T->Tested);	/* > MaxAttVal if unknown */

	    if ( Dv <= MaxAttVal[T->Tested] )
	    {
		ForEach(v, 1, T->Forks)
		{
		    if ( In(Dv, T->Subset[v]) )
		    {
			FindLeaf(Context, Case, T->Branch[v], T, Fraction);

			return;
		    }
		}

		/* Value not found in any subset -- treat as leaf  */

		goto LeafUpdate;
	    }
	    else
	    {
		FollowAllBranches(Context, Case, T, Fraction);
	    }
    }
}



/*************************************************************************/
/*                                                              	 */
/*	Follow all branches from a node, weighting them in proportion	 */
/*	to the number of training cases they contain			 */
/*                                                              	 */
/*************************************************************************/


void FollowAllBranches(c50_context *Context, DataRec Case, Tree T,
		       float Fraction)
/*   -----------------  */
{
    DiscrValue	v;

    ForEach(v, 1, T->Forks)
    {
	if ( T->Branch[v]->Cases > Epsilon )
	{
	    FindLeaf(Context, Case, T->Branch[v], T,
		     (Fraction * T->Branch[v]->Cases) / T->Cases);
	}
    }
}



/*************************************************************************/
/*                                                              	 */
/*	Classify a case using a ruleset					 */
/*                                                              	 */
/*************************************************************************/


ClassNo RuleClassify(c50_context *Context, DataRec Case, CRuleSet RS)
/*      ------------  */
{
    ClassNo	c, Best;
    float	TotWeight=0;
    int		a, u=1, d;
    CRule	R;
    RuleNo	r;

    ForEach(c, 0, MaxClass)
    {
	Context->class_sum[c] = 0;
	Context->most_specific_rules[c] = Nil;
    }

    /*  Find active rules  */

    Context->active_rule_count = 0;

    if ( RS->RT )
    {
	MarkActive(Context, RS->RT, Case);
    }
    else
    {
	ForEach(r, 1, RS->SNRules)
	{
	    R = RS->SRule[r];

	    if ( Matches(R, Case) )
	    {
		Context->active_rules[Context->active_rule_count++] = r;
	    }
	}
    }

    /*  Must sort rules if using utility bands  */

    if ( UtilBand )
    {
	SortActive(Context);
    }

    /*  Vote active rules  */

    ForEach(a, 0, Context->active_rule_count-1)
    {
	r = Context->active_rules[a];
	R = RS->SRule[r];

	if ( Tested )
	{
	    ForEach(d, 1, R->Size)
	    {
		Tested[R->Lhs[d]->Tested] = true;	/* for usage */
	    }
	}
	if ( UtilBand )
	{
	    CheckUtilityBand(Context, &u, r, Class(Case), RS->SDefault);
	}
	Context->class_sum[R->Rhs] += R->Vote;
	TotWeight        += 1000.0;

	/*  Check whether this is the most specific rule for this class;
	    resolve ties in favor of rule with higher vote  */

	if ( ! Context->most_specific_rules[R->Rhs] ||
	     R->Cover < Context->most_specific_rules[R->Rhs]->Cover ||
	     ( R->Cover == Context->most_specific_rules[R->Rhs]->Cover &&
	       R->Vote > Context->most_specific_rules[R->Rhs]->Vote ) )
	{
	    Context->most_specific_rules[R->Rhs] = R;
	}
    }

    /*  Flush any remaining utility bands  */

    if ( UtilBand )
    {
	CheckUtilityBand(Context, &u, RS->SNRules+1, Class(Case),
			 RS->SDefault);
    }

    /*  Check for default and normalise class sums  */

    if ( ! TotWeight )
    {
	Context->confidence = 0.5;
	return RS->SDefault;
    }

    ForEach(c, 1, MaxClass)
    {
	Context->class_sum[c] /= TotWeight;
    }

    Best = SelectClass(Context, RS->SDefault, false);

    /*  Set confidence to the vote for the most specific rule of class Best  */

    Context->confidence = Context->most_specific_rules[Best]->Vote / 1000.0;

    return Best;
}



/*************************************************************************/
/*                                                              	 */
/*	Determine outcome of a test on a case.				 */
/*	Return -1 if value of tested attribute is unknown		 */
/*                                                              	 */
/*************************************************************************/


int FindOutcome(DataRec Case, Condition OneCond)
/*  -----------  */
{
    DiscrValue  v, Outcome;
    Attribute	Att;

    Att = OneCond->Tested;

    /*  Determine the outcome of this test on this case  */

    switch ( OneCond->NodeType )
    {
	case BrDiscr:  /* test of discrete attribute */

	    v = XDVal(Case, Att);
	    Outcome = ( v == 0 ? -1 : v );
	    break;

	case BrThresh:  /* test of continuous attribute */

	    Outcome = ( Unknown(Case, Att) ? -1 :
			NotApplic(Case, Att) ? 1 :
			CVal(Case, Att) <= OneCond->Cut ? 2 : 3 );
	    break;

	case BrSubset:  /* subset test on discrete attribute  */

	    v = XDVal(Case, Att);
	    Outcome = ( v <= MaxAttVal[Att] && In(v, OneCond->Subset) ?
			OneCond->TestValue : 0 );
    }

    return Outcome;
}



/*************************************************************************/
/*									 */
/*	Determine whether a case satisfies a condition			 */
/*									 */
/*************************************************************************/


Boolean Satisfies(DataRec Case, Condition OneCond)
/*      ---------  */
{
    return ( FindOutcome(Case, OneCond) == OneCond->TestValue );
}



/*************************************************************************/
/*									 */
/*	Determine whether a case satisfies all conditions of a rule	 */
/*									 */
/*************************************************************************/


Boolean Matches(CRule R, DataRec Case)
/*      -------  */
{
    int d;

    ForEach(d, 1, R->Size)
    {
	if ( ! Satisfies(Case, R->Lhs[d]) )
	{
	    return false;
	}
    }

    return true;
}



/*************************************************************************/
/*									 */
/*	Make sure that Active[] has space for at least N rules		 */
/*									 */
/*************************************************************************/


void CheckActiveSpace(c50_context *Context, int N)
/*   ----------------  */
{
    if ( Context->active_rule_capacity <= N )
    {
	Realloc(Context->active_rules,
		(Context->active_rule_capacity = N + 1), RuleNo);
    }
}



/*************************************************************************/
/*									 */
/*	Use RT to enter active rules in Active[]			 */
/*									 */
/*************************************************************************/


void MarkActive(c50_context *Context, RuleTree RT, DataRec Case)
/*   ----------  */
{
    DiscrValue	v;
    int		ri;
    RuleNo	r;

    if ( ! RT ) return;

    /*  Enter any rules satisfied at this node  */

    if ( RT->Fire )
    {
	for ( ri = 0 ; (r = RT->Fire[ri]) ; ri++ )
	{
	    Context->active_rules[Context->active_rule_count++] = r;
	}
    }

    if ( ! RT->Branch ) return;

    /*  Explore subtree for rules that include condition at this node  */

    if ( (v = FindOutcome(Case, RT->CondTest)) > 0 && v <= RT->Forks )
    {
	MarkActive(Context, RT->Branch[v], Case);
    }

    /*  Explore default subtree for rules that do not include condition  */

    MarkActive(Context, RT->Branch[0], Case);
}



/*************************************************************************/
/*									 */
/*	Sort active rules for utility band error rates			 */
/*									 */
/*************************************************************************/


void SortActive(c50_context *Context)
/*   ----------  */
{
    RuleNo	r;
    int		a, aa, aLow;

    ForEach(a, 0, Context->active_rule_count-1)
    {
	aLow = a;

	ForEach(aa, a+1, Context->active_rule_count-1)
	{
	    if ( Context->active_rules[aa] < Context->active_rules[aLow] )
	    {
		aLow = aa;
	    }
	}

	r = Context->active_rules[a];
	Context->active_rules[a] = Context->active_rules[aLow];
	Context->active_rules[aLow] = r;
    }
}



/*************************************************************************/
/*									 */
/*	Update utility band error rates for all bands before rule r	 */
/*	that have not been competed yet.  Update current band.		 */
/*									 */
/*************************************************************************/


void CheckUtilityBand(c50_context *Context, int *u, RuleNo r,
		      ClassNo Actual, ClassNo Default)
/*   ----------------  */
{
    ClassNo	c;

    while ( *u < UTILITY && r > UtilBand[*u] )
    {
	c = SelectClass(Context, Default, false);
	if ( c != Actual )
	{
	    UtilErr[*u]++;
	    if ( MCost ) UtilCost[*u] += MCost[c][Actual];
	}

	(*u)++;
    }
}



/*************************************************************************/
/*									 */
/*	Classify a case using boosted tree or rule sequence.		 */
/*	Global variable Default must have been set prior to call	 */
/*									 */
/*	Note: boosting with costs is complicated.  With trees,		 */
/*	complete class distributions are accumulated and then a class	 */
/*	selected to minimize expected cost.  This cannot be done with	 */
/*	rulesets since a single ruleset does not give a reliable	 */
/*	class distribution; instead, the votes from all cost-adjusted	 */
/*	rulesets are combined without reference to costs.		 */
/*									 */
/*************************************************************************/


ClassNo BoostClassify(c50_context *Context, DataRec Case, int MaxTrial)
/*	-------------  */
{
    ClassNo	c, Best;
    int		t;
    float	Total=0;

    ForEach(c, 1, MaxClass)
    {
	Context->votes[c] = 0;
    }

    ForEach(t, 0, MaxTrial)
    {
	Best = ( RULES ? RuleClassify(Context, Case, RuleSet[t]) :
		 TreeClassify(Context, Case, Pruned[t]) );

	Context->votes[Best] += Context->confidence;
	Total += Context->confidence;

	Context->trial_predictions[t] = Best;
    }

    /*  Copy votes into the class sums  */

    ForEach(c, 1, MaxClass)
    {
	Context->class_sum[c] = Context->votes[c] / Total;
    }

    return SelectClass(Context, Default, false);
}



/*************************************************************************/
/*									 */
/*	Select the best class to return.  Take misclassification costs	 */
/*	into account if they are defined.				 */
/*									 */
/*************************************************************************/


ClassNo SelectClass(c50_context *Context, ClassNo Default, Boolean UseCosts)
/*      -----------  */
{
    ClassNo	c, cc, BestClass;
    float	ExpCost, BestCost=1E38, TotCost=0;

    BestClass = Default;

    if ( UseCosts )
    {
	ForEach(c, 1, MaxClass)
	{
	    ExpCost = 0;
	    ForEach(cc, 1, MaxClass)
	    {
		if ( cc == c ) continue;
		ExpCost += Context->class_sum[cc] * MCost[c][cc];
	    }

	    TotCost += ExpCost;

	    if ( ExpCost < BestCost )
	    {
		BestClass = c;
		BestCost  = ExpCost;
	    }
	}

	Context->confidence = 1 - BestCost / TotCost;
    }
    else
    {
	ForEach(c, 1, MaxClass)
	{
	    if ( Context->class_sum[c] > Context->class_sum[BestClass] ) BestClass = c;
	}

	Context->confidence = Context->class_sum[BestClass];
    }

    return BestClass;
}



/*************************************************************************/
/*								   	 */
/*	General classification routine					 */
/*								   	 */
/*************************************************************************/


ClassNo Classify(c50_context *Context, DataRec Case)
/*      --------  */
{

    return ( TRIALS > 1 ? BoostClassify(Context, Case, TRIALS-1) :
	     RULES ?	  RuleClassify(Context, Case, RuleSet[0]) :
		  TreeClassify(Context, Case, Pruned[0]) );
}



/*************************************************************************/
/*								   	 */
/*	Interpolate a single value between Lower, Mid and Upper		 */
/*	(All these have the same value unless using probabilistic	 */
/*	thresholds.)							 */
/*								   	 */
/*************************************************************************/


float Interpolate(Tree T, ContValue Val)
/*    -----------  */
{
    return ( Val <= T->Lower ? 1.0 :
	     Val >= T->Upper ? 0.0 :
	     Val <= T->Mid ?
		1 - 0.5 * (Val - T->Lower) / (T->Mid - T->Lower + 1E-6) :
		0.5 - 0.5 * (Val - T->Mid) / (T->Upper - T->Mid + 1E-6) );
}



/*************************************************************************/
/*									 */
/*	Free data structures for one classifier				 */
/*									 */
/*************************************************************************/


void FreeClassifier(int Trial)
/*   --------------  */
{
    if ( Raw )
    {
	FreeTree(Raw[Trial]);				Raw[Trial] = Nil;
    }

    if ( Pruned )
    {
	FreeTree(Pruned[Trial]);			Pruned[Trial] = Nil;
    }

    if ( RULES && RuleSet && RuleSet[Trial] )
    {
	FreeRules(RuleSet[Trial]);			RuleSet[Trial] = Nil;
    }
}
