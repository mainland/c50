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
/*	Routines for building a rule tree for faster classification.	 */
/*	A ruletree node consists of					 */
/*	* a list of rules satisfied at this node, terminated by 0	 */
/*	* a new test							 */
/*	* subtrees for each outcome (with branch 0 dealing with those	 */
/*	  rules that do not contain the new test)			 */
/*                                                              	 */
/*************************************************************************/


#include "defns.i"
#include "extern.i"
#include "c50_api_internal.h"

/*************************************************************************/
/*                                                              	 */
/*    Construct ruletree for ruleset RS					 */
/*                                                              	 */
/*************************************************************************/


void ConstructRuleTree(c50_context *Context, CRuleSet RS)
/*   -----------------  */
{
    int		r, c;
    RuleNo	*All;

    Context->rule_tree.tests = Alloc((Context->rule_tree.test_capacity = 1000), Condition);
    Context->rule_tree.test_count = 0;

    All = Alloc(RS->SNRules, RuleNo);
    ForEach(r, 1, RS->SNRules)
    {
	All[r-1] = r;

	ForEach(c, 1, RS->SRule[r]->Size)
	{
	    SetTestIndex(Context, RS->SRule[r]->Lhs[c]);
	}
    }

    Context->rule_tree.test_occurrences = Alloc(Context->rule_tree.test_count, int);
    Context->rule_tree.tests_used  = AllocZero(Context->rule_tree.test_count, Boolean);

    Context->rule_tree.rule_conditions_satisfied = AllocZero(RS->SNRules+1, int);

    RS->RT = GrowRT(Context, All, RS->SNRules, RS->SRule);

    Free(All);
    Free(Context->rule_tree.tests);
    Free(Context->rule_tree.tests_used);
    Free(Context->rule_tree.test_occurrences);
    Free(Context->rule_tree.rule_conditions_satisfied);
}



/*************************************************************************/
/*                                                              	 */
/*	Set test number for a condition.  If no existing test matches,	 */
/*	add new test to Context->rule_tree.tests[]						 */
/*                                                              	 */
/*************************************************************************/


void SetTestIndex(c50_context *Context, Condition C)
/*   ------------  */
{
    int		t;
    Condition	CC;
    Attribute	Att;

    Att = C->Tested;

    ForEach(t, 0, Context->rule_tree.test_count-1)
    {
	CC = Context->rule_tree.tests[t];
	if ( CC->Tested != Att || CC->NodeType != C->NodeType ) continue;

	switch ( C->NodeType )
	{
	    case BrDiscr:
		C->TestI = t;
		return;

	    case BrSubset:
		if ( ! memcmp(C->Subset, CC->Subset, (Context->schema.max_attribute_value[Att]>>3)+1) )
		{
		    C->TestI = t;
		    return;
		}
		break;

	    case BrThresh:
		if ( ( C->TestValue == 1 && CC->TestValue == 1 ) ||
		     ( C->TestValue != 1 && CC->TestValue != 1 &&
		       C->Cut == CC->Cut ) )
		{
		    C->TestI = t;
		    return;
		}
		break;
	}
    }

    /*  New test -- make sure have enough space  */

    if ( Context->rule_tree.test_count >= Context->rule_tree.test_capacity )
    {
	Realloc(Context->rule_tree.tests, (Context->rule_tree.test_capacity += 1000), Condition);
    }

    Context->rule_tree.tests[Context->rule_tree.test_count] = C;
    C->TestI = Context->rule_tree.test_count++;
}



/*************************************************************************/
/*                                                              	 */
/*	Construct ruletree for rules RR					 */
/*                                                              	 */
/*************************************************************************/


RuleTree GrowRT(c50_context *Context, RuleNo *RR, int RRN, CRule *Rule)
/*       ------  */
{
    RuleTree	Node;
    RuleNo	r, *LR;
    int		FP=0, ri, TI, *Expect, LRN;
    size_t	FireCount;
    DiscrValue	v;

    if ( RRN <= 0 ) return Nil;

    Node = AllocZero(1, RuleTreeRec);

    /*  Record and swap to front any rules that are satisfied  */

    ForEach(ri, 0, RRN-1)
    {
	r = RR[ri];

	if ( Context->rule_tree.rule_conditions_satisfied[r] == Rule[r]->Size )
	{
	    RR[ri] = RR[FP];
	    RR[FP] = r;
	    FP++;
	}
    }

    if ( FP > 0 )
    {
	FireCount = (size_t) FP;
	Node->Fire = Alloc(FireCount+1, RuleNo);
	memcpy(Node->Fire, RR, FireCount * sizeof(RuleNo));
	Node->Fire[FireCount] = 0;
	RR  += FP;
	RRN -= FP;
    }
    else
    {
	Node->Fire = Nil;
    }

    if ( ! RRN ) return Node;

    /*  Choose test for this node  */

    TI = SelectTest(Context, RR, RRN, Rule);
    Context->rule_tree.tests_used[TI] = true;

    Node->CondTest = Context->rule_tree.tests[TI];

    /*  Find the desired outcome for each rule  */

    Expect = Alloc(RRN, int);
    ForEach(ri, 0, RRN-1)
    {
	Expect[ri] = DesiredOutcome(Context, Rule[RR[ri]], TI);
    }

    /*  Now construct individual branches.  Rules that do not reference
	the selected test go down branch 0; at classification time,
	any case with an unknown outcome for the selected test also
	goes to branch 0.  */

    Node->Forks =
	( Context->rule_tree.tests[TI]->NodeType == BrDiscr ? Context->schema.max_attribute_value[Context->rule_tree.tests[TI]->Tested] :
	  Context->rule_tree.tests[TI]->NodeType == BrSubset ? 1 : 3 );

    Node->Branch = Alloc(Node->Forks+1, RuleTree);

    LR = Alloc(RRN, RuleNo);
    ForEach(v, 0, Node->Forks)
    {
	/*  Extract rules with outcome v and increment conditions satisfied,
	    if relevant  */

	LRN = 0;
	ForEach(ri, 0, RRN-1)
	{
	    if ( abs(Expect[ri]) == v )
	    {
		LR[LRN++] = RR[ri];

		if ( Expect[ri] > 0 ) Context->rule_tree.rule_conditions_satisfied[RR[ri]]++;
	    }
	}

	/*  LR now contains rules with outcome v  */

	Node->Branch[v] = GrowRT(Context, LR, LRN, Rule);

	if ( v )
	{
	    /*  Restore conditions satisfied  */

	    ForEach(ri, 0, LRN-1)
	    {
		Context->rule_tree.rule_conditions_satisfied[LR[ri]]--;
	    }
	}
    }

    Context->rule_tree.tests_used[TI] = false;

    /*  Free local storage  */

    Free(LR);
    Free(Expect);

    return Node;
}



/*************************************************************************/
/*                                                              	 */
/*	Check whether rule uses Context->rule_tree.tests[TI].				 */
/*	Return 0 (no) or test outcome required for rule			 */
/*                                                              	 */
/*************************************************************************/


int DesiredOutcome(c50_context *Context, CRule R, int TI)
/*  --------------  */
{
    int		c;
    Boolean	ContinTest;

    ContinTest = Continuous(Context->rule_tree.tests[TI]->Tested);	/* test of continuous att */

    ForEach(c, 1, R->Size)
    {
	if ( R->Lhs[c]->TestI == TI )
	{
	    return R->Lhs[c]->TestValue;
	}

	/*  If this test references the same continuous attribute but
	    with a different threshold, may be able to exploit outcome:
	      -2 means "rule can only be matched down branch 2"
	      -3 means "rule can only be matched down branch 3"  */

	if ( ContinTest && Context->rule_tree.tests[TI]->Tested == R->Lhs[c]->Tested )
	{
	    switch ( R->Lhs[c]->TestValue )
	    {
		case 1:
		    return 1;

		case 2:
		    if ( R->Lhs[c]->Cut < Context->rule_tree.tests[TI]->Cut ) return -2;
		    break;

		case 3:
		    if ( R->Lhs[c]->Cut > Context->rule_tree.tests[TI]->Cut ) return -3;
	    }
	}
    }

    return 0;
}



/*************************************************************************/
/*                                                              	 */
/*	Select most frequently-occurring test to partition rules in RR	 */
/*                                                              	 */
/*************************************************************************/


int SelectTest(c50_context *Context, RuleNo *RR, int RRN, CRule *Rule)
/*  ----------  */
{
    int		c, cc, ri;
    RuleNo	r;

    /*  Count test occurrences  */

    ForEach(c, 0, Context->rule_tree.test_count-1)
    {
	Context->rule_tree.test_occurrences[c] = 0;
    }

    ForEach(ri, 0, RRN-1)
    {
	r = RR[ri];

	ForEach(c, 1, Rule[r]->Size)
	{
	    Context->rule_tree.test_occurrences[Rule[r]->Lhs[c]->TestI]++;
	}
    }

    /*  Find most frequently-occurring test  */

    cc = -1;
    ForEach(c, 0, Context->rule_tree.test_count-1)
    {
	if ( ! Context->rule_tree.tests_used[c] && ( cc < 0 || Context->rule_tree.test_occurrences[c] > Context->rule_tree.test_occurrences[cc] ) )
	{
	    cc = c;
	}
    }

    return cc;
}



/*************************************************************************/
/*                                                              	 */
/*	Free ruletree							 */
/*                                                              	 */
/*************************************************************************/


void FreeRuleTree(RuleTree RT)
/*   ------------  */
{
    int		b;

    if ( ! RT ) return;

    if ( RT->Branch )
    {
	ForEach(b, 0, RT->Forks )
	{
	    FreeRuleTree(RT->Branch[b]);
	}
	Free(RT->Branch);
    }

    /*  Don't free RT->Cond since this is just a pointer to a condition
	in one of the rules  */

    FreeUnlessNil(RT->Fire);
    Free(RT);
}
