/*************************************************************************/
/*									 */
/*  Copyright 2010 Rulequest Research Pty Ltd.				 */
/*  Author: Ross Quinlan (quinlan@rulequest.com) [Rev Jan 2016]		 */
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
/*		Definitions used in C5.0				 */
/*              ------------------------				 */
/*									 */
/*************************************************************************/


#define	 RELEASE	"2.07 GPL Edition"

				/*  Uncomment following line to enable
				    sample estimates for large datasets.
				    This can lead to some variablility,
				    especially when used with SMP  */
//#define	SAMPLE_ESTIMATES

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <ctype.h>
#include <limits.h>
#include <float.h>

#include <c50/c50.h>

#include "c50_input.h"
#include "c50_rng.h"
#include "text.i"



/*************************************************************************/
/*									 */
/*		Definitions dependent on cc options			 */
/*									 */
/*************************************************************************/


#define Goodbye(x)		C50Exit(x)

#ifdef	VerbOpt
#include <assert.h>
#define	Verbosity(d,s)		if(Context->options.verbosity >= d) {s;}
#else
#define	 assert(x)
#define Verbosity(d,s)
#endif


/*  Alternative random number generator  */

#define AltRandom		drand48()
#define	AltSeed(x)		srand48(x)

#define Free(x)			{free(x); x=0;}


/*************************************************************************/
/*									 */
/*		Constants, macros etc.					 */
/*									 */
/*************************************************************************/

#define	 THEORYFRAC	0.23	/* discount rate for estimated coding cost */

#define	 Nil	   0		/* null pointer */
#define	 false	   0
#define	 true	   1
#define	 None	   -1
#define	 Epsilon   1E-4
#define	 MinLeaf   0.05		/* minimum weight for non-null leaf */
#define	 Width	   80		/* approx max width of output */

#define  EXCLUDE   1		/* special attribute status: do not use */
#define  SKIP	   2		/* do not use in classifiers */
#define  DISCRETE  4		/* ditto: collect values as data read */
#define  ORDERED   8		/* ditto: ordered discrete values */
#define  DATEVAL   16		/* ditto: YYYY/MM/DD or YYYY-MM-DD */
#define  STIMEVAL  32		/* ditto: HH:MM:SS */
#define	 TSTMPVAL  64		/* date time */

#define	 CMINFO	   1		/* generate confusion matrix */
#define	 USAGEINFO 2		/* print usage information */

				/* unknown and N/A values are represented by
				   unlikely floating-point numbers
				   (octal 01600000000 and 01) */
#define	 UNKNOWN   01600000000	/* 1.5777218104420236e-30 */
#define	 NA	   01		/* 1.4012984643248171e-45 */

#define	 BrDiscr   1
#define	 BrThresh  2
#define	 BrSubset  3

#define  Plural(n)		((n) != 1 ? "s" : "")

#define  AllocZero(N,T)		(T *) Pcalloc(N, sizeof(T))
#define  Alloc(N,T)		AllocZero(N,T) /* for safety */
#define  Realloc(V,N,T)		V = (T *) Prealloc(V, (N)*sizeof(T))

#define	 Max(a,b)               ((a)>(b) ? (a) : (b))
#define	 Min(a,b)               ((a)<(b) ? (a) : (b))

#define	 Log2			0.69314718055994530942
#define	 Log(x)			((x) <= 0 ? 0.0 : log((double)x) / Log2)

#define	 Bit(b)			(1 << (b))
#define	 In(b,s)		((s[(b) >> 3]) & Bit((b) & 07))
#define	 ClearBits(n,s)		memset(s,0,n)
#define	 CopyBits(n,f,t)	memcpy(t,f,n)
#define	 SetBit(b,s)		(s[(b) >> 3] |= Bit((b) & 07))
#define	 ResetBit(b,s)		(s[(b) >> 3] ^= Bit((b) & 07))

#define	 ForEach(v,f,l)		for(v=f ; v<=l ; ++v)

#define	 CountCases(Context,f,l) \
	(Context->costs.unit_weights ? (l-(f)+1.0) : SumWeights(Context,f,l))

#define	 StatBit(a,b)		(Context->schema.special_status[a]&(b))
#define	 Exclude(a)		StatBit(a,EXCLUDE)
#define	 Skip(a)		StatBit(a,EXCLUDE|SKIP)
#define  Discrete(a)		(Context->schema.max_attribute_value[a] || StatBit(a,DISCRETE))
#define  Continuous(a)		(! Context->schema.max_attribute_value[a] && ! StatBit(a,DISCRETE))
#define	 Ordered(a)		StatBit(a,ORDERED)
#define	 DateVal(a)		StatBit(a,DATEVAL)
#define	 TimeVal(a)		StatBit(a,STIMEVAL)
#define	 TStampVal(a)		StatBit(a,TSTMPVAL)

#define  FreeUnlessNil(p)	if((p)!=Nil) Free(p)

#define  CheckClose(f)		if(f) {fclose(f); f=Nil;}

#define	 Int(x)			((int)(x+0.5))

#define  Space(s)	(s == ' ' || s == '\n' || s == '\r' || s == '\t')
#define	 P1(x)		(rint((x)*10) / 10)

#define	 No(f,l)	((l)-(f)+1)

#define	 EmptyNA(T)	(T->Branch[1]->Cases < 0.01)

#define  Before(n1,n2)  (n1->Tested < n2->Tested ||\
			(n1->Tested == n2->Tested && n1->Cut < n2->Cut))

#define	 Swap(a,b)	{DataRec xab;\
			 assert(a >= 0 && a <= Context->cases.max_case &&\
			        b >= 0 && b <= Context->cases.max_case);\
			 xab = Context->cases.records[a];\
			 Context->cases.records[a] = Context->cases.records[b];\
			 Context->cases.records[b] = xab;}


#define	 NOFILE		 0
#define	 BADCLASSTHRESH	 1
#define	 LEQCLASSTHRESH	 2
#define	 BADATTNAME	 3
#define	 EOFINATT	 4
#define	 SINGLEATTVAL	 5
#define	 BADATTVAL	 6
#define	 BADNUMBER	 7
#define	 BADCLASS	 8
#define	 BADCOSTCLASS	 9
#define	 BADCOST	10
#define	 NOMEM		11
#define	 TOOMANYVALS	12
#define	 BADDISCRETE	13
#define	 NOTARGET	14
#define	 BADCTARGET	15
#define	 BADDTARGET	16
#define	 LONGNAME	17
#define	 HITEOF		18
#define	 MISSNAME	19
#define	 BADDATE	20
#define	 BADTIME	21
#define	 BADTSTMP	22
#define	 DUPATTNAME	23
#define	 UNKNOWNATT	24
#define	 BADDEF1	25
#define	 BADDEF2	26
#define	 BADDEF3	27
#define	 BADDEF4	28
#define	 SAMEATT	29
#define	 MODELFILE	30
#define	 CWTATTERR	31

#define	 READDATA	 1
#define	 WINNOWATTS	 2
#define	 FORMTREE	 3
#define	 SIMPLIFYTREE	 4
#define	 FORMRULES	 5
#define	 SIFTRULES	 6
#define	 EVALTRAIN	 7
#define	 READTEST	 8
#define	 EVALTEST	 9
#define	 CLEANUP	10
#define	 ALLOCTABLES	11
#define	 RESULTS	12
#define	 READXDATA	13


/*************************************************************************/
/*									 */
/*		Type definitions					 */
/*									 */
/*************************************************************************/


typedef  unsigned char	Boolean, BranchType, *Set, Byte;
typedef	 char		*String;

typedef  int	CaseNo;		/* data case number */
typedef  float	CaseCount;	/* count of (partial) cases */

typedef  int	ClassNo,	/* class number, 1..Context->schema.max_class */
		DiscrValue,	/* discrete attribute value */
		Attribute;	/* attribute number, 1..Context->schema.max_attribute */

#ifdef USEDOUBLE
typedef	 double	ContValue;	/* continuous attribute value */
#define	 PREC	14		/* precision */
#else
typedef	 float	ContValue;	/* continuous attribute value */
#define	 PREC	 7		/* precision */
#endif


typedef  union	 _def_val
	 {
	    String	_s_val;		/* att val for comparison */
	    ContValue	_n_val;		/* number for arith */
	 }
	 DefVal;

typedef  struct  _def_elt
	 {
	    short	_op_code;	/* type of element */
	    DefVal	_operand;	/* string or numeric value */
	 }
	 DefElt, *Definition;

typedef  struct  _elt_rec
	 {
	    int		Fi,		/* index of first char of element */
			Li;		/* last ditto */
	    char	Type;		/* 'B', 'S', or 'N' */
	 }
	 EltRec;

#define	 DefOp(DE)	DE._op_code
#define	 DefSVal(DE)	DE._operand._s_val
#define	 DefNVal(DE)	DE._operand._n_val

#define	 OP_ATT			 0	/* opcodes */
#define	 OP_NUM			 1
#define	 OP_STR			 2
#define	 OP_MISS		 3
#define	 OP_AND			10
#define	 OP_OR			11
#define	 OP_EQ			20
#define	 OP_NE			21
#define	 OP_GT			22
#define	 OP_GE			23
#define	 OP_LT			24
#define	 OP_LE			25
#define	 OP_SEQ			26
#define	 OP_SNE			27
#define	 OP_PLUS		30
#define	 OP_MINUS		31
#define	 OP_UMINUS		32
#define	 OP_MULT		33
#define	 OP_DIV			34
#define	 OP_MOD			35
#define	 OP_POW			36
#define	 OP_SIN			40
#define	 OP_COS			41
#define	 OP_TAN			42
#define	 OP_LOG			43
#define	 OP_EXP			44
#define	 OP_INT			45
#define	 OP_END			99


typedef  union  _attribute_value
	 {
	    DiscrValue	_discr_val;
	    ContValue	_cont_val;
	 }
	 AttValue, *DataRec;

typedef	 struct _sort_rec
	 {
	    ContValue	V;
	    ClassNo	C;
	    float	W;
	 }
	 SortRec;

#define  CVal(Case,Att)		Case[Att]._cont_val
#define  DVal(Case,Att)		Case[Att]._discr_val
#define  XDVal(Case,Att)	(Case[Att]._discr_val & 077777777)
#define  SVal(Case,Att)		Case[Att]._discr_val
#define  Class(Case)		(*Case)._discr_val
#define  Weight(Case)		(*(Case-1))._cont_val

#define	 Unknown(Case,Att)	(DVal(Case,Att)==UNKNOWN)
#define	 UnknownVal(AV)		(AV._discr_val==UNKNOWN)
#define	 NotApplic(Context,Case,Att) \
	(Att != (Context)->schema.class_attribute && DVal(Case,Att)==NA)
#define	 NotApplicVal(AV)	(AV._discr_val==NA)

#define	 RelCWt(Context,Case) \
	( Unknown(Case,(Context)->schema.case_weight_attribute) || \
	  NotApplic(Context,Case,(Context)->schema.case_weight_attribute) || \
	  CVal(Case,(Context)->schema.case_weight_attribute) <= 0 ? 1 : \
	  CVal(Case,(Context)->schema.case_weight_attribute) / \
	      (Context)->average_case_weight )

typedef  struct _treerec	*Tree;
typedef  struct _treerec
	 {
	    BranchType	NodeType;
	    ClassNo	Leaf;		/* best class at this node */
	    CaseCount	Cases,		/* no of cases at this node */
			*ClassDist,	/* class distribution of cases */
	    		Errors;		/* est or resub errors at this node */
	    Attribute	Tested; 	/* attribute referenced in test */
	    int		Forks,		/* number of branches at this node */
			Leaves;		/* number of non-empty leaves in tree */
	    ContValue	Cut,		/* threshold for continuous attribute */
		  	Lower,		/* lower limit of soft threshold */
		  	Upper,		/* upper limit ditto */
			Mid;		/* midpoint for soft threshold */
	    Set         *Subset;	/* subsets of discrete values  */
	    Tree	*Branch,	/* Branch[x] = subtree for outcome x */
			Parent;		/* node above this one */
	 }
	 TreeRec;


typedef	 struct _environment
	 {
	    CaseNo	Xp, Ep;			/* start and end of scan  */
	    double	Cases,			/* total cases */
			KnownCases,		/* ditto less missing values */
			ApplicCases,		/* cases with numeric values */
			HighCases, LowCases,	/* cases above/below cut */
			NAInfo,			/* info for N/A values */
			FixedSplitInfo,		/* split info for ?, N/A */
			BaseInfo,		/* info before split */
			UnknownRate,		/* proportion of ? values */
			MinSplit,		/* min cases before/after cut */
			**Freq,			/* local Freq[4][class] */
			*ClassFreq,		/* local class frequencies */
			*ValFreq;		/* cases with val i */
	    ClassNo	HighClass, LowClass;	/* class after/before cut */
	    ContValue	HighVal, LowVal;	/* values after/before cut */
	    SortRec	*SRec;			/* for Cachesort() */
	    Set		**Subset,		/* Subset[att][number] */
			*WSubset;		/* working subsets */
	    int		*Subsets,		/* no of subsets for att */
			Blocks,			/* intermediate no of subsets */
			Bytes,			/* size of each subset */
			ReasonableSubsets;
	    double	*SubsetInfo,		/* subset info */
			*SubsetEntr,		/* subset entropy */
			**MergeInfo,		/* info of merged subsets i,j */
			**MergeEntr;		/* entropy ditto */
	 }
	 EnvRec;


typedef  int	RuleNo;			/* rule number */

typedef  struct _condrec
	 {
	    BranchType	NodeType;	/* test type (see tree nodes) */
	    Attribute	Tested;		/* attribute tested */
	    ContValue	Cut;		/* threshold (if relevant) */
	    Set		Subset;		/* subset (if relevant) */
	    int		TestValue,	/* specified outcome of test */
			TestI;		/* rule tree index of this test */
	 }
	 CondRec, *Condition;


typedef  struct _rulerec
	 {
	    RuleNo	RNo;		/* rule number */
	    int		TNo,		/* trial number */
	    		Size;		/* number of conditions */
	    Condition	*Lhs;		/* conditions themselves */
	    ClassNo	Rhs;		/* class given by rule */
	    CaseCount	Cover,		/* number of cases covered by rule */
			Correct;	/* number on which correct */
	    float	Prior;		/* prior probability of RHS */
	    int		Vote;		/* unit = 0.001 */
	 }
	 RuleRec, *CRule;


typedef  struct _ruletreerec *RuleTree;
typedef  struct _ruletreerec
	 {
	    RuleNo	*Fire;		/* rules matched at this node */
	    Condition	CondTest;	/* new test */
	    int		Forks;		/* number of branches */
	    RuleTree	*Branch;	/* subtrees */
	 }
	 RuleTreeRec;


typedef struct _rulesetrec
	 {
	    RuleNo	SNRules;	/* number of rules */
	    CRule	*SRule;		/* rules */
	    ClassNo	SDefault;	/* default class for this ruleset */
	    RuleTree	RT;		/* rule tree (see ruletree.c) */
	 }
	 RuleSetRec, *CRuleSet;



/*************************************************************************/
/*									 */
/*		Function prototypes					 */
/*									 */
/*************************************************************************/

	/* c50.c */

int	    main(int, char *[]);
void	    FreeClassifier(c50_context *Context, int trial);

	/* construct.c */

void	    ConstructClassifiers(c50_context *Context);
void	    InitialiseWeights(c50_context *Context);
void	    SetAvCWt(c50_context *Context);
void	    Evaluate(c50_context *Context, int Flags);
void	    EvaluateSingle(c50_context *Context, int Flags);
void	    EvaluateBoost(c50_context *Context, int Flags);
void	    RecordAttUsage(c50_context *Context, DataRec Case, int *Usage);

	/* getnames.c */

Boolean	    ReadName(c50_context *Context, FILE *f, String s, int n,
		     char ColonOpt);
Boolean	    ReadNameInput(c50_context *Context, c50_input *f, String s,
			  int n, char ColonOpt);
void	    GetNames(c50_context *Context, c50_input *Nf);
void	    ExplicitAtt(c50_context *Context, c50_input *Nf);
int	    Which(String Val, String *List, int First, int Last);
void	    ListAttsUsed(c50_context *Context);
void	    FreeNames(c50_context *Context);
int	    InChar(c50_context *Context, c50_input *f);

	/* implicitatt.c */

void	    ImplicitAtt(c50_context *Context, c50_input *Nf);
void	    ReadDefinition(c50_context *Context, c50_input *f);
void	    Append(c50_context *Context, char c);
Boolean	    Expression(c50_context *Context);
Boolean	    Conjunct(c50_context *Context);
Boolean	    SExpression(c50_context *Context);
Boolean	    AExpression(c50_context *Context);
Boolean	    Term(c50_context *Context);
Boolean	    Factor(c50_context *Context);
Boolean	    Primary(c50_context *Context);
Boolean	    Atom(c50_context *Context);
Boolean	    Find(c50_context *Context, const char *S);
int	    FindOne(c50_context *Context, const char *Alt);
Attribute   FindAttName(c50_context *Context);
void	    DefSyntaxError(c50_context *Context, String Msg);
void	    DefSemanticsError(c50_context *Context, int Fi, String Msg,
			      int OpCode);
void	    Dump(c50_context *Context, char OpCode, ContValue F, String S,
		     int Fi);
void	    DumpOp(c50_context *Context, char OpCode, int Fi);
Boolean	    UpdateTStack(c50_context *Context, char OpCode, ContValue F,
			 String S, int Fi);
AttValue    EvaluateDef(c50_context *Context, Definition D, DataRec Case);

	/* getdata.c */

void	    GetData(c50_context *Context, FILE *Df, Boolean Train,
		    Boolean AllowUnknownClass);
void	    GetDataInput(c50_context *Context, c50_input *Input, Boolean Train,
			 Boolean AllowUnknownClass);
DataRec	    GetDataRec(c50_context *Context, FILE *Df, Boolean Train);
DataRec	    GetDataRecInput(c50_context *Context, c50_input *Input,
			    Boolean Train);
CaseNo	    CountData(FILE *Df);
CaseNo	    CountDataInput(c50_input *Input);
int	    StoreIVal(c50_context *Context, String s);
void	    FreeData(c50_context *Context);
void	    CheckValue(c50_context *Context, DataRec Case, Attribute Att);

	/* mcost.c */

void	    GetMCosts(c50_context *Context, FILE *f);
void	    GetMCostsInput(c50_context *Context, c50_input *Input);

	/* attwinnow.c */

void	    WinnowAtts(c50_context *Context);
float	    TrialTreeCost(c50_context *Context, Boolean FirstTime);
float	    ErrCost(c50_context *Context, Tree T, CaseNo Fp, CaseNo Lp);
void	    ScanTree(Tree T, Boolean *Used);

	/* formtree.c */

void	    InitialiseTreeData(c50_context *Context);
void	    FreeTreeData(c50_context *Context);
void	    SetMinGainThresh(c50_context *Context);
void	    FormTree(c50_context *Context, CaseNo, CaseNo, int, Tree *);
void	    SampleEstimate(c50_context *Context, CaseNo Fp, CaseNo Lp,
			   CaseCount Cases);
void	    Sample(c50_context *Context, CaseNo Fp, CaseNo Lp, CaseNo N);
Attribute   ChooseSplit(c50_context *Context, CaseNo Fp, CaseNo Lp,
			CaseCount Cases, Boolean Sampled);
void	    ProcessQueue(c50_context *Context, CaseNo WFp, CaseNo WLp,
			 CaseCount WCases);
Attribute   FindBestAtt(c50_context *Context, CaseCount Cases);
void	    EvalDiscrSplit(c50_context *Context, Attribute Att,
			   CaseCount Cases);
CaseNo	    Group(c50_context *Context, DiscrValue, CaseNo, CaseNo, Tree);
CaseCount   SumWeights(c50_context *Context, CaseNo, CaseNo);
CaseCount   SumNocostWeights(c50_context *Context, CaseNo, CaseNo);
void	    FindClassFreq(c50_context *Context, double [], CaseNo, CaseNo);
void	    FindAllFreq(c50_context *Context, CaseNo, CaseNo);
void	    Divide(c50_context *Context, Tree Node, CaseNo Fp, CaseNo Lp,
		   int Level);

	/* discr.c */

void	    EvalDiscreteAtt(c50_context *Context, Attribute Att,
			    CaseCount Cases);
void	    EvalOrderedAtt(c50_context *Context, Attribute Att,
			   CaseCount Cases);
void	    SetDiscrFreq(c50_context *Context, Attribute Att);
double	    DiscrKnownBaseInfo(c50_context *Context, CaseCount KnownCases,
			       DiscrValue MaxVal);
void	    DiscreteTest(c50_context *Context, Tree Node, Attribute Att);

	/* contin.c */

void	    EvalContinuousAtt(c50_context *Context, Attribute Att,
			      CaseNo Fp, CaseNo Lp);
void	    EstimateMaxGR(c50_context *Context, Attribute Att, CaseNo Fp,
			  CaseNo Lp);
void	    PrepareForContin(c50_context *Context, Attribute Att, CaseNo Fp,
			     CaseNo Lp);
CaseNo	    PrepareForScan(c50_context *Context, CaseNo Lp);
void	    ContinTest(c50_context *Context, Tree Node, Attribute Att);
void	    AdjustAllThresholds(c50_context *Context, Tree T);
void	    AdjustThresholds(c50_context *Context, Tree T, Attribute Att,
			     CaseNo *Ep);
ContValue   GreatestValueBelow(c50_context *Context, ContValue Th,
			       CaseNo *Ep);

	/* info.c */

double	    ComputeGain(c50_context *Context, double BaseInfo, float UnknFrac,
			DiscrValue MaxVal, CaseCount TotalCases);
double	    TotalInfo(double V[], DiscrValue MinVal, DiscrValue MaxVal);
void	    PrintDistribution(c50_context *Context, Attribute Att,
			DiscrValue MinVal, DiscrValue MaxVal, double **Freq,
			double *ValFreq, Boolean ShowNames);

	/* subset.c */

void	    InitialiseBellNumbers(c50_context *Context);
void	    EvalSubset(c50_context *Context, Attribute Att, CaseCount Cases);
void	    Merge(c50_context *Context, DiscrValue x, DiscrValue y,
		  CaseCount Cases);
void	    EvaluatePair(c50_context *Context, DiscrValue x, DiscrValue y,
			 CaseCount Cases);
void	    PrintSubset(c50_context *Context, Attribute Att, Set Ss);
void	    SubsetTest(c50_context *Context, Tree Node, Attribute Att);
Boolean	    SameDistribution(c50_context *Context, DiscrValue V1,
			     DiscrValue V2);
void	    AddBlock(c50_context *Context, DiscrValue V1, DiscrValue V2);
void	    MoveBlock(c50_context *Context, DiscrValue V1, DiscrValue V2);

	/* prune.c */

void	    Prune(c50_context *Context, Tree T);
void	    EstimateErrs(c50_context *Context, Tree T, CaseNo Fp, CaseNo Lp,
			 int Sh, int Flags);
void	    GlobalPrune(c50_context *Context, Tree T);
void	    FindMinCC(Tree T);
void	    InsertParents(c50_context *Context, Tree T, Tree P);
void	    CheckSubsets(c50_context *Context, Tree T, Boolean);
void	    InitialiseExtraErrs(c50_context *Context);
float	    ExtraErrs(c50_context *Context, CaseCount N, CaseCount E,
		      ClassNo C);
float	    RawExtraErrs(c50_context *Context, CaseCount N, CaseCount E);
void	    RestoreDistribs(c50_context *Context, Tree T);
void	    CompressBranches(c50_context *Context, Tree T);
void	    SetGlobalUnitWeights(c50_context *Context, int LocalFlag);

	/* p-thresh.c */

void	    SoftenThresh(c50_context *Context, Tree T);
void	    ResubErrs(c50_context *Context, Tree T, CaseNo Fp, CaseNo Lp);
void	    FindBounds(c50_context *Context, Tree T, CaseNo Fp, CaseNo Lp);

	/* classify.c */

ClassNo	    TreeClassify(c50_context *Context, DataRec Case,
			 Tree DecisionTree);
void	    FollowAllBranches(c50_context *Context, DataRec Case, Tree T,
			      float Fraction);
ClassNo	    RuleClassify(c50_context *Context, DataRec Case, CRuleSet RS);
int	    FindOutcome(c50_context *Context, DataRec Case,
			Condition OneCond);
Boolean	    Matches(c50_context *Context, CRule R, DataRec Case);
void	    CheckActiveSpace(c50_context *Context, int N);
void	    MarkActive(c50_context *Context, RuleTree RT, DataRec Case);
void	    SortActive(c50_context *Context);
void	    CheckUtilityBand(c50_context *Context, int *u, RuleNo r,
			     ClassNo Actual, ClassNo Default);
ClassNo	    BoostClassify(c50_context *Context, DataRec Case, int MaxTrial);
ClassNo	    SelectClass(c50_context *Context, ClassNo Default,
			Boolean UseCosts);
ClassNo	    Classify(c50_context *Context, DataRec Case);
float	    Interpolate(Tree T, ContValue Val);

	/* special case for dual-purpose routines  */

void	    FindLeaf(c50_context *Context, DataRec Case, Tree T, Tree PT,
		     float Wt);
Boolean	    Satisfies(c50_context *Context, DataRec Case,
		      Condition OneCond);

	/* sort.c */

void	    Quicksort(c50_context *Context, CaseNo Fp, CaseNo Lp,
		      Attribute Att);
void	    Cachesort(CaseNo Fp, CaseNo Lp, SortRec *SRec);

	/* trees.c */

void	    FindDepth(Tree T);
void	    PrintTree(c50_context *Context, Tree T, String Title);
void	    Show(c50_context *Context, Tree T, int Sh);
void	    ShowBranch(c50_context *Context, int Sh, Tree T, DiscrValue v,
		       DiscrValue BrNo);
DiscrValue  Elements(c50_context *Context, Attribute Att, Set S,
		     DiscrValue *Last);
int	    MaxLine(c50_context *Context, Tree SubTree);
void	    Indent(int Sh, int BrNo);
void	    FreeTree(Tree T);
Tree	    Leaf(c50_context *Context, double *Freq, ClassNo NodeClass,
		 CaseCount Cases, CaseCount Errors);
void	    Sprout(Tree T, DiscrValue Branches);
void	    UnSprout(Tree T);
int	    TreeSize(Tree T);
int	    ExpandedLeafCount(c50_context *Context, Tree T);
int	    TreeDepth(Tree T);
Tree	    CopyTree(c50_context *Context, Tree T);

	/* utility.c */

void	    PrintHeader(String Title);
char	    ProcessOption(int Argc, char **Argv, char *Str);
void	    *Pmalloc(size_t Bytes);
void	    *Prealloc(void *Present, size_t Bytes);
void	    *Pcalloc(size_t Number, unsigned int Size);
void	    FreeVector(void **V, int First, int Last);
DataRec	    NewCase(c50_context *Context);
void	    FreeCases(c50_context *Context);
void	    FreeLastCase(c50_context *Context, DataRec Case);
void	    Error(int ErrNo, String S1, String S2);
void	    ErrorContext(c50_context *Context, int ErrNo, String S1,
			 String S2);
void	    C50Exit(int Status);
String	    CaseLabel(c50_context *Context, CaseNo N);
FILE *	    GetFile(String Extension, String RW);
double	    ExecTime(void);
int	    Denominator(ContValue Val);
int	    GetInt(String S, int N);
int	    DateToDay(String DS);
void	    DayToDate(int DI, String Date);
int	    TimeToSecs(String TS);
void	    SecsToTime(int Secs, String Time);
void	    SetTSBase(int y);
int	    TStampToMins(String TS);
void	    Check(float Val, float Low, float High);
void	    CValToStr(c50_context *Context, ContValue CV, Attribute Att,
		      String DS);
double	    rint(double v);
void	    Cleanup(c50_context *Context);
#ifdef UTF8
int	    UTF8CharWidth(unsigned char *U);
int	    wcwidth(wchar_t ucs);
int	    wcswidth(const wchar_t *pwcs, size_t n);
#endif

	/* confmat.c */

void	    PrintConfusionMatrix(c50_context *Context, CaseNo *ConfusionMat);
void	    PrintErrorBreakdown(c50_context *Context, CaseNo *ConfusionMat);
void	    PrintUsageInfo(c50_context *Context, CaseNo *Usage);

	/* formrules.c */

CRuleSet    FormRules(c50_context *Context, Tree T);
void	    Scan(c50_context *Context, Tree T);
void	    SetupNCost(c50_context *Context);
void	    PushCondition(c50_context *Context);
void	    PopCondition(c50_context *Context);
void	    PruneRule(c50_context *Context, Condition Cond[],
		      ClassNo TargetClass);
void	    ProcessLists(c50_context *Context);
void	    AddToList(CaseNo *List, CaseNo N);
void	    DeleteFromList(CaseNo *Before, CaseNo N);
int	    SingleFail(CaseNo i);
void	    Increment(c50_context *Context, int d, CaseNo i,
		      double *Total, double *Errors);
void	    FreeFormRuleData(void);

	/* rules.c */

Boolean	    NewRule(c50_context *Context, Condition Cond[], int NConds,
		    ClassNo TargetClass,
		    Boolean *Deleted, CRule Existing,
		    CaseCount Cover, CaseCount Correct, float Prior);
void	    ListSort(int *L, int Fp, int Lp);
Byte	    *Compress(int *L);
void	    Uncompress(Byte *CL, int *UCL);
Boolean	    SameRule(c50_context *Context, RuleNo r, Condition Cond[],
		     int NConds, ClassNo TargetClass);
void	    FreeRule(CRule R);
void	    FreeRules(CRuleSet RS);
void	    PrintRules(c50_context *Context, CRuleSet, String);
void	    PrintRule(c50_context *Context, CRule R);
void	    PrintCondition(c50_context *Context, Condition C);

	/* siftrules.c */

void	    SiftRules(c50_context *Context, float EstErrRate);
void	    InvertFires(c50_context *Context);
void	    FindTestCodes(c50_context *Context);
float	    CondBits(c50_context *Context, Condition C);
void	    SetInitialTheory(c50_context *Context);
void	    CoverClass(c50_context *Context, ClassNo Target);
double	    MessageLength(c50_context *Context, RuleNo NR, double RuleBits,
			  float Errs);
void	    HillClimb(c50_context *Context);
void	    InitialiseVotes(c50_context *Context);
void	    CountVotes(c50_context *Context, CaseNo i);
void	    UpdateDeltaErrs(c50_context *Context, CaseNo i, double Delta,
			    RuleNo Toggle);
CaseCount   CalculateDeltaErrs(c50_context *Context);
void	    PruneSubsets(c50_context *Context);
void	    SetDefaultClass(c50_context *Context);
void	    SwapRule(c50_context *Context, RuleNo A, RuleNo B);
int	    OrderByUtility(c50_context *Context);
int	    OrderByClass(c50_context *Context);
void	    OrderRules(c50_context *Context);
void	    GenerateLogs(int MaxN);
void	    FreeSiftRuleData(c50_context *Context);

	/* ruletree.c */

void	    ConstructRuleTree(c50_context *Context, CRuleSet RS);
void	    SetTestIndex(c50_context *Context, Condition C);
RuleTree    GrowRT(c50_context *Context, RuleNo *RR, int RRN, CRule *Rule);
int	    DesiredOutcome(c50_context *Context, CRule R, int TI);
int	    SelectTest(RuleNo *RR, int RRN, CRule *Rule);
void	    FreeRuleTree(RuleTree RT);

	/* modelfiles.c */

void	    CheckFile(c50_context *Context, String Extension, Boolean Write);
void	    WriteFilePrefix(c50_context *Context, String Extension);
void	    ReadFilePrefix(c50_context *Context, String Extension);
void	    SaveDiscreteNames(c50_context *Context);
void	    SaveTree(c50_context *Context, Tree T, String Extension);
void	    OutTree(c50_context *Context, Tree T);
void	    SaveRules(c50_context *Context, CRuleSet RS, String Extension);
void	    AsciiOut(String Pre, String S);
void	    ReadHeader(c50_context *Context, c50_input *Input);
void	    ReadHeaderMemory(c50_context *Context, c50_input *Input,
			     c50_input *CostsInput);
Tree	    GetTree(c50_context *Context, String Extension);
Tree	    InTree(c50_context *Context, c50_input *Input);
Tree	    InTreeAt(c50_context *Context, c50_input *Input, Tree *Slot);
CRuleSet    GetRules(c50_context *Context, String Extension);
CRuleSet    InRules(c50_context *Context, c50_input *Input);
CRuleSet    InRulesAt(c50_context *Context, c50_input *Input,
		      CRuleSet *Slot);
CRule	    InRule(c50_context *Context, c50_input *Input);
CRule	    InRuleAt(c50_context *Context, c50_input *Input, CRule *Slot);
Condition   InCondition(c50_context *Context, c50_input *Input);
Condition   InConditionAt(c50_context *Context, c50_input *Input,
			  Condition *Slot);
int	    ReadProp(c50_context *Context, c50_input *Input, char *Delim);
String	    RemoveQuotes(String S);
Set	    MakeSubset(c50_context *Context, Attribute Att);
void	    StreamIn(c50_input *Input, String S, int n);

	/* update.c (Unix) or winmain.c (WIN32) */

void	    NotifyStage(int);
void	    Progress(float);

	/* xval.c */

void	    CrossVal(c50_context *Context);
void	    Prepare(c50_context *Context);
void	    Shuffle(c50_context *Context, int *Vec);
void	    Summary(c50_context *Context);
float	    SE(float sum, float sumsq, int no);
