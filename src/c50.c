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
/*	Main routine, C5.0						 */
/*	------------------						 */
/*									 */
/*************************************************************************/


#include "defns.i"
#include "extern.i"
#include "c50_api_internal.h"
#include <signal.h>

#include <sys/unistd.h>
#include <sys/time.h>
#include <sys/resource.h>

#define SetFOpt(V)	V = strtod(OptArg, &EndPtr);\
			if ( ! EndPtr || *EndPtr != '\00' ) break;\
			ArgOK = true
#define SetIOpt(V)	V = strtol(OptArg, &EndPtr, 10);\
			if ( ! EndPtr || *EndPtr != '\00' ) break;\
			ArgOK = true


int main(int Argc, char *Argv[])
/*  ----  */
{
    int			o;
    extern String	OptArg, Option;
    char		*EndPtr;
    Boolean		FirstTime=true, ArgOK;
    double		StartTime;
    FILE		*F;
    c50_input		NamesInput;
    CaseNo		SaveMaxCase;
    Attribute		Att;
    c50_context	*Context = NULL;

    struct rlimit RL;

    if ( c50_context_create(&Context) != C50_STATUS_OK ) return 1;

    /*  Make sure there is a largish runtime stack  */

    getrlimit(RLIMIT_STACK, &RL);

    RL.rlim_cur = Max(RL.rlim_cur, 20 * 1024 * 1024);

    if ( RL.rlim_max > 0 )	/* -1 if unlimited */
    {
	RL.rlim_cur = Min(RL.rlim_max, RL.rlim_cur);
    }

    setrlimit(RLIMIT_STACK, &RL);


    /*  Check for output to be saved to a file  */

    if ( Argc > 2 && ! strcmp(Argv[Argc-2], "-o") )
    {
	Of = fopen(Argv[Argc-1], "w");
	Argc -= 2;
    }

    if ( ! Of )
    {
	Of = stdout;
    }

    KRInit = time(0) & 07777;

    PrintHeader("");

    /*  Process options  */

    while ( (o = ProcessOption(Argc, Argv, "f+bpv+t+sm+c+S+I+ru+egX+wh")) )
    {
	if ( FirstTime )
	{
	    fprintf(Of, T_OptHeader);
	    FirstTime = false;
	}

	ArgOK = false;

	switch (o)
	{
	case 'f':   FileStem = OptArg;
		    fprintf(Of, T_OptApplication, FileStem);
		    ArgOK = true;
		    break;
	case 'b':   Context->options.boosting = true;
		    fprintf(Of, T_OptBoost);
		    if ( Context->options.trials == 1 ) Context->options.trials = 10;
		    ArgOK = true;
		    break;
	case 'p':   Context->options.probabilistic_thresholds = true;
		    fprintf(Of, T_OptProbThresh);
		    ArgOK = true;
		    break;
#ifdef VerbOpt
	case 'v':   SetIOpt(Context->options.verbosity);
		    fprintf(Of, "\tVerbosity level %d\n", Context->options.verbosity);
		    ArgOK = true;
		    break;
#endif
	case 't':   SetIOpt(Context->options.trials);
		    fprintf(Of, T_OptTrials, Context->options.trials);
		    Check(Context->options.trials, 3, 1000);
		    Context->options.boosting = true;
		    break;
	case 's':   Context->options.subset_splits = true;
		    fprintf(Of, T_OptSubsets);
		    ArgOK = true;
		    break;
	case 'm':   SetFOpt(Context->options.minimum_cases);
		    fprintf(Of, T_OptMinCases, Context->options.minimum_cases);
		    Check(Context->options.minimum_cases, 1, 1000000);
		    break;
	case 'c':   SetFOpt(Context->options.confidence_factor);
		    fprintf(Of, T_OptCF, Context->options.confidence_factor);
		    Check(Context->options.confidence_factor, 0, 100);
		    Context->options.confidence_factor /= 100;
		    break;
	case 'r':   Context->options.rules = true;
		    fprintf(Of, T_OptRules);
		    ArgOK = true;
		    break;
	case 'S':   SetFOpt(Context->options.sample_fraction);
		    fprintf(Of, T_OptSampling, Context->options.sample_fraction);
		    Check(Context->options.sample_fraction, 0.1, 99.9);
		    Context->options.sample_fraction /= 100;
		    break;
	case 'I':   SetIOpt(KRInit);
		    fprintf(Of, T_OptSeed, KRInit);
		    KRInit = KRInit & 07777;
		    break;
	case 'u':   SetIOpt(Context->options.utility_bands);
		    fprintf(Of, T_OptUtility, Context->options.utility_bands);
		    Check(Context->options.utility_bands, 2, 10000);
		    Context->options.rules = true;
		    break;
	case 'e':   Context->options.ignore_costs = true;
		    fprintf(Of, T_OptNoCosts);
		    ArgOK = true;
		    break;
	case 'w':   Context->options.winnow = true;
		    fprintf(Of, T_OptWinnow);
		    ArgOK = true;
		    break;
	case 'g':   Context->options.global_pruning = false;
		    fprintf(Of, T_OptNoGlobal);
		    ArgOK = true;
		    break;
	case 'X':   SetIOpt(Context->options.folds);
		    fprintf(Of, T_OptXval, Context->options.folds);
		    Check(Context->options.folds, 2, 1000);
		    Context->options.cross_validation = true;
		    break;
	}

	if ( ! ArgOK )
	{
	    if ( o != 'h' )
	    {
		fprintf(Of, T_UnregnizedOpt,
			    Option,
			    ( ! OptArg || OptArg == Option+2 ? "" : OptArg ));
		fprintf(Of, T_SummaryOpts);
	    }
	    fprintf(Of, T_ListOpts);
	    Goodbye(1);
	}
    }

    if ( Context->options.utility_bands && Context->options.boosting )
    {
	fprintf(Of, T_UBWarn);
    }

    StartTime = ExecTime();

    /*  Get information on training data  */

    if ( ! (F = GetFile(".names", "r")) ) Error(NOFILE, "", "");
    c50_input_init_file(&NamesInput, F);
    GetNames(Context, &NamesInput);
    fclose(F);

    if ( Context->schema.class_attribute )
    {
	fprintf(Of, T_ClassVar, Context->schema.attribute_names[Context->schema.class_attribute]);
    }

    NotifyStage(READDATA);
    Progress(-1.0);

    /*  Allocate space for Context->cases.some_missing[] and Context->cases.some_not_applicable[] */

    Context->cases.some_missing = AllocZero(Context->schema.max_attribute+1, Boolean);
    Context->cases.some_not_applicable   = AllocZero(Context->schema.max_attribute+1, Boolean);

    /*  Read data file  */

    if ( ! (F = GetFile(".data", "r")) ) Error(NOFILE, "", "");
    GetData(Context, F, true, false);
    fprintf(Of, TX_ReadData(Context->cases.max_case+1, Context->schema.max_attribute, FileStem));

    if ( Context->options.cross_validation && (F = GetFile(".test", "r")) )
    {
	SaveMaxCase = Context->cases.max_case;
	GetData(Context, F, false, false);
	fprintf(Of, TX_ReadTest(Context->cases.max_case-SaveMaxCase, FileStem));
    }

    /*  Check whether case weight attribute appears  */

    if ( Context->schema.case_weight_attribute )
    {
	fprintf(Of, T_CWtAtt);
    }

    if ( ! Context->options.ignore_costs && (F = GetFile(".costs", "r")) )
    {
	GetMCosts(Context, F);
	if ( Context->costs.matrix )
	{
	    fprintf(Of, T_ReadCosts, FileStem);
	}
    }

    /*  Note any attribute exclusions/inclusions  */

    if ( AttExIn )
    {
	fprintf(Of, "%s", ( AttExIn == -1 ? T_AttributesOut : T_AttributesIn ));

	ForEach(Att, 1, Context->schema.max_attribute)
	{
	    if ( Att != Context->schema.class_attribute &&
		 Att != Context->schema.case_weight_attribute &&
		 ( StatBit(Att, SKIP) > 0 ) == ( AttExIn == -1 ) )
	    {
		fprintf(Of, "    %s\n", Context->schema.attribute_names[Att]);
	    }
	}
    }

    /*  Build decision trees  */

    if ( ! Context->options.boosting )
    {
	Context->options.trials = 1;
    }

    InitialiseTreeData(Context);
    if ( Context->options.rules )
    {
	Context->rules.sets = AllocZero(Context->options.trials+1, CRuleSet);
    }

    if ( Context->options.winnow )
    {
	NotifyStage(WINNOWATTS);
	Progress(-Context->schema.max_attribute);
	WinnowAtts(Context);
    }

    if ( Context->options.cross_validation )
    {
	CrossVal(Context);
    }
    else
    {
	ConstructClassifiers(Context);

	/*  Evaluation  */

	fprintf(Of, T_EvalTrain, Context->cases.max_case+1);

	NotifyStage(EVALTRAIN);
	Progress(-Context->options.trials * (Context->cases.max_case+1.0));

	Evaluate(Context, CMINFO | USAGEINFO);

	if ( (F = GetFile(( Context->options.sample_fraction ? ".data" : ".test" ), "r")) )
	{
	    NotifyStage(READTEST);
	    fprintf(Of, "\n");

	    FreeData(Context);
	    GetData(Context, F, false, false);

	    fprintf(Of, T_EvalTest, Context->cases.max_case+1);

	    NotifyStage(EVALTEST);
	    Progress(-Context->options.trials * (Context->cases.max_case+1.0));

	    Evaluate(Context, CMINFO);
	}
    }

    fprintf(Of, T_Time, ExecTime() - StartTime);

#ifdef VerbOpt
    Cleanup(Context);
#endif

    c50_context_destroy(Context);

    return 0;
}
