/* Modified 2026 by Geoffrey Mainland: native C++ library integration. */
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

#include <cerrno>
#include <climits>
#include <cmath>
#include <limits>
#include <type_traits>

#if !defined(_WIN32)
#include <sys/resource.h>
#endif

// A value that is not a number at all, or that has trailing text, ends the
// option list as an unrecognized option, as in Release 2.07. Empty text is the
// number 0, because Release 2.07 only tests the end pointer that strtod and
// strtol leave on the terminating null. A number that overflows its type or is
// not finite reports the same "illegal value" diagnostic as a number outside
// the option's range. A number that underflows is its rounded value.
enum class OptionValue { Valid, Malformed, OutOfRange };


namespace
{

// Release 2.07 prints with %g, which keeps six digits. An integer option's
// bounds and value need up to ten digits to differ, as -I 2147483648 does.

void IllegalValue(c50_context *Context, double Value, double Low, double High,
                  bool Integer)
{
    if ( Integer )
    {
        fprintf(Context->io.output,
                "\t** illegal value %.10g -- should be between %.10g and %.10g\n",
                Value, Low, High);
    }
    else
    {
        fprintf(Context->io.output, TX_IllegalValue(Value, Low, High));
    }
    C50Exit(Context, 1);
}

OptionValue ParseOption(const char *Text, float &Value, double &Shown)
{
    char *End;
    errno = 0;
    const double Parsed = strtod(Text, &End);
    if ( *End ) return OptionValue::Malformed;
    Shown = Parsed;

    // ERANGE with a result no larger than the smallest normal double is
    // underflow, which leaves a zero or subnormal value that is not an error.

    if ( ( errno == ERANGE &&
           std::fabs(Parsed) > std::numeric_limits<double>::min() ) ||
         ! isfinite(Parsed) ||
         Parsed < -std::numeric_limits<float>::max() ||
         Parsed > std::numeric_limits<float>::max() )
    {
        return OptionValue::OutOfRange;
    }
    Value = static_cast<float>(Parsed);
    return OptionValue::Valid;
}

OptionValue ParseOption(const char *Text, int &Value, double &Shown)
{
    char *End;
    errno = 0;
    const long Parsed = strtol(Text, &End, 10);
    if ( *End ) return OptionValue::Malformed;
    Shown = static_cast<double>(Parsed);
    if ( errno == ERANGE ||
         Parsed < std::numeric_limits<int>::min() ||
         Parsed > std::numeric_limits<int>::max() )
    {
        // An overflowing value reports its own magnitude.
        if ( errno == ERANGE ) Shown = strtod(Text, nullptr);
        return OptionValue::OutOfRange;
    }
    Value = static_cast<int>(Parsed);
    return OptionValue::Valid;
}

// Reads the value of the current option into V. Returns false for a value that
// is malformed, which ends the option list as an unrecognized option, and does
// not return for a value outside the range printed in the diagnostic.

template <typename T>
bool SetOption(c50_context *Context, T &V, double Low, double High)
{
    double Shown;
    const OptionValue Status = ParseOption(Context->io.option_argument, V, Shown);

    if ( Status == OptionValue::Malformed ) return false;
    if ( Status == OptionValue::OutOfRange )
    {
        IllegalValue(Context, Shown, Low, High, std::is_integral<T>::value);
    }
    return true;
}

struct cli_arguments
{
    int argc;
    char **argv;
};

void Run(c50_context *Context, void *UserData)
/*   ---  */
{
    cli_arguments	*Arguments = static_cast<cli_arguments *>(UserData);
    int			Argc = Arguments->argc;
    char			**Argv = Arguments->argv;
    int			o;
    Boolean		FirstTime=true, ArgOK;
    double		StartTime;
    FILE		*F;
    c50_input		NamesInput;
    CaseNo		SaveMaxCase;
    Attribute		Att;

#if !defined(_WIN32)
    struct rlimit RL;

    /*  Make sure there is a largish runtime stack  */

    getrlimit(RLIMIT_STACK, &RL);

    RL.rlim_cur = Max(RL.rlim_cur, 20 * 1024 * 1024);

    if ( RL.rlim_max > 0 )	/* -1 if unlimited */
    {
	RL.rlim_cur = Min(RL.rlim_max, RL.rlim_cur);
    }

    setrlimit(RLIMIT_STACK, &RL);
#endif


    /*  Check for output to be saved to a file  */

    if ( Argc > 2 && ! strcmp(Argv[Argc-2], "-o") )
    {
	Context->io.output = fopen(Argv[Argc-1], "w");
	Argc -= 2;
    }

    if ( ! Context->io.output )
    {
	Context->io.output = stdout;
    }

    Context->io.random_initial_seed = time(0) & 07777;

    PrintHeader(Context, "");

    /*  Process options  */

    while ( (o = ProcessOption(Context, Argc, Argv, "f+bpv+t+sm+c+S+I+ru+egX+wh")) )
    {
	if ( FirstTime )
	{
	    fprintf(Context->io.output, T_OptHeader);
	    FirstTime = false;
	}

	ArgOK = false;

	switch (o)
	{
	case 'f':   Context->io.file_stem = Context->io.option_argument;
		    fprintf(Context->io.output, T_OptApplication, Context->io.file_stem);
		    ArgOK = true;
		    break;
	case 'b':   Context->options.boosting = true;
		    fprintf(Context->io.output, T_OptBoost);
		    if ( Context->options.trials == 1 ) Context->options.trials = 10;
		    ArgOK = true;
		    break;
	case 'p':   Context->options.probabilistic_thresholds = true;
		    fprintf(Context->io.output, T_OptProbThresh);
		    ArgOK = true;
		    break;
#ifdef VerbOpt
	case 'v':   if ( ! SetOption(Context, Context->options.verbosity, INT_MIN, INT_MAX) ) break;
		    ArgOK = true;
		    fprintf(Context->io.output, "\tVerbosity level %d\n", Context->options.verbosity);
		    break;
#endif
	case 't':   if ( ! SetOption(Context, Context->options.trials, 3, 1000) ) break;
		    ArgOK = true;
		    fprintf(Context->io.output, T_OptTrials, Context->options.trials);
		    Check(Context, Context->options.trials, 3, 1000);
		    Context->options.boosting = true;
		    break;
	case 's':   Context->options.subset_splits = true;
		    fprintf(Context->io.output, T_OptSubsets);
		    ArgOK = true;
		    break;
	case 'm':   if ( ! SetOption(Context, Context->options.minimum_cases, 1, 1000000) ) break;
		    ArgOK = true;
		    fprintf(Context->io.output, T_OptMinCases, Context->options.minimum_cases);
		    Check(Context, Context->options.minimum_cases, 1, 1000000);
		    break;
	case 'c':   if ( ! SetOption(Context, Context->options.confidence_factor, 0, 100) ) break;
		    ArgOK = true;
		    fprintf(Context->io.output, T_OptCF, Context->options.confidence_factor);
		    Check(Context, Context->options.confidence_factor, 0, 100);
		    Context->options.confidence_factor /= 100;
		    break;
	case 'r':   Context->options.rules = true;
		    fprintf(Context->io.output, T_OptRules);
		    ArgOK = true;
		    break;
	case 'S':   if ( ! SetOption(Context, Context->options.sample_fraction, 0.1, 99.9) ) break;
		    ArgOK = true;
		    fprintf(Context->io.output, T_OptSampling, Context->options.sample_fraction);
		    Check(Context, Context->options.sample_fraction, 0.1, 99.9);
		    Context->options.sample_fraction /= 100;
		    break;
	case 'I':   if ( ! SetOption(Context, Context->io.random_initial_seed, INT_MIN, INT_MAX) ) break;
		    ArgOK = true;
		    fprintf(Context->io.output, T_OptSeed, Context->io.random_initial_seed);
		    Context->io.random_initial_seed = Context->io.random_initial_seed & 07777;
		    break;
	case 'u':   if ( ! SetOption(Context, Context->options.utility_bands, 2, 10000) ) break;
		    ArgOK = true;
		    fprintf(Context->io.output, T_OptUtility, Context->options.utility_bands);
		    Check(Context, Context->options.utility_bands, 2, 10000);
		    Context->options.rules = true;
		    break;
	case 'e':   Context->options.ignore_costs = true;
		    fprintf(Context->io.output, T_OptNoCosts);
		    ArgOK = true;
		    break;
	case 'w':   Context->options.winnow = true;
		    fprintf(Context->io.output, T_OptWinnow);
		    ArgOK = true;
		    break;
	case 'g':   Context->options.global_pruning = false;
		    fprintf(Context->io.output, T_OptNoGlobal);
		    ArgOK = true;
		    break;
	case 'X':   if ( ! SetOption(Context, Context->options.folds, 2, 1000) ) break;
		    ArgOK = true;
		    fprintf(Context->io.output, T_OptXval, Context->options.folds);
		    Check(Context, Context->options.folds, 2, 1000);
		    Context->options.cross_validation = true;
		    break;
	}

	if ( ! ArgOK )
	{
	    if ( o != 'h' )
	    {
		fprintf(Context->io.output, T_UnregnizedOpt,
			    Context->io.option,
			    ( ! Context->io.option_argument || Context->io.option_argument == Context->io.option+2 ? "" : Context->io.option_argument ));
		fprintf(Context->io.output, T_SummaryOpts);
	    }
	    fprintf(Context->io.output, T_ListOpts);
	    Goodbye(1);
	}
    }

    if ( Context->options.utility_bands && Context->options.boosting )
    {
	fprintf(Context->io.output, T_UBWarn);
    }

    StartTime = ExecTime();

    /*  Get information on training data  */

    if ( ! (F = GetFile(Context, ".names", "r")) ) Error(Context, NOFILE, "", "");
    c50_input_init_file(&NamesInput, F);
    GetNames(Context, &NamesInput);
    fclose(F);

    if ( Context->schema.class_attribute )
    {
	fprintf(Context->io.output, T_ClassVar, Context->schema.attribute_names[Context->schema.class_attribute]);
    }

    NotifyStage(Context, READDATA);
    Progress(Context, -1.0);

    /*  Allocate space for Context->cases.some_missing[] and Context->cases.some_not_applicable[] */

    Context->cases.some_missing = AllocZero(Context->schema.max_attribute+1, Boolean);
    Context->cases.some_not_applicable   = AllocZero(Context->schema.max_attribute+1, Boolean);

    /*  Read data file  */

    if ( ! (F = GetFile(Context, ".data", "r")) ) Error(Context, NOFILE, "", "");
    GetData(Context, F, true, false);
    fprintf(Context->io.output, TX_ReadData(Context->cases.max_case+1, Context->schema.max_attribute, Context->io.file_stem));

    if ( Context->options.cross_validation && (F = GetFile(Context, ".test", "r")) )
    {
	SaveMaxCase = Context->cases.max_case;
	GetData(Context, F, false, false);
	fprintf(Context->io.output, TX_ReadTest(Context->cases.max_case-SaveMaxCase, Context->io.file_stem));
    }

    /*  Check whether case weight attribute appears  */

    if ( Context->schema.case_weight_attribute )
    {
	fprintf(Context->io.output, T_CWtAtt);
    }

    if ( ! Context->options.ignore_costs && (F = GetFile(Context, ".costs", "r")) )
    {
	GetMCosts(Context, F);
	if ( Context->costs.matrix )
	{
	    fprintf(Context->io.output, T_ReadCosts, Context->io.file_stem);
	}
    }

    /*  Note any attribute exclusions/inclusions  */

    if ( Context->io.attribute_exclusions )
    {
	fprintf(Context->io.output, "%s", ( Context->io.attribute_exclusions == -1 ? T_AttributesOut : T_AttributesIn ));

	ForEach(Att, 1, Context->schema.max_attribute)
	{
	    if ( Att != Context->schema.class_attribute &&
		 Att != Context->schema.case_weight_attribute &&
		 ( StatBit(Att, SKIP) > 0 ) == ( Context->io.attribute_exclusions == -1 ) )
	    {
		fprintf(Context->io.output, "    %s\n", Context->schema.attribute_names[Att]);
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
	NotifyStage(Context, WINNOWATTS);
	Progress(Context, -Context->schema.max_attribute);
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

	fprintf(Context->io.output, T_EvalTrain, Context->cases.max_case+1);

	NotifyStage(Context, EVALTRAIN);
	Progress(Context, -Context->options.trials * (Context->cases.max_case+1.0));

	Evaluate(Context, CMINFO | USAGEINFO);

	if ( (F = GetFile(Context, ( Context->options.sample_fraction ? ".data" : ".test" ), "r")) )
	{
	    NotifyStage(Context, READTEST);
	    fprintf(Context->io.output, "\n");

	    FreeData(Context);
	    GetData(Context, F, false, false);

	    fprintf(Context->io.output, T_EvalTest, Context->cases.max_case+1);

	    NotifyStage(Context, EVALTEST);
	    Progress(Context, -Context->options.trials * (Context->cases.max_case+1.0));

	    Evaluate(Context, CMINFO);
	}
    }

    fprintf(Context->io.output, T_Time, ExecTime() - StartTime);
}

void CleanupCli(c50_context *Context, void *UserData)
/*   ----------  */
{
    (void) UserData;
    Cleanup(Context);
}

} // namespace

int main(int Argc, char *Argv[])
/*  ----  */
{
    cli_arguments Arguments = {Argc, Argv};
    std::unique_ptr<c50_context> context;
    try
    {
        context = c50_make_context();
        context->continue_past_errors = 1;
        c50_run_operation(context.get(), Run, CleanupCli, &Arguments);
        return 0;
    }
    catch (const std::exception &error)
    {
        // Error and the option checks print their own diagnostics. Report
        // every other failure, which Release 2.07 had no way to raise, on the
        // stream that Error uses.
        if ( ! context || ! context->failure_reported )
        {
            std::fprintf(context && context->io.output ? context->io.output :
                         stderr, "\n*** %s\n", error.what());
        }
        return 1;
    }
}
